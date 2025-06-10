import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestCentroid
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from AMLCN_new import *
import shap
import matplotlib.pyplot as plt
import seaborn as sns  # 用于绘制热力图
from sklearn.metrics import confusion_matrix

class ProtoNet(nn.Module):
    def __init__(self, feature_dim=81, num_classes=15, num_prototypes=15):
        super(ProtoNet, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))  # 特征维度为81

    def forward(self, x):
        # 计算样本与原型之间的距离
        proto_dist = torch.cdist(x, self.prototypes.unsqueeze(0)).squeeze(0)
        return proto_dist

def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 如果模型返回的是一个 tuple，取第一个元素
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    return acc, f1, precision, recall

# 加载数据
def load_data_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    features = data.iloc[:, :-1].values  # 前 n-1 列为特征
    labels = data.iloc[:, -1].values  # 最后一列为标签
    return features, labels

# 不确定性采样
def uncertainty_sampling(alcn_model, data, threshold=0.5, batch_size=1000):
    """
    基于不确定性采样选择最不确定的样本。
    """
    alcn_model.eval()# 将模型设置为评估模式，禁用 dropout 和 batch normalization 的训练行为
    dataloader = DataLoader(data, batch_size=batch_size)# 创建数据加载器以按批处理数据
    uncertain_indices = []
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(next(alcn_model.parameters()).device)
            outputs = alcn_model(inputs)# 通过模型进行前向传播，获取预测输出

            # 处理模型可能返回的元组输出，通常 AMLCN 模型会返回 (final_output, layer_outputs)
            # 我们只关心最终的预测输出
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            confidence_scores, _ = torch.max(outputs, dim=1)
            # 找到置信度超过阈值下限的样本的索引，
            #之所以要选择置信度低于阈值的样本生成伪标签，主要目的在于通过动态筛选模型预测不确定性高的样本（即位于决策边界附近的“困难样本”），
            # 低置信度样本往往反映了模型当前分类边界的模糊区域，对其标注伪标签并训练可有效引导模型修正潜在错误决策面。
            batch_uncertain = (confidence_scores < threshold).nonzero(as_tuple=True)[0]
            uncertain_indices.extend(batch_uncertain.cpu().numpy())
        print(f"Number of uncertain samples: {len(uncertain_indices)}")  # 输出不确定样本的数量
    return uncertain_indices

def incremental_learning(alcn_model, proto_model, train_data, batch_size, device, epochs=1):
    """
    增量学习阶段，更新 ALCN 模型（包括特征提取器和分类器）。
    """
    alcn_model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # 分离特征提取器和分类器的优化器
    optimizer_fe = optim.Adam(
        [param for name, param in alcn_model.named_parameters() if "classifiers" not in name], lr=0.001
    )
    optimizer_cls = optim.Adam(
        [param for name, param in alcn_model.named_parameters() if "classifiers" in name], lr=0.001
    )

    # ===== 先用初始训练数据拟合 ProtoNet 模型 =====
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            features.append(alcn_model.get_features(inputs).cpu())
            labels.append(targets.cpu())

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    # 初始化 ProtoNet 并训练
    proto_model.train()
    proto_optimizer = optim.Adam(proto_model.parameters(), lr=0.001)
    for _ in range(10):
        proto_optimizer.zero_grad()
        outputs = proto_model(torch.tensor(features, dtype=torch.float32).to(device))
        loss = criterion(outputs, torch.tensor(labels, dtype=torch.long).to(device))
        loss.backward()
        proto_optimizer.step()
    print("ProtoNet initial training finished.")  # 输出ProtoNet初始训练完成

    for epoch in range(epochs):
        alcn_model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            final_output, layer_outputs = alcn_model(inputs)
            final_loss = criterion(final_output, targets)
            train_feature_extractor(alcn_model, optimizer_fe, final_loss)
            for i, output in enumerate(layer_outputs):
                layer_loss = criterion(output, targets)
                train_layer_classifier(alcn_model, optimizer_cls, layer_loss, i)
        print(f"ALCN training epoch {epoch+1} finished.")  # 输出ALCN训练epoch完成

        # ===== 半监督机制：关键修分 =====
        # 1. 不确定性采样
        uncertain_indices = uncertainty_sampling(alcn_model, train_data, threshold=0.5, batch_size=batch_size)

        # 2. 提取原始输入数据
        uncertain_inputs = train_data[:][0][uncertain_indices].to(device)

        # 3. 使用 ALCN 提取特征，隐藏层最后一层的输出作为特征表达
        with torch.no_grad():
            uncertain_features = alcn_model.get_features(uncertain_inputs)

        # 4. 使用 ProtoNet 生成伪标签
        with torch.no_grad():
            proto_outputs = proto_model(uncertain_features)
        pseudo_labels = torch.argmax(proto_outputs, dim=1).cpu()

        # 5. 创建伪标签数据集（存储原始输入数据+伪标签）
        pseudo_dataset = TensorDataset(
            uncertain_inputs.cpu(),  # 原始输入（77维）
            pseudo_labels
        )

        # 6. 合并数据
        new_features = torch.cat([train_data[:][0].cpu(), pseudo_dataset[:][0]])
        new_labels = torch.cat([train_data[:][1].cpu(), pseudo_dataset[:][1]])
        train_dataset = TensorDataset(new_features, new_labels)

        # ===== 重新训练 ProtoNet =====
        extended_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        extended_features, extended_labels = [], []
        with torch.no_grad():
            for inputs, targets in extended_loader:
                inputs = inputs.to(device)
                extended_features.append(alcn_model.get_features(inputs).cpu())
                extended_labels.append(targets.cpu())

        extended_features = torch.cat(extended_features).numpy()
        extended_labels = torch.cat(extended_labels).numpy()

        proto_model.train()
        for _ in range(10):
            proto_optimizer.zero_grad()
            outputs = proto_model(torch.tensor(extended_features, dtype=torch.float32).to(device))
            loss = criterion(outputs, torch.tensor(extended_labels, dtype=torch.long).to(device))
            loss.backward()
            proto_optimizer.step()
        print("ProtoNet retraining finished.")  # 输出ProtoNet重新训练完成

def feature_importance_analysis(model, data_loader, save_path="feature_importance_2017.pdf"):
    """
    分析模型对特征的全局重要性，并将图像保存到指定路径
    """
    model.eval()

    # 获取一批数据
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 包装模型使其输出单一张量
    def model_final_output(x):
        # 如果输入是 numpy.ndarray，则转换为 Torch Tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        # 前向传播
        return model(x)[0].detach().cpu().numpy()

    # 使用 KernelExplainer
    explainer = shap.KernelExplainer(model_final_output, inputs.cpu().numpy())
    shap_values = explainer.shap_values(inputs.cpu().numpy(), nsamples=100)  # 调整 nsamples 控制计算速度

    # 保存特征重要性图
    plt.figure()  # 创建一个新图形
    shap.summary_plot(shap_values, inputs.cpu().numpy(), show=False)  # 不在屏幕显示图像
    plt.savefig(save_path, bbox_inches='tight')  # 保存图像到指定路径
    plt.close()  # 关闭图像，释放内存
    print(f"Feature importance analysis plot saved to {save_path}")  # 输出特征重要性分析图保存路径

def plot_confusion_matrix(model, data_loader, class_names, device, save_path="confusion_matrix.pdf"):
    """
    绘制模型的混淆矩阵热力图并保存到指定路径

    Args:
        model: 训练好的模型
        data_loader: 测试数据加载器
        class_names: 类别名称列表
        device: 运行设备（GPU/CPU）
        save_path: 混淆矩阵图像保存路径
    """
    model.eval()
    all_labels = []
    all_preds = []
    # all_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    # all_preds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    all_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    all_preds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    # 计算真实标签和预测标签
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 如果模型返回的是一个 tuple，取第一个元素
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)  # 获取预测类别
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(10, 8))  # 设置图像大小
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # 保存热力图到指定路径
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()  # 关闭图像，释放内存
    print(f"Confusion matrix plot saved to {save_path}")  # 输出混淆矩阵图保存路径

if __name__ == "__main__":
    # 参数设置
    input_dim = 77  # 特征数量
    hidden_dim = 128
    output_dim = 15  # 类别数量
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    features, labels = load_data_from_csv("./Dataset/CIC-IDS2017_combined.csv")
    train_features, test_features = features[:int(len(features) * 0.8)], features[int(len(features) * 0.8):]
    train_labels, test_labels = labels[:int(len(labels) * 0.8)], labels[int(len(labels) * 0.8):]

    train_dataset = TensorDataset(
        torch.tensor(train_features,  dtype=torch.float32).to(device),
        torch.tensor(train_labels,  dtype=torch.long).to(device)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_features,  dtype=torch.float32).to(device),
        torch.tensor(test_labels,  dtype=torch.long).to(device)
    )
    print("Data loaded.")  # 输出数据加载完成

    # 初始化 ALCN 模型和 ProtoNet
    alcn_model = AMLCN(input_dim, hidden_dim, output_dim, n_layers=12, n_classifiers=3)
    proto_model = ProtoNet(81, output_dim).to(device)
    print("Models initialized.")  # 输出模型初始化完成

    # 增量学习
    incremental_learning(alcn_model, proto_model, train_dataset, batch_size, device, epochs=10)
    print("Incremental learning finished.")  # 输出增量学习完成

    # 模型评估
    acc, f1, precision, recall = evaluate_model(alcn_model, DataLoader(test_dataset, batch_size=batch_size), device)
    print("Model evaluation results:")  # 输出模型评估结果
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # 绘制并保存混淆矩阵热力图
    plot_confusion_matrix(
        alcn_model,
        DataLoader(test_dataset, batch_size=batch_size),
        class_names=['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'FTP-Patator',
                     'SSH Patator', 'DoS slowloris', 'DoS Slowhttptest', 'Bot', 'Web Attack Brute Force',
                     'Web Attack XSS', 'Infiltration', 'Web Attack Sql Injection','Heartbleed'],
        # class_names=['BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP', 'DrDoS_NetBIOS',
        #              'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 'LDAP', 'MSSQL', 'NetBIOS',
        #              'Portmap', 'Syn', 'TFTP', 'UDP-lag', 'WebDDoS'],
        device=device,
        save_path="pic/confusion_matrix_cicids2017_pn.png"  # 自定义保存路径
    )

    # 分析特征重要性并保存到指定路径
    feature_importance_analysis(
        alcn_model,
        DataLoader(test_dataset, batch_size=batch_size),
        save_path="pic/shap_cicids2017_pn.png"  # 自定义保存路径
    )