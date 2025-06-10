import random
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from numpy.linalg import eig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import deque
from torch.optim.lr_scheduler import CosineAnnealingLR
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 假设这些模块已在其他地方定义或导入
from PCAbasedmodelreuse_mechanism import *
from AdaptiveMultiLayerClassifierNetwork import *

# 定义滑动窗口大小，因数据分布和类别的不同两个数据集的窗口超参数不一样,NB15前几百个窗口效果很好，是因为数据分布，具体看折线图
WINDOW_SIZE = 2000  # 窗口大小
STEP_SIZE = 1500  # 滑动窗口步长
#上面两个是关键超参数，后续可以优化
# 定义模型历史队列
MODEL_HISTORY_QUEUE_SIZE = 5  # 模型历史队列的最大长度

# 模型历史队列：存储模型和是否被 Reptile 微调的标志
model_history = deque(maxlen=MODEL_HISTORY_QUEUE_SIZE)
distribution_history = deque(maxlen=MODEL_HISTORY_QUEUE_SIZE)


def reptile_finetune(model, data_loader, device, lr=0.001, inner_steps=20):
    """
    使用 Reptile 方法对模型进行微调
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    initial_state = copy.deepcopy(model.state_dict())  # 记录初始模型参数

    for _ in range(inner_steps):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = FocalLoss(gamma=2)(outputs, targets)  # 假设使用 FocalLoss
            loss.backward()
            optimizer.step()

    # 更新模型权重：权重 = 初始权重 + lr * (当前权重 - 初始权重)
    updated_state = model.state_dict()
    for name, param in initial_state.items():
        updated_state[name] = param + lr * (updated_state[name] - param)

    model.load_state_dict(updated_state)  # 应用更新的权重
    return model


# 滑动窗口数据提取
def get_windowed_data(data, window_size, step_size):
    for start in range(0, len(data) - window_size + 1, step_size):
        yield data[start:start + window_size]


# 定义 FocalLoss 类，用于处理不平衡数据
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        logpt = torch.nn.functional.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = torch.nn.functional.nll_loss(logpt, target, weight=self.alpha, reduction=self.reduction)
        return loss

# 定义数据集类，用于加载特征和标签
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 模型保存函数
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 数据加载函数
def load_full_data(file_path):
    data = pd.read_csv(file_path)
    features = data.drop(columns=["attack_cat"]).values#2017“ Label”,NB15"attack_cat"
    labels = data["attack_cat"].values
    return features, labels

# 定义AMLCN类（也可以从AdaptiveMultiLayerClassifierNetwork中导入）
class AMLCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=12, n_classifiers=3, eta=0.1):
        super(AMLCN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        classifier_index = 0

        # 卷积层与池化层做初步的特征提取
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 经过卷积层和池化层后进行展平
        temp_input = torch.zeros(1, 1, input_dim)
        temp_output = self.pool(F.relu(self.conv1(temp_input)))
        flattened_dim = temp_output.numel()

        # 展平后的连接层
        self.hidden_layers.append(nn.Linear(flattened_dim, hidden_dim))

        # 继续添加隐藏层做特征提取
        for i in range(1, n_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

            # 每三个隐藏层后面都加上一个层分类器
            if (i + 1) % 3 == 0 and classifier_index < n_classifiers:
                self.classifiers.append(nn.Linear(hidden_dim, output_dim))
                classifier_index += 1

        self.softmax = nn.Softmax(dim=1)
        self.alpha = torch.ones(len(self.classifiers), requires_grad=False).to(device) / len(self.classifiers)
        self.eta = eta
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        classifier_outputs = []

        # Reshape input for Conv1d
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # 前向传播
        for i, hidden in enumerate(self.hidden_layers):
            x = F.relu(hidden(x))
            x = self.dropout(x)

            # 收集各个层分类器的输出
            if (i + 1) % 3 == 0 and (i + 1) // 3 - 1 < len(self.classifiers):
                classifier_outputs.append(self.classifiers[(i + 1) // 3 - 1](x))

        # 对层分类器的输出进行加权
        final_output = sum(a * o for a, o in zip(self.alpha, classifier_outputs))
        return final_output, classifier_outputs

    def update_alpha(self, losses):
        """
        根据每一层的损失值更新alpha权重，损失越大权重越小
        """
        losses = torch.tensor([loss.item() for loss in losses], dtype=torch.float32).to(device)
        inv_losses = 1 / (losses + 1e-8)  # 增加一个很小的epsilon防止除以0
        self.alpha = inv_losses / inv_losses.sum()

# 训练特征提取模块
def train_feature_extractor(model, optimizer, final_output_loss):
    optimizer.zero_grad()
    # 冻结所有分类器参数
    for param in model.classifiers.parameters():
        param.requires_grad = False
    # 解冻特征提取器参数
    for param in model.conv1.parameters():
        param.requires_grad = True
    for param in model.hidden_layers.parameters():
        param.requires_grad = True

    final_output_loss.backward()
    optimizer.step()

# 训练层分类器
def train_layer_classifier(model, optimizer, classifier_losses):
    # 冻结特征提取器参数
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.hidden_layers.parameters():
        param.requires_grad = False
    # 解冻所有分类器参数
    for param in model.classifiers.parameters():
        param.requires_grad = True

    for loss in classifier_losses:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 在训练批次结束后，统一解冻所有参数，方便下次迭代
def unfreeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

# 训练模型函数，综合上面的特征提取训练和层分类器训练
def train_model_on_data(model, data_loader, device, epochs=10):
    model.to(device)
    # 优化器应在每次参数状态改变（冻结/解冻）后重新初始化，或者管理好参数组
    # 优化器会正确地只更新requires_grad=True的参数
    optimizer_fe = optim.Adam(
        [param for name, param in model.named_parameters() if "classifiers" not in name], lr=0.001
    )
    optimizer_cls = optim.Adam(
        [param for name, param in model.named_parameters() if "classifiers" in name], lr=0.001
    )
    criterion = FocalLoss(gamma=2).to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # --- 步骤 1: 训练特征提取模块 ---
            # 冻结分类器参数，只更新特征提取器
            for param in model.classifiers.parameters():
                param.requires_grad = False
            for param in model.conv1.parameters():
                param.requires_grad = True
            for param in model.hidden_layers.parameters():
                param.requires_grad = True

            optimizer_fe.zero_grad()
            final_output, layer_outputs_for_fe_loss = model(inputs)  # 再次进行前向传播以获取新鲜的图
            final_loss = criterion(final_output, targets)
            final_loss.backward()  # 不保留图，因为特征提取器优化后，这部分图就不再需要了
            optimizer_fe.step()
            running_loss += final_loss.item()

            # --- 步骤 2: 训练层分类器并更新alpha权重 ---
            # 冻结特征提取器参数，只更新分类器
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.hidden_layers.parameters():
                param.requires_grad = False
            for param in model.classifiers.parameters():
                param.requires_grad = True  # 解冻分类器参数

            # 再次进行前向传播以获取新鲜的图，这次主要为了获取classifier_outputs的梯度
            # 因为上一步的backward已经清空了图
            _, layer_outputs_for_cls_loss = model(inputs)

            current_layer_losses = []
            for output in layer_outputs_for_cls_loss:
                loss = criterion(output, targets)
                current_layer_losses.append(loss)
                optimizer_cls.zero_grad()  # 为每个分类器清零梯度
                loss.backward(retain_graph=True if loss != current_layer_losses[-1] else False)  # 仅在非最后一个损失时保留图
                optimizer_cls.step()

            # 在每个批次训练完所有分类器后，调用 update_alpha
            if current_layer_losses:  # 确保有层分类器存在
                model.update_alpha(current_layer_losses)

            # --- 步骤 3: 解除所有参数的冻结，为下一次循环做准备 ---
            unfreeze_all_parameters(model)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}")

    return model

# 计算指标函数
def calculate_metrics(model, data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    model.train()  # 评估结束后将模型设置回训练模式
    return accuracy, f1, precision, recall

# 生成新数据，即选择模型表现较好的数据，可在后续研究中分析原因
def generate_csv_with_labels(window_index, features, labels, output_dir="./Dataset/nb15_dataclean.csv"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    window_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
    window_df[" Label"] = labels
    output_path = os.path.join(output_dir, f"window_{window_index + 1}.csv")
    window_df.to_csv(output_path, index=False)
    print(f"窗口 {window_index + 1} 数据保存至 {output_path}")

# 定义评估并绘制混淆矩阵热力图的函数，增加保存功能
def evaluate_and_plot_confusion_matrix(model, data_loader, device, class_names=None, save_path='./pic/NB15.png'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Heatmap (Last Window)")

    if save_path is not None:
        plt.savefig(save_path)
        print(f"热力图已保存至: {save_path}")

    plt.show()

    return conf_matrix

# 修改后的滑动窗口训练函数，增加了输出各项指标折线图的功能
def sliding_window_training_with_cleaning(data, labels, model, device, accuracy_drop_threshold, accuracy_range=None,
                                          output_csv_path="cleaned_data.csv", max_reptile_tuning_attempts=3,
                                          metrics_save_path="./pic/nb15_F1_line_chart.png",
                                          heatmap_save_path="final_nb15_confusion_matrix.png"):
    previous_accuracy = None
    previous_features = None
    # 用于存储每个窗口的指标
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    selected_data = []
    reptile_tuning_attempts = 0

    # 新增两个列表，用于保存每个窗口的数据（如果需要其它用途）
    all_windows_features = []
    all_windows_labels = []

    for window_index, window_data in enumerate(get_windowed_data(data, WINDOW_SIZE, STEP_SIZE)):
        new_features, new_labels = window_data[:, :-1], window_data[:, -1]

        # 保存当前窗口数据到列表中
        all_windows_features.append(new_features)
        all_windows_labels.append(new_labels)

        new_dataset = CustomDataset(torch.tensor(new_features, dtype=torch.float32),
                                    torch.tensor(new_labels, dtype=torch.long))
        new_data_loader = DataLoader(new_dataset, batch_size=32, shuffle=True)

        # 计算当前窗口的指标
        # 在这里，我们需要确保模型在评估前是完全训练好的，
        # 如果模型刚刚经过重训练/微调，这里会使用更新后的模型进行评估。
        accuracy, f1, precision, recall = calculate_metrics(model, new_data_loader, device)
        print(
            f"窗口 {window_index} 的指标: 准确率: {accuracy:.4f}, F1: {f1:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        if accuracy_range:
            min_accuracy, max_accuracy = accuracy_range
            if min_accuracy <= accuracy <= max_accuracy:
                print(f"窗口 {window_index} 符合准确率范围: {accuracy:.4f}，保存该窗口数据")
                selected_data.append(window_data)

        if accuracy < 0.8 or (
                previous_accuracy is not None and (
                previous_accuracy - accuracy) > accuracy_drop_threshold):  # 进入模型重用机制的判定条件
            print(f"窗口 {window_index} 准确率下降超过阈值或低于0.8，启用漂移检测...")
            if previous_features is not None:
                # 基于PCA的漂移角度检测
                pca_obj, explained_variance_ratio, new_pca = detect_data_drift_pca(previous_features, new_features)
                closest_model_entry, min_angle = find_closest_pca_model(new_pca, model_history, distribution_history)
                closest_model_entry = {'model': copy.deepcopy(model), 'fine_tuned': False}  # 模拟找到一个模型

                print(f"找到最接近的角度: {min_angle:.2f} 度")
                if min_angle < 60:
                    print("直接重用最相似的历史模型")
                    model = closest_model_entry['model']
                elif 60 <= min_angle < 120:
                    print("使用 Reptile 微调最相似的历史模型")
                    # 使用copy.deepcopy(model)确保我们微调的是一个副本
                    model = reptile_finetune(copy.deepcopy(closest_model_entry['model']), new_data_loader, device,
                                             lr=0.01,
                                             inner_steps=5)
                    model_history.append({'model': copy.deepcopy(model), 'fine_tuned': True})
                    reptile_tuning_attempts += 1
                    if reptile_tuning_attempts >= max_reptile_tuning_attempts:
                        print(f"连续 {max_reptile_tuning_attempts} 次 Reptile 微调后模型仍然表现不佳，重新训练模型...")
                        fine_tuned_model = next(
                            (entry['model'] for entry in reversed(model_history) if entry['fine_tuned']), None)
                        if fine_tuned_model is not None:
                            print("使用最近微调的模型进行初始化")
                            model = train_model_on_data(copy.deepcopy(fine_tuned_model), new_data_loader, device)
                        else:
                            print("未找到微调过的模型，使用原始模型进行重新训练")
                            model = train_model_on_data(copy.deepcopy(model), new_data_loader, device)
                        reptile_tuning_attempts = 0  # 重置尝试计数
                    else:
                        print(f"Reptile 微调尝试次数: {reptile_tuning_attempts}")
                else:
                    print("重新训练新模型")
                    fine_tuned_model = next(
                        (entry['model'] for entry in reversed(model_history) if entry['fine_tuned']), None)
                    if fine_tuned_model is not None:
                        print("使用最近微调的模型进行初始化")
                        model = train_model_on_data(copy.deepcopy(fine_tuned_model), new_data_loader, device)
                    else:
                        print("未找到微调过的模型，使用原始模型进行重新训练")
                        model = train_model_on_data(copy.deepcopy(model), new_data_loader, device)
                    model_history.append({'model': copy.deepcopy(model), 'fine_tuned': False})
                    # distribution_history.append(new_pca) # 模拟处理
        previous_accuracy = accuracy
        previous_features = new_features

    # 输出所有窗口的平均指标以反映综合性能
    avg_accuracy = np.mean(accuracies)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    print(
        f"\n所有窗口的平均指标: 准确率: {avg_accuracy:.4f}, F1: {avg_f1:.4f}, 精确率: {avg_precision:.4f}, 召回率: {avg_recall:.4f}")

    if selected_data:
        selected_data = np.concatenate(selected_data, axis=0)
        selected_data_df = pd.DataFrame(selected_data,
                                        columns=[f'feature_{i}' for i in range(selected_data.shape[1] - 1)] + [
                                            " Label"])
        selected_data_df.to_csv(output_csv_path, index=False)
        print(f"符合条件的窗口数据已保存至: {output_csv_path}")
    else:
        print("没有符合条件的窗口数据")

    # 绘制折线图，展示每个窗口的各项指标变化
    plt.figure(figsize=(10, 6))
    x = range(len(f1_scores))
    window_size = 5
    f1_scores_smoothed = np.convolve(f1_scores, np.ones(window_size) / window_size, mode='valid')
    plt.plot(x[:len(f1_scores_smoothed)], f1_scores_smoothed, linestyle='-', linewidth=2, color='red', label="F1-Score")
    plt.xlabel("Window Index")
    plt.ylabel("Metric Value")
    plt.title("Metrics over Sliding Windows")
    plt.legend()
    plt.grid(True)
    plt.savefig(metrics_save_path)
    print(f"折线图已保存至: {metrics_save_path}")
    plt.show()

    # 如果需要输出最后 100 个窗口的混淆矩阵热力图：
    num_windows = len(all_windows_features)
    num_to_use = 100 if num_windows >= 100 else num_windows
    last_features = np.concatenate(all_windows_features[-num_to_use:], axis=0)
    last_labels = np.concatenate(all_windows_labels[-num_to_use:], axis=0)

    last_dataset = CustomDataset(torch.tensor(last_features, dtype=torch.float32),
                                 torch.tensor(last_labels, dtype=torch.long))
    last_loader = DataLoader(last_dataset, batch_size=32, shuffle=False)
    class_names = [str(i) for i in range(14)]  # 根据类别数或具体类别名称修改
    print("输出并保存最后 100 个窗口的混淆矩阵热力图：")
    evaluate_and_plot_confusion_matrix(model, last_loader, device, class_names=class_names, save_path=heatmap_save_path)


# 读取完整数据集
features, labels = load_full_data("./Dataset/UNSW_NB15_combined.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AMLCN(39, 128, 14).to(device)#两个数据集特征维度不一样(2017：76，128，15：NB15：39，128，14)

# 执行滑动窗口训练并清洗数据，同时保存最后窗口的混淆矩阵热力图（因为只保存最后窗口，所以类别显示不全）
sliding_window_training_with_cleaning(features, labels, model, device, accuracy_drop_threshold=0.1,
                                      accuracy_range=(0.9, 1.0), output_csv_path="nb15_cleaned.csv",
                                      heatmap_save_path="nb15_final_confusion_matrix.png")