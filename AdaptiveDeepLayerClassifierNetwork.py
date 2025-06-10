import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_csv_files(folder_path, file_list):
    """
    从指定文件夹中读取指定的文件列表，并返回合并后的数据和标签。
    """
    data = []
    labels = []

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        data.append(df.iloc[:, :-1].values)  # 假设最后一列是标签
        labels.append(df.iloc[:, -1].values)

    # 合并所有文件的数据
    data = np.vstack(data)
    labels = np.hstack(labels)
    return data, labels

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型定义
class ADLCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=12, n_classifiers=3, eta=0.1):
        super(ADLCN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - n_classifiers):#前几层为隐藏层
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.classifiers = nn.ModuleList()
        for _ in range(n_classifiers):#后几层为层分类器
            self.classifiers.append(nn.Linear(hidden_dim, output_dim))
        self.softmax = nn.Softmax(dim=1)
        self.alpha = torch.ones(n_classifiers, requires_grad=False).to(device) / n_classifiers  # 初始化alpha，均分权重
        self.eta = eta
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):#前向传播
        for hidden in self.hidden_layers:
            x = F.relu(hidden(x))
            x = self.dropout(x)
        layer_outputs = [classifier(x) for classifier in self.classifiers]
        final_output = sum(a * o for a, o in zip(self.alpha, layer_outputs))
        return final_output, layer_outputs

    def update_alpha(self, losses):
        """
        根据每一层的损失值更新权重 alpha，损失大的权重小
        """
        losses = torch.tensor([loss.item() for loss in losses], dtype=torch.float32).to(device)
        inv_losses = 1 / (losses + 1e-8)
        self.alpha = inv_losses / inv_losses.sum()

# 训练函数
def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0  # 记录每一轮的总损失
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # 前向传播，获取最终输出和各层分类器的输出
            final_output, layer_outputs = model(inputs)
            # 计算最终输出的损失，并仅对特征提取层反向传播
            final_loss = criterion(final_output, targets)
            final_loss.backward(retain_graph=True)  # 保留计算图以便后续反向传播
            # 冻结隐藏层的权重，使它们不受分类器损失的影响
            for param in model.hidden_layers.parameters():
                param.requires_grad = False
            # 分别更新每个分类器
            for i, (classifier, output) in enumerate(zip(model.classifiers, layer_outputs)):
                # 冻结所有分类器的参数
                for param in model.classifiers.parameters():
                    param.requires_grad = False
                # 解冻当前分类器的参数
                for param in classifier.parameters():
                    param.requires_grad = True
                # 计算当前分类器的损失并反向传播
                loss = criterion(output, targets)
                loss.backward(retain_graph=True if i < len(layer_outputs) - 1 else False)
                # 冻结当前分类器
                for param in classifier.parameters():
                    param.requires_grad = False
            # 解除隐藏层的冻结，恢复它们的梯度
            for param in model.hidden_layers.parameters():
                param.requires_grad = True
            optimizer.step()
            running_loss += final_loss.item()
        # 调度学习率
        scheduler.step()

        # 在验证集上评估
        model.eval()
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(targets.cpu().numpy())

        # 计算验证集准确率
        accuracy = accuracy_score(val_labels, val_predictions)

        # 打印每一轮的损失和验证集准确率
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.4f}")

# 测试函数
def test(model, test_data, test_labels, device):
    model.to(device)
    model.eval()
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs, _ = model(test_data)
        _, predicted = torch.max(outputs, 1)

    y_true = test_labels.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

# 主函数
def main():
    # 文件路径和参数设置
    train_folder_path = "./Dataset/CIC-DDoS2019/task/train/"
    test_folder_path = "./Dataset/CIC-DDoS2019/task/test/"
    num_files = 30  # 随机选择的文件数量
    batch_size = 128
    epochs = 30
    learning_rate = 0.001
    hidden_dim = 128
    output_dim = 18  # 根据实际分类数设置
    n_layers = 12

    # 获取文件列表
    train_files = sorted([f for f in os.listdir(train_folder_path) if f.startswith("train_task")])
    test_files = sorted([f for f in os.listdir(test_folder_path) if f.startswith("test_task")])

    # 确保训练和测试文件夹中有对应的文件
    assert len(train_files) == len(test_files), "训练和测试文件数量不匹配！"

    # 随机选择文件
    selected_indices = random.sample(range(1,60), num_files)
    selected_train_files = [train_files[i] for i in selected_indices]
    selected_test_files = [test_files[i] for i in selected_indices]

    # 读取数据
    train_data, train_labels = read_csv_files(train_folder_path, selected_train_files)
    test_data, test_labels = read_csv_files(test_folder_path, selected_test_files)

    # 分割训练集和验证集
    #x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    # 转换为 Tensor
    x_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    #x_val = torch.tensor(x_val, dtype=torch.float32)
    #y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器和学习率调度器
    model = ADLCN(input_dim=x_train.shape[1], hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练和测试
    train(model, train_loader, test_loader, optimizer, scheduler, device, epochs=epochs)
    accuracy, precision, recall, f1 = test(model, x_test, y_test, device)

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    torch.save(model.state_dict(), './Model/CIC-DDoS2019/model/pre/adlcn_model.pt')

if __name__ == "__main__":
    main()
