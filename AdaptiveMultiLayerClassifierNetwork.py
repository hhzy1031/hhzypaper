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

# 检查是否可以使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取CSV文件的函数
def read_csv_files(folder_path, num_files, file_prefix):
    data_list = []
    labels_list = []
    for i in range(num_files):
        file_path = os.path.join(folder_path, f"{file_prefix}_task{i}.csv")
        df = pd.read_csv(file_path)
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        data_list.append(data)
        labels_list.append(labels)
    data = np.vstack(data_list)
    labels = np.hstack(labels_list)
    return data, labels

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
        inv_losses = 1 / (losses + 1e-8)
        self.alpha = inv_losses / inv_losses.sum()

# 打印梯度的函数
def print_gradients(model, phase):
    print(f"=== Gradients during {phase} update ===")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"{name}: {param.grad.norm().item():.6f}")
        elif not param.requires_grad:
            print(f"{name}: No gradient computed")
    print("=" * 40)

# 修改训练函数
def train_with_gradient_check(model, train_loader, val_loader, optimizer, scheduler, device, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0  # 记录每一轮的总损失
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 前向传播，获取最终输出和每层分类器的输出
            final_output, layer_outputs = model(inputs)

            # 计算最终输出的损失，并仅对特征提取层反向传播
            final_loss = criterion(final_output, targets)
            final_loss.backward(retain_graph=True)  # 保留计算图以便后续反向传播

            # 打印特征提取器更新前的梯度
            print_gradients(model, "Feature Extractor Update")

            # 冻结隐藏层的权重，使它们不受分类器损失的影响
            for param in model.hidden_layers.parameters():
                param.requires_grad = False

            # 计算每个分类器的损失，并逐一更新分类器层
            for i, output in enumerate(layer_outputs):
                loss = criterion(output, targets)
                loss.backward(retain_graph=True if i < len(layer_outputs) - 1 else False)

            # 打印分类器更新后的梯度
            print_gradients(model, f"Classifier {i + 1} Update")

            # 解除隐藏层的冻结，恢复它们的梯度
            for param in model.hidden_layers.parameters():
                param.requires_grad = True

            optimizer.step()
            running_loss += final_loss.item()

        # 更新学习率
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
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {accuracy:.4f}")
