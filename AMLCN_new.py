import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 FocalLoss 类，用于处理不平衡数据
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        logpt = torch.nn.functional.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = torch.nn.functional.nll_loss(logpt, target, weight=self.alpha, reduction=self.reduction)
        return loss

class AMLCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=12, n_classifiers=3, eta=0.1):
        super(AMLCN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.prototype_feature_layer = nn.Linear(hidden_dim, 81)  # 新增：独立的特征提取层
        classifier_index = 0

        # 全连接层初始化
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))

        # 添加剩余隐藏层
        for i in range(1, n_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if (i + 1) % 3 == 0 and classifier_index < n_classifiers:
                self.classifiers.append(nn.Linear(hidden_dim, output_dim))
                classifier_index += 1

        self.softmax = nn.Softmax(dim=1)
        self.alpha = torch.ones(len(self.classifiers), requires_grad=False).to(device) / len(self.classifiers)
        self.eta = eta
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        classifier_outputs = []

        # 前向传播
        for i, hidden in enumerate(self.hidden_layers):
            x = F.relu(hidden(x))
            x = self.dropout(x)
            if (i + 1) % 3 == 0 and (i + 1) // 3 - 1 < len(self.classifiers):
                classifier_outputs.append(self.classifiers[(i + 1) // 3 - 1](x))

        # 计算加权输出
        final_output = sum(a * o for a, o in zip(self.alpha, classifier_outputs))
        return final_output, classifier_outputs

    def update_alpha(self, losses):
        """
        根据每一层的损失值更新alpha权重，损失越大权重越小
        """
        losses = torch.tensor([loss.item() for loss in losses], dtype=torch.float32).to(device)
        inv_losses = 1 / (losses + 1e-8)
        self.alpha = inv_losses / inv_losses.sum()

    def get_features(self, x):
        # 选择最后一个隐藏层的输出作为特征
        for hidden in self.hidden_layers:
            x = F.relu(hidden(x))
            x = self.dropout(x)
        prototype_features = self.prototype_feature_layer(x)
        return prototype_features

def train_feature_extractor(model, optimizer, final_output_loss):
    """
    更新特征提取器，只依据 final_output 的损失更新。
    """
    # 清零梯度
    optimizer.zero_grad()

    # 将分类器层的 requires_grad 设置为 False，防止更新分类器层的参数
    for name, param in model.named_parameters():
        if "classifiers" in name:  # 排除分类器层
            param.requires_grad = False  # 停止计算分类器层的梯度

    # 计算损失并反向传播
    final_output_loss.backward(retain_graph=True)
    # 更新特征提取器层的参数
    optimizer.step()
    # 打印特征提取器更新后的梯度
    #print_gradients(model, "Feature Extractor Update")

def train_layer_classifier(model, optimizer, classifier_loss, layer_idx):
    """
    更新指定的层分类器，仅依据其输出计算的损失更新。
    """
    # 清零梯度
    optimizer.zero_grad()
    #将分类器层的 requires_grad 设置为 True
    for name, param in model.named_parameters():
        if "classifiers" in name:
            param.requires_grad = True

    # 冻结卷积层和隐藏层
    for name, param in model.named_parameters():
        if "conv1" in name or "hidden_layers" in name:
            param.requires_grad = False

    # 计算损失并反向传播
    classifier_loss.backward(retain_graph=True)

    # 仅更新该层分类器
    for name, param in model.named_parameters():
        if f"classifiers.{layer_idx}" in name and param.requires_grad:
            optimizer.step()
            # 打印分类器更新后的梯度
            #print_gradients(model, f"Classifier {layer_idx} Update")

    # 解冻卷积层和隐藏层
    for name, param in model.named_parameters():
        if "conv1" in name or "hidden_layers" in name:
            param.requires_grad = True

# 模型训练函数，使用新数据进行训练
def train_model_on_data(model, data_loader, device, epochs=10):
    model.to(device)
    optimizer_fe = optim.Adam(
        [param for name, param in model.named_parameters() if "classifiers" not in name], lr=0.001
    )
    optimizer_cls = optim.Adam(
        [param for name, param in model.named_parameters() if "classifiers" in name], lr=0.001
    )
    criterion = FocalLoss(gamma=2).to(device)  # 假设你已经定义了FocalLoss

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0  # 记录每一轮的总损失
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播，获取最终输出和每层分类器的输出
            final_output, layer_outputs = model(inputs)

            # 计算最终输出的损失，并仅对特征提取层更新
            final_loss = criterion(final_output, targets)
            train_feature_extractor(model, optimizer_fe, final_loss)

            # 逐一计算每个分类器的损失并更新
            for i, output in enumerate(layer_outputs):
                loss = criterion(output, targets)
                train_layer_classifier(model, optimizer_cls, loss, layer_idx=i)

            running_loss += final_loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}")

    return model