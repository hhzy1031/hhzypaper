import torch
import torch.nn as nn
import torch.optim as optim
class ProtoNet(nn.Module):
    def __init__(self, feature_dim=81, num_classes=15, num_prototypes=15):
        super(ProtoNet, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))  # 特征维度为77

    def forward(self, x):
        # 计算样本与原型之间的距离
        proto_dist = torch.cdist(x, self.prototypes.unsqueeze(0)).squeeze(0)
        return proto_dist