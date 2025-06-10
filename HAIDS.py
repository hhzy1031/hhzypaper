import higher
import random
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from math import acos, degrees
from kl_divergence import *
from PCA_detector import *
from AdaptiveDeepLayerClassifierNetwork import *
from torch.optim.lr_scheduler import CosineAnnealingLR

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
#切换数据集时，下面代码的目录也需要统一变更，同时model0和model1的输入也需要改
model0 = ADLCN(82,128,18,n_layers=12, n_classifiers=3, eta=0.1).to(device)#17，18,mixed数据集维度76，类别数为15，15，2，19数据集特征维度为82，类别数18
model1 = ADLCN(82,128,18,n_layers=12, n_classifiers=3, eta=0.1).to(device)
model0.load_state_dict(torch.load("./Model/CIC-DDoS2019/model/pre/adlcn_model.pt"))
model1.load_state_dict(torch.load("./Model/CIC-DDoS2019/model/pre/adlcn_model.pt"))
# 寄存器来交换的参数
model_register = ADLCN(82,128,18,n_layers=12, n_classifiers=3, eta=0.1).to(device)

# 定义数据集
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def detect_data_drift_kl(data_old, data_new, kl_threshold=0.35):
    kl_drift_detected = calculate_kl_divergence(data_old, data_new) > kl_threshold
    return kl_drift_detected

#加载测试数据集
def load_test_data(test_data_dir):
    test_files = [file for file in os.listdir(test_data_dir) if file.endswith(".csv")]
    test_data = pd.concat([pd.read_csv(os.path.join(test_data_dir, f)) for f in test_files])
    features = torch.tensor(test_data.drop(columns=[' Label']).values, dtype=torch.float32)#2019为' Label',其余为'Label'
    labels = torch.tensor(test_data[' Label'].values, dtype=torch.long)
    dataset = CustomDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    return data_loader

# 定义一个函数用于计算准确率和F1指标
def calculate_metrics(model, data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    #recall = (f1 * precision) / (2 * precision - f1) if (2 * precision - f1) != 0 else 0#手动计算recall
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    # 计算混淆矩阵
    # cm = confusion_matrix(true_labels, pred_labels)
    # print("Confusion Matrix:")
    # print(cm)

    model.train()
    return accuracy, f1, precision, recall

def sample_tasks(old_data_loader, new_data_loader, num_samples_per_task=32):
    """
    从旧数据和新数据加载器中随机采样任务数据集。

    参数:
    - old_data_loader: 代表"旧数据"的DataLoader，通常用作支持集。
    - new_data_loader: 代表"新数据"的DataLoader，通常用作查询集。
    - num_samples_per_task: 每个任务（支持集和查询集）中样本的数量。

    返回:
    - 两个元组，分别代表一个模拟任务的“支持集”和“查询集”的数据加载器。
    """
    # 确保加载器有足够数据
    assert len(old_data_loader.dataset) >= num_samples_per_task, "Not enough samples in the old data_paper1 loader"
    assert len(new_data_loader.dataset) >= num_samples_per_task, "Not enough samples in the new data_paper1 loader"

    # 随机采样索引
    old_indices = random.sample(range(len(old_data_loader.dataset)), num_samples_per_task)
    new_indices = random.sample(range(len(new_data_loader.dataset)), num_samples_per_task)

    # 创建子采样器
    old_sampler = SubsetRandomSampler(old_indices)
    new_sampler = SubsetRandomSampler(new_indices)

    # 使用子采样器创建新的数据加载器
    sampled_old_data_loader = torch.utils.data.DataLoader(old_data_loader.dataset,
                                                          batch_size=old_data_loader.batch_size,
                                                          sampler=old_sampler)
    sampled_new_data_loader = torch.utils.data.DataLoader(new_data_loader.dataset,
                                                          batch_size=new_data_loader.batch_size,
                                                          sampler=new_sampler)

    return sampled_old_data_loader, sampled_new_data_loader

#多视图分级漂移检测算法
class DriftDetection:
    def __init__(self):
        self.warning_state = False # 指示当前是否处于警告状态
        self.warning_counter = 0 # 记录连续触发警告的次数

    def detect_drift(self, data_old, data_new):#基于 PCA 和 KL 散度的漂移检测
        drift_pca = detect_data_drift_pca(data_old, data_new)
        drift_kl = detect_data_drift_kl(data_old, data_new)

        drift_detected = False# 初始化漂移检测结果

        if drift_pca and drift_kl:# 如果 PCA 和 KL 散度都检测到漂移，则认为发生了漂移，并重置警告状态
            self.reset_warning_state()
            drift_detected = True
        elif drift_pca or drift_kl:# 如果只有一种方法检测到漂移，则进入警告状态或增加警告计数器
            if self.warning_state:
                self.warning_counter += 1# 如果已经处于警告状态，则增加警告计数器
                if self.warning_counter >= 5:#漂移警告阈值
                    self.reset_warning_state()
                    drift_detected = True
            else:
                self.warning_state = True# 如果之前没有处于警告状态，则设置警告状态为 True 并初始化计数器
                self.warning_counter = 1
        else:
            self.reset_warning_state()# 如果两种方法都没有检测到漂移，则重置警告状态

        return drift_detected, self.warning_state

    def reset_warning_state(self):#重置警告状态和警告计数器。当检测到确定的漂移或没有检测到漂移时，调用此方法。
        self.warning_state = False
        self.warning_counter = 0

detector = DriftDetection()

# 添加L2正则化和Dropout
regularization_lambda = 1e-4
change = False

# 加载测试数据
test_data_loader = load_test_data("./Dataset/CIC-DDoS2019/task/test")

# 初始化基础分类模型
models = [model0, model1]

def update_model_parameters(source_model, target_model):
    target_model.load_state_dict(source_model.state_dict())

# 文件范围限制
start_index =12  # 起始文件索引（可以自己指定）
end_index = 73 # 结束文件索引（None表示读取到文件列表的末尾）

# 加载所有CSV文件
data_files = sorted([file for file in os.listdir("./Dataset/CIC-DDoS2019/task/train") if file.endswith(".csv")])
if end_index is None:
    end_index = len(data_files)
previous_data = None
# 保存结果的列表
results = []
test_results = []  # 存储每个测试任务的准确率和其他指标

# 训练循环
for file_index, file_name in enumerate(data_files[start_index:end_index]):
    # 加载新数据
    data = pd.read_csv(os.path.join("./Dataset/CIC-DDoS2019/task/train", file_name))
    new_features = torch.tensor(data.drop(columns=[' Label']).values, dtype=torch.float32)
    new_labels = torch.tensor(data[' Label'].values, dtype=torch.long)
    new_dataset = CustomDataset(new_features, new_labels)
    new_data_loader = DataLoader(new_dataset, batch_size=128, shuffle=True)

    if previous_data is not None:
        # 加载旧数据
        old_features = torch.tensor(previous_data.drop(columns=[' Label']).values, dtype=torch.float32)
        old_labels = torch.tensor(previous_data[' Label'].values, dtype=torch.long)
        old_dataset = CustomDataset(old_features, old_labels)
        old_data_loader = DataLoader(old_dataset, batch_size=128, shuffle=True)

        # 检测数据漂移并返回警告状态
        drift_detected, warning_state = detector.detect_drift(previous_data, data)
        print("检测到数据漂移" if drift_detected else "未检测到数据漂移")
        print("进入警告状态" if warning_state else "未进入警告状态")

        if drift_detected:
            # 进行在线学习
            for idx, model in enumerate(models):
                if idx == 0:
                    if change:
                        # 进行在线学习，更新 model0
                        update_model_parameters(model, model_register)
                        update_model_parameters(model1, model)
                        # model1用model0的原参数分类
                        update_model_parameters(model_register, model1)
                    change = False

                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    scheduler = CosineAnnealingLR(optimizer, T_max=30)
                    criterion = nn.CrossEntropyLoss()
                    for epoch in range(30):  # 训练 30 个 epoch
                        model.train()
                        running_loss = 0.0  # 记录每轮总损失
                        for batch_index, (inputs, targets) in enumerate(new_data_loader):
                            inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到 GPU/CPU

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

                        # 打印每轮的损失
                        print(f"Epoch [{epoch + 1}/30], Loss: {running_loss / len(new_data_loader):.4f}")

                    # 保存模型
                    torch.save(model.state_dict(), f'./Model/CIC-DDoS2019/model/online/adlcn_{file_index}.pt')

                    # 测试模型性能
                    acc, f1, precision, recall = calculate_metrics(model, test_data_loader, device)
                    print(f"Model {idx} 对新到达数据 {file_index} 更新时的测试准确率: {acc:.4f}, F1 Score: {f1:.4f}, "
                          f"Pre: {precision:.4f}, Recall: {recall:.4f}")

                else:
                    acc, f1, precision, recall = calculate_metrics(model, test_data_loader, device)
                    print(f"Model {idx}在新到达数据{file_index}的测试准确率: {acc:.4f}, F1 Score: {f1:.4f},"
                          f"Pre: {precision:.4f}, Recall: {recall:.4f}")
                    results.append(f"Model{idx}在数据流{file_index}上的测试结果: Accuracy: {acc}, F1: {f1}, "
                                   f"Precision: {precision}, Recall: {recall}")
                    update_model_parameters(model0, model)
        elif warning_state:
            # 进行元学习训练
            change = True
            for idx, model in enumerate(models):
                if idx == 1:
                    meta_lr = 0.0001
                    inner_lr = 0.001
                    num_tasks_per_epoch = 50  # 假设每个epoch处理两个任务
                    num_inner_steps = 20
                    num_epochs = 5

                    meta_optimizer = torch.optim.Adamax(model.parameters(), lr=meta_lr)
                    inner_optimizer = optim.Adam(model.parameters(), lr=inner_lr)
                    old_features = torch.tensor(previous_data.drop(columns=[' Label']).values, dtype=torch.float32)
                    old_labels = torch.tensor(previous_data[' Label'].values, dtype=torch.long)
                    old_dataset = CustomDataset(old_features, old_labels)
                    old_data_loader = DataLoader(old_dataset, batch_size=128, shuffle=True)

                    for epoch in range(num_epochs):
                        for _ in range(num_tasks_per_epoch):  # 对于每个任务
                            # 模拟从任务分布中随机采样支持集和查询集
                            sampled_old_data, sampled_new_data = sample_tasks(old_data_loader, new_data_loader)

                            # 外循环（元更新）
                            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                                # 内循环（旧任务学习）
                                for _ in range(num_inner_steps):
                                    support_inputs, support_labels = next(iter(sampled_old_data))
                                    support_inputs, support_labels = support_inputs.to(device), support_labels.to(
                                        device)
                                    support_outputs, _ = fmodel(support_inputs)
                                    inner_loss = nn.CrossEntropyLoss()(support_outputs, support_labels)
                                    diffopt.step(inner_loss)

                                # 适应新任务（模拟查询集）
                                query_inputs, query_targets = next(iter(sampled_new_data))
                                query_inputs, query_targets = query_inputs.to(device), query_targets.to(device)
                                query_outputs, _ = fmodel(query_inputs)
                                outer_loss = nn.CrossEntropyLoss()(query_outputs, query_targets)

                            meta_optimizer.zero_grad()
                            outer_loss.backward()  # 反向传播外循环的损失
                            meta_optimizer.step()

                        # 计算并打印准确率
                        accuracy, f1, precision, recall = calculate_metrics(model, test_data_loader, device)
                        print(f"MamlEpoch {epoch + 1}: Average Loss: {outer_loss.item():.4f}, "
                              f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f},"
                              f"Precision: {precision:.4f}, Recall: {recall:.4f}")

                    torch.save(model.state_dict(), f'./Model/CIC-DDoS2019/model/meta/adlcn_{idx}.pt')
                else:
                    # 模型0分类
                    acc, f1, precision, recall = calculate_metrics(model, test_data_loader, device)
                    print(f"Model {idx}在新到达数据{file_index}的测试准确率: {acc:.4f}, F1 Score: {f1:.4f},"
                          f"Pre: {precision:.4f}, Recall: {recall:.4f}")
                    results.append(f"Model{idx}在数据流{file_index}上的测试结果: Accuracy: {acc}, F1: {f1}, "
                                   f"Precision: {precision}, Recall: {recall}")
        else:
            # 无需更新，只需继续检测
            change = False
            print("未检测到数据漂移且未进入警告状态，继续检测无需更新")
            for idx, model in enumerate(models):
                if idx == 0:
                    acc, f1, precision, recall = calculate_metrics(model, test_data_loader, device)
                    print(f"Model {idx}在新到达数据{file_index}的测试准确率: {acc:.4f}, F1 Score: {f1:.4f},"
                          f"Pre: {precision:.4f}, Recall: {recall:.4f}")
                    results.append(f"Model{idx}在数据流{file_index}上的测试结果: Accuracy: {acc}, F1: {f1}, "
                                   f"Precision: {precision}, Recall: {recall}")
    else:
        print("第一次加载数据，跳过漂移检测")

    previous_data = data

# 将结果写入文件
with open("results_CIC-DDoS2019.txt", "w") as f:
    for result in results:
        f.write(result + "\n")

# 定义文件路径
file_path = "./results_CIC-DDoS2019.txt"

# 读取文件并解析结果
def read_results(file_path):
    results = []
    with open(file_path, "r") as f:
        for line in f:
            # 假设每一行的格式为：ModelX在数据流Y上的测试结果: Accuracy: 0.8864815698437314, F1: 0.8649399306264598, Precision: 0.9067828863736238, Recall: 0.8267882783794089
            parts = line.strip().split(",")
            try:
                result = {
                    "accuracy": float(parts[0].split(":")[-1].strip()),
                    "f1": float(parts[1].split(":")[-1].strip()),
                    "precision": float(parts[2].split(":")[-1].strip()),
                    "recall": float(parts[3].split(":")[-1].strip()),
                }
                results.append(result)
            except (IndexError, ValueError):
                print(f"跳过无法解析的行: {line.strip()}")
    return results

# 计算平均值
def calculate_average(metrics):
    avg_metrics = {
        key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]
    }
    return avg_metrics

# 读取结果文件并解析
results = read_results(file_path)

# 确保任务集数量足够
if len(results) < 30:
    print("结果数量不足30，无法进行计算！")
else:
    results = sorted(results, key=lambda x: x["accuracy"], reverse=True)[:60]
    selected_results = results
    # 计算平均指标
    avg_metrics = calculate_average(selected_results)
    # 输出平均值
    print("平均指标:")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"F1 Score: {avg_metrics['f1']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")