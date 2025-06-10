import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

def calculate_kl_divergence(data_old, data_new):
    # 标准化两个数据集的特征值
    scaler = StandardScaler()
    data_old_standardized = scaler.fit_transform(data_old)
    data_new_standardized = scaler.transform(data_new)

    # 计算每个特征的KL散度
    kl_divergence = []
    for feature_old, feature_new in zip(data_old_standardized.T, data_new_standardized.T):
        p = np.histogram(feature_old, bins=30, density=True)[0]
        q = np.histogram(feature_new, bins=30, density=True)[0]

        # 增加一个小的值来防止零概率
        p += 1e-10
        q += 1e-10

        # 计算KL散度
        kl_divergence.append(entropy(p, q))

    return np.mean(kl_divergence)


def detect_data_drift(data_old, data_new):
    # 计算KL散度
    kl_threshold = 0.1  # 设置一个合适的阈值，具体值需要根据数据集和实验结果调整
    kl_drift_detected = calculate_kl_divergence(data_old, data_new) > kl_threshold

    return kl_drift_detected


# 示例数据
# data_old = pd.read_csv("../../data_paper1/CICDS2017/task_stand_num_littleshot/train/train_task2.csv")
# data_new = pd.read_csv("../../data_paper1/CICDS2017/task_stand_num_littleshot/train/train_task3.csv")
#
# data_old = pd.read_csv("../../data_paper1/CICDS2018/task_stand_num_resampled/train/train_task40.csv")
# data_new = pd.read_csv("../../data_paper1/CICDS2018/task_stand_num_resampled/train/train_task41.csv")

# data_old = pd.read_csv("../../data_paper1/CICDS2018/task_stand_num_resampled/train/train_task3.csv")
# data_new = pd.read_csv("../../data_paper1/CICDS2018/task_stand_num_resampled/test/test_task3.csv")
# #
# # 检测数据漂移
# drift_detected = detect_data_drift(data_old, data_new)
# print("Data drift detected:", drift_detected)
