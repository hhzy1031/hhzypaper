from sklearn.decomposition import PCA
from math import acos, degrees
import numpy as np
from sklearn.preprocessing import StandardScaler

# PCA based drift detection
def detect_data_drift_pca(data_old, data_new, weight_vector=0.5):
    """
    检测PCA基础上的数据漂移，通过结合基于协方差矩阵和PCA计算的夹角。
    :param data_old: 历史数据
    :param data_new: 当前数据
    :param weight_vector: 两种夹角计算方式的加权比例（0-1之间）
    :return: 是否检测到漂移，夹角，主成分
    """
    # 将数据转换为 numpy 数组
    data_old = np.array(data_old)
    data_new = np.array(data_new)

    # 使用相同的标准化器进行拟合和转换
    scaler = StandardScaler()
    data_old_standardized = scaler.fit_transform(data_old)
    data_new_standardized = scaler.transform(data_new)

    # 计算协方差矩阵
    cov_matrix_old = np.cov(data_old_standardized, rowvar=False)
    cov_matrix_new = np.cov(data_new_standardized, rowvar=False)

    # 计算协方差矩阵的特征值和特征向量
    eig_values_old, eig_vectors_old = np.linalg.eig(cov_matrix_old)
    eig_values_new, eig_vectors_new = np.linalg.eig(cov_matrix_new)

    # 使用PCA提取主成分
    pca_old = PCA()
    pca_old.fit(data_old_standardized)
    components_old = pca_old.components_

    pca_new = PCA()
    pca_new.fit(data_new_standardized)
    components_new = pca_new.components_

    # 第一种夹角计算：基于协方差矩阵的特征向量
    angles_cov_based = []
    for i in range(len(eig_vectors_old)):
        dot_product = np.dot(eig_vectors_old[:, i], eig_vectors_new[:, i])
        dot_product = min(1, max(-1, dot_product))  # 确保余弦值在 -1 到 1 之间
        angle = acos(dot_product)
        angles_cov_based.append(degrees(angle))  # 转换为角度

    # 第二种夹角计算：基于PCA计算的主成分之间的夹角
    angles_pca_based = []
    for i in range(len(components_old)):
        dot_product = np.dot(components_old[i], components_new[i].T)
        dot_product = min(1, max(-1, dot_product))  # 确保余弦值在 -1 到 1 之间
        angle = acos(dot_product)
        angles_pca_based.append(degrees(angle))  # 转换为角度

    # 合并两种夹角计算方式，通过加权
    combined_angles = []
    for i in range(len(angles_cov_based)):
        weighted_angle = weight_vector * angles_cov_based[i] + (1 - weight_vector) * angles_pca_based[i]
        combined_angles.append(weighted_angle)

    # 检查是否存在漂移，根据加权后的夹角判断是否超过阈值
    drift_detected = any(angle >= 60 for angle in combined_angles)

    return drift_detected, combined_angles, pca_new.components_