from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig
import numpy as np
from math import acos, degrees

# 漂移检测函数，使用 PCA 计算新旧数据之间的偏移角度
def detect_data_drift_pca(data_old, data_new, weight_vector=0.5):
    data_old = np.array(data_old)
    data_new = np.array(data_new)

    scaler = StandardScaler()
    data_old_standardized = scaler.fit_transform(data_old)
    data_new_standardized = scaler.transform(data_new)

    cov_matrix_old = np.cov(data_old_standardized, rowvar=False)
    cov_matrix_new = np.cov(data_new_standardized, rowvar=False)

    eig_values_old, eig_vectors_old = np.linalg.eig(cov_matrix_old)
    eig_values_new, eig_vectors_new = np.linalg.eig(cov_matrix_new)

    pca_old = PCA()
    pca_old.fit(data_old_standardized)
    components_old = pca_old.components_

    pca_new = PCA()
    pca_new.fit(data_new_standardized)
    components_new = pca_new.components_

    angles_cov_based = []
    for i in range(len(eig_vectors_old)):
        dot_product = np.dot(eig_vectors_old[:, i], eig_vectors_new[:, i])
        dot_product = min(1, max(-1, dot_product))
        angle = acos(dot_product)
        angles_cov_based.append(degrees(angle))

    angles_pca_based = []
    for i in range(len(components_old)):
        dot_product = np.dot(components_old[i], components_new[i].T)
        dot_product = min(1, max(-1, dot_product))
        angle = acos(dot_product)
        angles_pca_based.append(degrees(angle))

    combined_angles = []
    for i in range(len(angles_cov_based)):
        weighted_angle = weight_vector * angles_cov_based[i] + (1 - weight_vector) * angles_pca_based[i]
        combined_angles.append(weighted_angle)

    drift_detected = any(angle >= 60 for angle in combined_angles)

    return drift_detected, combined_angles, pca_new.components_

# 计算余弦相似度的函数
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

#寻找最相似的历史模型
def find_closest_pca_model(new_pca, models, pca_history):
    min_angle = float('inf')
    closest_model = None
    closest_pca = None
    for i, old_pca in enumerate(pca_history):
        # 计算新数据和当前历史数据主成分之间的余弦相似度
        cosine_sim = cosine_similarity(new_pca.flatten(), old_pca.flatten())
        angle = degrees(acos(cosine_sim))
        # 将余弦相似度转换为角度（弧度 -> 度）
        # arccos 返回弧度值，degrees 将弧度转换为度
        if angle < min_angle:
            min_angle = angle
            closest_model = models[i]
            closest_pca = old_pca
        # 如果当前角度小于已知的最小角度，则更新最小角度和最相似的模型
    return closest_model, min_angle