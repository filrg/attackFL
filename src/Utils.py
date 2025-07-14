import copy
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.stats import median_abs_deviation as MAD


class ICUData(Dataset):
    def __init__(self, dataframe, vitals_cols, labs_cols, label_col):
        self.vitals = dataframe[vitals_cols].values
        self.labs = dataframe[labs_cols].values
        self.labels = dataframe[label_col].values

        self.vitals = torch.tensor(self.vitals, dtype=torch.float32)
        self.labs = torch.tensor(self.labs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vitals[idx], self.labs[idx], self.labels[idx]


def compute_distance(state_dict1, state_dict2, p=2):
    """
    Tính khoảng cách giữa hai state_dict của PyTorch bằng norm Lp.

    Args:
        state_dict1 (dict): state_dict đầu tiên.
        state_dict2 (dict): state_dict thứ hai.
        p (int): Bậc của norm (mặc định là L2 norm).

    Returns:
        float: Khoảng cách giữa hai state_dict.
    """
    total_distance = 0.0

    for key in state_dict1.keys():
        if key in state_dict2:
            diff = state_dict1[key] - state_dict2[key]
            total_distance += torch.linalg.norm(diff, ord=p).item()

    return total_distance


def create_random_base_model(state_dict, perturbation=1e6):
    base_model = copy.deepcopy(state_dict)
    for param in base_model.keys():
        if base_model[param].dtype != torch.long:
            base_model[param] += torch.randn_like(base_model[param]) * perturbation
    return base_model


def check_min_max_distance(all_genuine_models, max_distance, malicious_model):
    malicious_distance = 0.0

    for genuine_model in all_genuine_models:
        distance = compute_distance(malicious_model, genuine_model)
        if distance > malicious_distance:
            malicious_distance = distance

    return malicious_distance < max_distance


def check_min_sum_distance(all_genuine_models, max_distance, malicious_model):
    malicious_distance = 0.0

    for genuine_model in all_genuine_models:
        malicious_distance += compute_distance(malicious_model, genuine_model) ** 2

    return malicious_distance < max_distance


### ATTACK SCHEMES


def calculate_mean_and_std(state_dicts):
    mean_std_dict = {}

    for key in state_dicts[0].keys():
        if state_dicts[0][key].dtype != torch.long:
            params = torch.stack([state_dict[key] for state_dict in state_dicts])
            mean_val = torch.mean(params, dim=0)
            std_val = torch.std(params, dim=0)
            sign_val = torch.sign(mean_val)
            mean_std_dict[key] = {
                'mean': mean_val,
                'std': std_val,
                'sign': sign_val
            }

    return mean_std_dict


def create_opt_fang_model(state_dict, all_genuine_models, gamma=50.0, tau=1.0):
    if len(all_genuine_models) <= 1:
        return state_dict

    malicious_model = None
    mean_std_dict = calculate_mean_and_std(all_genuine_models)

    max_distance = 0.0
    for i in range(len(all_genuine_models) - 1):
        for j in range(i + 1, len(all_genuine_models)):
            distance = compute_distance(all_genuine_models[i], all_genuine_models[j])
            if distance > max_distance:
                max_distance = distance

    step = gamma
    gamma_succ = 0.0

    while abs(gamma_succ - gamma) > tau:
        print(f"Gamma is {gamma}")
        malicious_model = all_genuine_models[0]
        for key, stats in mean_std_dict.items():
            # benign_mean + gamma * perturbation
            malicious_model[key] = stats['mean'] - gamma * stats['sign']

        if check_min_max_distance(all_genuine_models, max_distance, malicious_model):
            gamma_succ = gamma
            gamma = gamma + step / 2
        else:
            gamma = gamma - step / 2
        step = step / 2

    return malicious_model


def create_min_max_model(state_dict, all_genuine_models, gamma=50.0, tau=1.0):
    if len(all_genuine_models) <= 1:
        return state_dict

    malicious_model = None
    mean_std_dict = calculate_mean_and_std(all_genuine_models)

    max_distance = 0.0
    for i in range(len(all_genuine_models) - 1):
        for j in range(i + 1, len(all_genuine_models)):
            distance = compute_distance(all_genuine_models[i], all_genuine_models[j])
            if distance > max_distance:
                max_distance = distance

    step = gamma
    gamma_succ = 0.0

    while abs(gamma_succ - gamma) > tau:
        print(f"Gamma is {gamma}")
        malicious_model = all_genuine_models[0]
        for key, stats in mean_std_dict.items():
            # benign_mean + gamma * perturbation
            malicious_model[key] = stats['mean'] - gamma * stats['std']

        if check_min_max_distance(all_genuine_models, max_distance, malicious_model):
            gamma_succ = gamma
            gamma = gamma + step / 2
        else:
            gamma = gamma - step / 2
        step = step / 2

    return malicious_model


def create_min_sum_model(state_dict, all_genuine_models, gamma=50.0, tau=1.0):
    if len(all_genuine_models) <= 1:
        return state_dict

    malicious_model = None
    mean_std_dict = calculate_mean_and_std(all_genuine_models)

    max_distance = 0.0
    for i in range(len(all_genuine_models)):
        sum_distance = 0.0
        for j in range(len(all_genuine_models)):
            if i == j:
                continue
            sum_distance += compute_distance(all_genuine_models[i], all_genuine_models[j]) ** 2

        if sum_distance > max_distance:
            max_distance = sum_distance

    step = gamma
    gamma_succ = 0.0

    while abs(gamma_succ - gamma) > tau:
        print(f"Gamma is {gamma}")
        malicious_model = all_genuine_models[0]
        for key, stats in mean_std_dict.items():
            # benign_mean + gamma * perturbation
            malicious_model[key] = stats['mean'] - gamma * stats['std']

        if check_min_sum_distance(all_genuine_models, max_distance, malicious_model):
            gamma_succ = gamma
            gamma = gamma + step / 2
        else:
            gamma = gamma - step / 2
        step = step / 2

    return malicious_model


def create_LIE_state_dict(list_dict, scaling_factor=0.74):
    mean_std_dict = calculate_mean_and_std(list_dict)
    malicious_state_dict = list_dict[0]

    for key, stats in mean_std_dict.items():
        malicious_state_dict[key] = stats['mean'] + scaling_factor * stats['std']

    return malicious_state_dict

#defence 

def cosine_similarity(state_dict1, state_dict2):
    vector1 = torch.cat([param.flatten() for param in state_dict1.values()])
    vector2 = torch.cat([param.flatten() for param in state_dict2.values()])
    cos = torch.nn.CosineSimilarity(dim=0)
    return cos(vector1, vector2)

# shieldFL
def state_dict_to_vector(state_dict):
    return torch.cat([param.flatten() for param in state_dict.values()])

def byzantine_tolerance_aggregation(models, threshold=0.9):
    """Lọc mô hình độc hại dựa trên độ tương đồng cosine."""
    if len(models) == 0:
        return None
    global_weights = models[0]  # Giả sử mô hình đầu tiên là mô hình gốc
    filtered_models = []

    for weights in models:
        sim = cosine_similarity(global_weights, weights)
        if sim >= threshold:
            filtered_models.append(weights)

    if len(filtered_models) == 0:
        print("[Warning] No model passed the similarity threshold. Using fallback.")
        filtered_models = models  # Dùng toàn bộ nếu không có model nào hợp lệ

    avg_weights = {}
    for key in global_weights.keys():
        avg_weights[key] = sum(w[key] for w in filtered_models) / len(filtered_models)

    return avg_weights

def get_weight_vector(state_dict):
    """Chuyển trọng số của mô hình từ state_dict sang vector."""
    weight_vector = []
    for key in state_dict:
        weight_vector.extend(state_dict[key].cpu().numpy().flatten())
    return np.array(weight_vector)

def train_gmm_model(benign_gradients, malicious_gradients, n_components=2):
    """Huấn luyện mô hình Gaussian Mixture Model (GMM)."""
    all_gradients = np.vstack((benign_gradients, malicious_gradients))
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm_model = gmm.fit(all_gradients)
    return gmm_model

#trimmed mean defence 
import torch

def trimmed_mean_aggregation(models, trim_ratio=0.1):
    """
    Thực hiện trimmed mean aggregation trên danh sách các state_dict.
    
    Args:
        models (List[dict]): List các state_dict của các client models.
        trim_ratio (float): Tỉ lệ bị loại bỏ ở mỗi đầu (ví dụ 0.1 = 10%).

    Returns:
        dict: state_dict sau khi trimmed mean.
    """
    if not models:
        return None

    num_clients = len(models)
    trim_k = int(num_clients * trim_ratio)
    
    if 2 * trim_k >= num_clients:
        raise ValueError("Số lượng client quá ít so với trim_ratio đã chọn.")

    result = {}

    for key in models[0].keys():
        # Stack tất cả tensors cùng key thành 1 tensor 2D: (num_clients, ...)
        stacked = torch.stack([model[key] for model in models], dim=0)
        
        # Flatten theo batch, sort theo giá trị từng phần tử
        sorted_vals, _ = torch.sort(stacked, dim=0)
        
        # Trim: bỏ trim_k nhỏ nhất và trim_k lớn nhất
        trimmed = sorted_vals[trim_k : num_clients - trim_k]

        # Tính trung bình sau khi trim
        result[key] = torch.mean(trimmed, dim=0)

    return result

def calculate_md(gradient, mean_vector, covariance_matrix):
    """Tính Mahalanobis Distance giữa gradient và phân phối."""
    delta = gradient - mean_vector
    inv_covmat = np.linalg.inv(covariance_matrix)
    md = np.sqrt(np.dot(np.dot(delta.T, inv_covmat), delta))
    return md

def verify_gradient(gradient, gmm_model, threshold):
    """Kiểm tra gradient là benign hay malicious."""
    probabilities = gmm_model.predict_proba([gradient])[0]
    cluster = np.argmax(probabilities)
    mean_vector = gmm_model.means_[cluster]
    covariance_matrix = gmm_model.covariances_[cluster]

    md = calculate_md(gradient, mean_vector, covariance_matrix)

    if md > threshold:
        return "malicious"
    else:
        return "benign"


def krum(vectors, f):
    """
    Thực hiện thuật toán Krum để chọn trọng số "an toàn" nhất.
    vectors: list các vector torch.Tensor (flatten model weights).
    f: số lượng client độc hại giả định.
    """
    vectors_np = [v.detach().cpu().numpy() for v in vectors]
    n = len(vectors_np)
    scores = []

    for i in range(n):
        distances = [np.linalg.norm(vectors_np[i] - vectors_np[j])**2 for j in range(n) if j != i]
        closest = sorted(distances)[:n - f - 2]
        scores.append(sum(closest))

    selected = vectors[np.argmin(scores)]
    return selected

def median_aggregation(models):
    """
    Thực hiện phép gộp trung vị (median) trên danh sách các mô hình (state_dict).
    :param models: List các state_dict.
    :return: state_dict sau khi gộp.
    """
    if not models:
        return None

    result = {}
    for key in models[0].keys():
        stacked = torch.stack([model[key] for model in models])
        result[key] = torch.median(stacked, dim=0).values
    return result

# fltracer
def flatten_state_dict(state_dict):
    return torch.cat([v.view(-1).float().cpu() for v in state_dict.values()])

def fltracer_detect_anomalies(weight_matrix, threshold=2.5):
    pca = PCA(n_components=1)
    z = pca.fit_transform(weight_matrix).squeeze()
    mad = MAD(z)
    median = np.median(z)
    scores = np.abs(z - median) / (1.4826 * mad + 1e-6)
    return np.where(scores > threshold)[0]

# scionfl
def quantize_vector(vec):
    smin, smax = vec.min(), vec.max()
    probs = (vec - smin) / (smax - smin + 1e-6)
    sigma = torch.bernoulli(probs)
    return sigma, smin.item(), smax.item()

def l2_norm(sigma, smin, smax):
    count_1 = sigma.sum().item()
    count_0 = len(sigma) - count_1
    return np.sqrt(count_0 * smin**2 + count_1 * smax**2)

def dequantize(sigma, smin, smax):
    return smin + sigma * (smax - smin)

def cosine_distance(vec1, vec2):
    return 1 - torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

# hyper detection

def cosine(previous_embeddings, current_embedding):
    # old embedding
    embeddings_normal_history = np.array(previous_embeddings)
    E_norm_nor = embeddings_normal_history / np.linalg.norm(embeddings_normal_history, axis=1, keepdims=True)
    # new embedding
    em_test = current_embedding / np.linalg.norm(current_embedding, axis=1, keepdims=True)
    E_norm_mean = np.mean(E_norm_nor, axis=0)
    cosine_similarity = np.dot(em_test, E_norm_mean) / (np.linalg.norm(em_test) * np.linalg.norm(E_norm_mean))

    # Tính cosine similarity cho từng embedding trong history
    cosine_history = np.sum(embeddings_normal_history * E_norm_mean, axis=1) / (
        np.linalg.norm(embeddings_normal_history, axis=1) * np.linalg.norm(E_norm_mean)
    )

    # Tính ngưỡng động
    mu = np.mean(cosine_history)
    sigma = np.std(cosine_history)
    sigma = max(sigma, 1e-6)
    
    k = 2
    threshold = mu - k * sigma
    anomalies = cosine_similarity < threshold

    if anomalies: print("Anomalies detection !!!")

    return anomalies


def DBSCAN_phase2(embeddings_before, embeddings_after, selected_clients, n_components=3, eps=0.008, min_samples=3):
    embeddings_ = np.array([embeddings_after[client] for client in selected_clients])
    embeddings = np.array([embeddings_before[client] for client in selected_clients])
    delta = embeddings_ - embeddings
    delta = delta.squeeze() 
    pca = PCA(n_components=n_components)
    delta_3d = pca.fit_transform(delta)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(delta_3d)

    labels = clustering.labels_

    outliers = np.where(labels == -1)[0]
    outlier_clients = [selected_clients[i] for i in outliers]
    # print("Các client bị tấn công ( phân cụm ):", [selected_clients[i] for i in outliers])

    print("Các client bị tấn công ( phân cụm ):", outlier_clients)
    return outlier_clients
