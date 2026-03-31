import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
import time

def skew_symmetric(v):
    """生成向量v的斜对称矩阵"""
    zero = torch.zeros_like(v[0])
    return torch.stack([
        torch.stack([zero, -v[2], v[1]]),
        torch.stack([v[2], zero, -v[0]]),
        torch.stack([-v[1], v[0], zero])
    ])

def se3_exp(xi, device):
    """将李代数转换为SE(3)变换矩阵（更精确的实现）"""
    rot_vec = xi[:3]
    t = xi[3:]
    angle = torch.norm(rot_vec)
    
    # 处理零旋转
    if angle < 1e-6:
        R = torch.eye(3, device=device)
        J = torch.eye(3, device=device)
    else:
        axis = rot_vec / angle
        K = skew_symmetric(axis)
        R = torch.eye(3, device=device) + torch.sin(angle)*K + (1-torch.cos(angle))*(K@K)
        # 计算右雅可比矩阵
        J = (torch.sin(angle)/angle)*torch.eye(3, device=device) + \
            (1 - torch.sin(angle)/angle)*torch.outer(axis, axis) + \
            ((1 - torch.cos(angle))/angle)*K
    
    # 计算平移部分
    V = J @ t.reshape(3,1)
    return R, V.flatten()

class SinkhornDistance(nn.Module):
    """改进的熵正则化Sinkhorn距离计算（包含协方差项）"""
    def __init__(self, epsilon=1, max_iter=100):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def compute_cost_matrix(self, mu_A, cov_A, mu_B, cov_B):
        """
        W2距离矩阵的优化计算（GPU并行版，分块处理）
        """
        ****EMPTY****
        return mean_term + covariance_term

    def matrix_sqrt_eigh(self, M):
        """
        使用特征分解计算对称矩阵的平方根（添加正则化确保数值稳定性）
        """
        epsilon = 1e-6 * torch.eye(M.size(-1), device=M.device, dtype=M.dtype)
        M_reg = M + epsilon
        eigenvalues, eigenvectors = torch.linalg.eigh(M_reg)
        sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0.0))
        sqrt_M = eigenvectors @ (sqrt_eigenvalues.unsqueeze(-1) * eigenvectors.transpose(-2, -1))
        return sqrt_M


    def forward(self, mu_A, cov_A, w_A, mu_B, cov_B, w_B):
        C = self.compute_cost_matrix(mu_A, cov_A, mu_B, cov_B)
        # Sinkhorn算法（对数域稳定实现）
        time_1 = time.perf_counter()
        log_a = torch.log(w_A + 1e-10).to(mu_A.device)
        log_b = torch.log(w_B + 1e-10).to(mu_B.device)
        log_K = -C / self.epsilon
        
        log_u = torch.zeros_like(log_a).to(mu_A.device)
        log_v = torch.zeros_like(log_b).to(mu_B.device)
        tolerance = 1e-5
        prev_log_v = log_v.clone()
        for _ in range(self.max_iter):
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
            if torch.max(torch.abs(log_v - prev_log_v)) < tolerance:
                print("break",_)
                break
            prev_log_v = log_v.clone()
        
        # 稳定计算 pi
        log_pi = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        max_log = torch.max(log_pi)
        log_pi_stable = log_pi - max_log
        pi = torch.exp(log_pi_stable)
        #pi = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
        return torch.sum(pi * C)

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵（数值稳定实现）"""
    q = q / q.norm()  # 确保单位四元数
    
    w, x, y, z = q[0], q[1], q[2], q[3]
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z

    R = torch.stack([
        1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy),
        2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx),
        2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)
    ]).reshape(3, 3)
    
    return R

class GaussianRegistration(nn.Module):
    """支持多种参数化方式的高斯配准模型"""
    def __init__(self, param_method='se3', device=None, lambda_reg=0.1,
                 xi_init=None, q_init=None, t_init=None, axis_angle_init=None):
        super().__init__()
        self.device = device
        self.param_method = param_method
        self.lambda_reg = lambda_reg

        # 初始化参数 --------------------------------------------------
        if param_method == 'se3':  # 李代数参数化
            if xi_init is not None:
                self.xi = nn.Parameter(torch.tensor(xi_init, device=device, dtype=torch.float32))
            else:
                self.xi = nn.Parameter(torch.zeros(6, device=device, dtype=torch.float32))
                
        elif param_method == 'quaternion':  # 四元数+平移参数化
            if q_init is not None:
                q = torch.tensor(q_init, device=device, dtype=torch.float32)
                q = q / q.norm()  # 单位化
            else:
                q = torch.tensor([1.0, 0, 0, 0], device=device, dtype=torch.float32)
            self.q = nn.Parameter(q)
            
            if t_init is not None:
                self.t = nn.Parameter(torch.tensor(t_init, device=device, dtype=torch.float32))
            else:
                self.t = nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
                
        elif param_method == 'axis_angle':  # 轴角+平移参数化
            if axis_angle_init is not None:
                self.axis_angle = nn.Parameter(torch.tensor(axis_angle_init, device=device, dtype=torch.float32))
            else:
                self.axis_angle = nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
            
            if t_init is not None:
                self.t = nn.Parameter(torch.tensor(t_init, device=device, dtype=torch.float32))
            else:
                self.t = nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported parameterization method: {param_method}")

    def get_transformation(self):
        """获取当前变换参数（用于结果验证）"""
        with torch.no_grad():
            if self.param_method == 'se3':
                R, t = se3_exp(self.xi, self.device)
            elif self.param_method == 'quaternion':
                R = quaternion_to_rotation_matrix(self.q)
                t = self.t
            elif self.param_method == 'axis_angle':
                rot_vec = self.axis_angle
                angle = torch.norm(rot_vec)
                if angle < 1e-6:
                    R = torch.eye(3, device=self.device)
                else:
                    axis = rot_vec / angle
                    K = skew_symmetric(axis)
                    R = torch.eye(3, device=self.device) + torch.sin(angle)*K + (1-torch.cos(angle))*(K@K)
                t = self.t
            return R, t

    def forward(self, source_mu, source_cov, source_w, target_mu, target_cov, target_w, batch_size=5000):
        # 计算当前变换参数 --------------------------------------------
        if self.param_method == 'se3':
            R, t = se3_exp(self.xi, self.device)
        elif self.param_method == 'quaternion':
            R = quaternion_to_rotation_matrix(self.q)
            t = self.t
        elif self.param_method == 'axis_angle':
            rot_vec = self.axis_angle
            angle = torch.norm(rot_vec)
            if angle < 1e-6:
                R = torch.eye(3, device=self.device)
            else:
                axis = rot_vec / angle
                K = skew_symmetric(axis)
                R = torch.eye(3, device=self.device) + torch.sin(angle)*K + (1-torch.cos(angle))*(K@K)
            t = self.t

        # 应用变换（与原实现相同）---------------------------------------
        transformed_mu = (source_mu @ R.T) + t
        transformed_cov = torch.einsum('ij,njk,lk->nil', R, source_cov, R)
        
        # 子采样和损失计算（与原实现相同）-------------------------------
        idx_source = torch.randperm(transformed_mu.size(0))[:batch_size]
        idx_target = torch.randperm(target_mu.size(0))[:batch_size]
        
        sub_mu_A = transformed_mu[idx_source]
        sub_cov_A = transformed_cov[idx_source]
        sub_w_A = source_w[idx_source]
        sub_w_A = sub_w_A / sub_w_A.sum()
        
        sub_mu_B = target_mu[idx_target]
        sub_cov_B = target_cov[idx_target]
        sub_w_B = target_w[idx_target]
        sub_w_B = sub_w_B / sub_w_B.sum()
        
        sinkhorn = SinkhornDistance(epsilon=0.1)
        sinkhorn_loss = sinkhorn(sub_mu_A, sub_cov_A, sub_w_A, sub_mu_B, sub_cov_B, sub_w_B)
        reg_loss = self.lambda_reg * torch.sum(torch.abs(sub_w_A))
        return sinkhorn_loss + reg_loss


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_gmm(mu, cov, ax, color='b', alpha=0.5, label=None):
    """
    可视化高斯球（每个高斯分布的3D球体），传入均值和协方差矩阵
    """
    # 绘制每个高斯分布的球体
    for i in range(mu.shape[0]):
        mean = mu[i].detach().cpu().numpy()
        cov_matrix = cov[i].detach().cpu().numpy()
        
        # 生成球体点
        u, s, v = torch.svd(torch.tensor(cov_matrix))  # 奇异值分解
        radii = torch.sqrt(s)  # 使用奇异值作为标准差
        u = u.detach().cpu().numpy()

        # 参数化球体
        phi = torch.linspace(0, torch.pi, 20)
        theta = torch.linspace(0, 2 * torch.pi, 20)
        phi, theta = torch.meshgrid(phi, theta, indexing='ij')  # 添加 indexing 参数以消除警告
        x = radii[0] * torch.sin(phi) * torch.cos(theta)
        y = radii[1] * torch.sin(phi) * torch.sin(theta)
        z = radii[2] * torch.cos(phi)

        # 变换到中心
        x = x + mean[0]
        y = y + mean[1]
        z = z + mean[2]

        # 绘制球体
        ax.plot_surface(x, y, z, color=color, alpha=alpha, label=label)
    
    if label:
        ax.set_title(label)

def log_SO3(R):
    """
    计算旋转矩阵 R 的李代数表示（旋转部分）
    使用 Rodrigues' 公式将旋转矩阵转换为李代数
    """
    # 计算旋转矩阵的角度和旋转轴
    cos_theta = 0.5 * (torch.trace(R) - 1)
    theta = torch.acos(cos_theta)  # 旋转角度
    if theta < 1e-6:  # 处理旋转角度接近零的情况
        return torch.zeros(3)
    
    r = (R - R.T) / (2 * torch.sin(theta))
    omega = torch.tensor([r[2, 1], r[0, 2], r[1, 0]], dtype=torch.float32)
    
    # 计算李代数的旋转部分
    return omega * theta

def se3_log(R, t):
    """
    将旋转矩阵 R 和平移向量 t 转换为李代数的表示
    """
    omega = log_SO3(R)
    
    # 李代数的平移部分
    v = t
    
    # 返回李代数的组合表示
    xi = torch.cat((omega, v), dim=0)  # 6维李代数向量
    return xi

def visualize_gmm(source_mu, source_cov, target_mu, target_cov, transformed_mu, transformed_cov, save_path):
    """
    可视化配准前后的高斯球体，并保存为图片
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 可视化初始化的高斯球（源）
    plot_gmm(source_mu, source_cov, ax, color='r', alpha=0.3, label="Source GMM (Before Registration)")

    # 可视化目标高斯球（目标）
    plot_gmm(target_mu, target_cov, ax, color='g', alpha=0.3, label="Target GMM")

    # 可视化配准后的高斯球（源）
    plot_gmm(transformed_mu, transformed_cov, ax, color='b', alpha=0.6, label="Source GMM (After Registration)")

    # 添加图例
    ax.legend()

    # 保存图片
    plt.savefig(save_path)
    plt.close()  # 关闭图形，避免占用内存
       


# 数据生成函数（保持原样）
def generate_gmm(num_points, scale=1.0):
    mu = torch.randn(num_points, 3) * scale
    cov = torch.randn(num_points, 3, 3)
    cov = torch.bmm(cov, cov.transpose(1,2)) + torch.eye(3).unsqueeze(0)*0.1
    alpha = torch.softmax(torch.randn(num_points), dim=0)
    return mu, cov, alpha

from sklearn.cluster import KMeans

def simplify_gmm(mu, cov, w, num_clusters):
    # 记录原始设备
    mu = mu.cpu()
    cov = cov.cpu()
    w = w.cpu()
    original_device = mu.device

    # 将数据移到CPU进行聚类（sklearn需要numpy数据）
    mu_np = mu.cpu().numpy()
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(mu_np)
    
    new_mu, new_cov, new_w = [], [], []
    for i in range(num_clusters):
        # 创建设备正确的mask张量
        mask = (cluster_ids == i)
        mask_tensor = torch.tensor(mask, device=original_device)  # 关键修复：指定原始设备
        
        if not torch.any(mask_tensor):
            continue
            
        # 提取当前簇的成分（确保在原始设备）
        cluster_mu = mu[mask_tensor]
        cluster_cov = cov[mask_tensor]
        cluster_w = w[mask_tensor]
        total_weight = cluster_w.sum()
        
        # 计算合并后的均值（保持设备一致）
        merged_mu = (cluster_w[:, None] * cluster_mu).sum(dim=0) / total_weight
        
        # 计算合并后的协方差（保持设备一致）
        delta = cluster_mu - merged_mu.unsqueeze(0)
        expanded_cov = cluster_cov + torch.einsum('ni,nj->nij', delta, delta)
        merged_cov = (cluster_w[:, None, None] * expanded_cov).sum(dim=0) / total_weight
        
        new_mu.append(merged_mu)
        new_cov.append(merged_cov)
        new_w.append(total_weight)
    
    # 合并结果并确保设备正确
    new_mu = torch.stack(new_mu).to('cuda')
    new_cov = torch.stack(new_cov).to('cuda')
    new_w = torch.stack(new_w).to('cuda')
    new_w = new_w / new_w.sum()
    
    return new_mu, new_cov, new_w


def simplify_gmm_vectorized(mu, cov, w, num_clusters):
    original_device = mu.device
    # 将数据移到CPU进行聚类（sklearn需要numpy数据）
    mu_np = mu.cpu().numpy()
    w = w.to(original_device)
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(mu_np)
    cluster_ids_tensor = torch.tensor(cluster_ids, device=original_device, dtype=torch.long)
    
    # 计算每个簇的总权重
    total_weight_per_cluster = torch.bincount(cluster_ids_tensor, weights=w, minlength=num_clusters)
    
    # 计算合并后的均值（向量化）
    sum_mu = torch.zeros(num_clusters, mu.size(1), device=original_device)
    sum_mu.index_add_(0, cluster_ids_tensor, w[:, None] * mu)
    merged_mu_all = sum_mu / (total_weight_per_cluster[:, None] + 1e-8)  # 避免除以零
    
    # 计算delta矩阵（各成分均值与对应簇中心的差值）
    merged_mu_per_component = merged_mu_all[cluster_ids_tensor]
    delta = mu - merged_mu_per_component
    
    # 计算扩展协方差并合并（向量化）
    expanded_cov = cov + torch.einsum('ni,nj->nij', delta, delta)
    sum_cov = torch.zeros(num_clusters, mu.size(1), mu.size(1), device=original_device)
    sum_cov.index_add_(0, cluster_ids_tensor, w[:, None, None] * expanded_cov)
    merged_cov_all = sum_cov / (total_weight_per_cluster[:, None, None] + 1e-8)
    
    # 过滤空簇并归一化权重
    valid_clusters = total_weight_per_cluster > 1e-8
    new_mu = merged_mu_all[valid_clusters]
    new_cov = merged_cov_all[valid_clusters]
    new_w = total_weight_per_cluster[valid_clusters]
    new_w /= new_w.sum()  # 确保权重归一化
    
    return new_mu, new_cov, new_w

# 测试用例
if __name__ == "__main__":
    # 在测试用例中设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maxs=5000
    sou=5000
    # 生成数据
    target_mu, target_cov, target_alpha = generate_gmm(maxs, scale=1.0)
    target_w = target_alpha / target_alpha.sum()



    source_mu = target_mu[:sou].clone().to(device)
    source_cov = target_cov[:sou].clone().to(device)
    source_w = (target_alpha[:sou].clone() / target_alpha[:sou].sum()).to(device)
    
    target_mu = target_mu.to(device)
    target_cov = target_cov.to(device)
    target_alpha = target_alpha.to(device)
# 简化源和目标GMM

    # 定义真实变换
    theta = np.pi
    R_true = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32).to(device)
    t_true = torch.tensor([5.5, 5.0, 3.3], dtype=torch.float32).to(device)
    

    # 应用变换到源点云
    transformed_mu = (source_mu @ R_true.T) + t_true
    transformed_cov = torch.einsum('ij,njk,lk->nil', R_true, source_cov, R_true)
    # 初始化xi
    theta_init = np.pi/1.1
    R_init = torch.tensor([
        [np.cos(theta_init), -np.sin(theta_init), 0],
        [np.sin(theta_init), np.cos(theta_init), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    t_init = torch.tensor([5.5-2, 5.0-2, 3.3+1], dtype=torch.float32)

    # xi_init = se3_log(R_init, t_init)
    # #model = GaussianRegistration(xi_init, device).to(device)



    # # 李代数参数化（原始方式）
    # model = GaussianRegistration(param_method='se3', xi_init=xi_init, device=device)

    #四元数参数化（需要提供初始四元数和平移）
    theta_init = np.pi/1.1
    q_init = [np.cos(theta_init/2), 0, 0, np.sin(theta_init/2)]
    t_init = [5.5-2, 5.0-2, 3.3+1]
    model = GaussianRegistration(
        param_method='quaternion',
        q_init=q_init,
        t_init=t_init,
        device=device
    )

    # # 轴角参数化（需要提供旋转向量和平移）
    # axis_angle_init = [0, 0, theta_init]  # 绕z轴旋转
    # model = GaussianRegistration(
    #     param_method='axis_angle',
    #     axis_angle_init=axis_angle_init,
    #     t_init=t_init,
    #     device=device
    # )

    # 训练过程保持不变



    optimizer = optim.Adam(model.parameters(), lr=0.5)  # 降低学习率
    
    num_clusters = 100
    time_1 = time.time()
    target_mu, target_cov, target_w = simplify_gmm_vectorized(target_mu, target_cov, target_w, num_clusters)
    print(time.time() - time_1)
    transformed_mu, transformed_cov, source_w = simplify_gmm_vectorized(transformed_mu, transformed_cov, source_w, num_clusters)

    target_mu = target_mu.to("cuda")
    target_cov = target_cov.to("cuda")
    target_w = target_w.to("cuda") 

    transformed_mu = transformed_mu.to("cuda")
    transformed_cov = transformed_cov.to("cuda")
    source_w = source_w.to("cuda")

    # 训练循环
    for epoch in range(1000):  # 增加训练轮数
        optimizer.zero_grad()
        loss = model(transformed_mu, transformed_cov, source_w,
                     target_mu, target_cov, target_w)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} Loss: {loss.item():.4f}')
            with torch.no_grad():
                R_est, t_est = model.get_transformation()
                R_true_inv = R_true.T
                t_true_inv = -R_true_inv @ t_true
                


                R_error = torch.norm(R_est - R_true_inv).cpu()
                t_error = torch.norm(t_est - t_true_inv).cpu()
                
                print("\n=== 结果 ===")
                print(f"旋转误差: {R_error.item():.6f} rad")
                print(f"平移误差: {t_error.item():.6f} m")
                print("估计平移向量:", t_est.cpu().numpy())

            # 评估结果

    # 评估结果
    with torch.no_grad():
        R_est, t_est = model.get_transformation()
        R_true_inv = R_true.T
        t_true_inv = -R_true_inv @ t_true
        


        R_error = torch.norm(R_est - R_true_inv).cpu()
        t_error = torch.norm(t_est - t_true_inv).cpu()
        
    print("\n=== 最终结果 ===")
    print(f"旋转误差: {R_error.item():.6f} rad")
    print(f"平移误差: {t_error.item():.6f} m")
    print("估计旋转矩阵:\n", R_est.cpu().numpy())
    print("估计平移向量:", t_est.cpu().detach().numpy())

    transformed1_mu = (transformed_mu @ R_est.T) + t_est
    transformed1_cov = torch.einsum('ij,njk,lk->nil', R_est, transformed_cov, R_est)

    # 可视化结果
    visualize_gmm(transformed_mu, transformed_cov, source_mu, source_cov, transformed1_mu, transformed1_cov, "./output.jpg")