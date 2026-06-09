"""
PINN 入门代码：用 PyTorch 求解一阶常微分方程

问题：
    du/dt = -u,  t in [0, 3]
    u(0) = 1

解析解：
    u(t) = exp(-t)

运行方式：
    python3 pinn_ode_intro_torch.py

依赖：
    pip install torch numpy matplotlib

这个脚本对应 PINN_入门教程.ipynb 的一阶 ODE 示例，但写成完整
Python 文件，并加入更详细的中文注释，适合作为后续学习 PDE/热传导方程
之前的最小可运行模板。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# =========================
# 1. 基本设置
# =========================

# 固定随机种子，保证每次运行时初始化和采样点基本一致。
# 这对学习阶段很重要：如果结果变了，优先怀疑代码改动，而不是随机性。
torch.manual_seed(42)
np.random.seed(42)

# 自动选择运行设备。
# 有 NVIDIA GPU 且 PyTorch 安装了 CUDA 版本时使用 cuda，否则使用 cpu。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出目录：训练结果图会保存到这里。
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")


# =========================
# 2. 定义神经网络
# =========================


class PINN(nn.Module):
    """
    用一个全连接神经网络表示未知函数 u(t)。

    普通监督学习中，网络通常学习 x -> y 的数据映射。
    PINN 中，网络学习的是连续函数：

        t -> u_theta(t)

    其中 theta 是神经网络的全部参数。

    本例：
        输入维度：1，对应时间 t
        输出维度：1，对应 u(t)
        隐藏层：3 层，每层 20 个神经元
        激活函数：Tanh

    为什么用 Tanh？
        PINN 需要对网络输出求导。Tanh 是光滑函数，适合自动微分。
        ReLU 虽然常见，但二阶导几乎处处为 0，不适合很多 PDE 问题。
    """

    def __init__(self, hidden_dim=20):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Xavier 初始化是一种常用的权重初始化方法。
        # 对简单问题不是必须，但可以让训练更稳定。
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, t):
        return self.net(t)


# =========================
# 3. 定义 PINN 损失函数
# =========================


def pinn_loss(model, t_pde, t_ic, u_ic, lambda_ic=10.0):
    """
    计算 PINN 的总损失。

    本问题有两个约束：

    1. 微分方程约束：
           du/dt = -u

       移项后写成残差形式：
           du/dt + u = 0

       如果网络预测完全满足方程，则 residual = du/dt + u 应该处处为 0。

    2. 初始条件约束：
           u(0) = 1

    参数说明：
        model:
            神经网络，输入 t，输出 u(t)

        t_pde:
            区间内部采样点，用来检查微分方程是否成立。
            shape 为 [N_pde, 1]

        t_ic:
            初始条件采样点。本例只有 t=0。
            shape 为 [1, 1]

        u_ic:
            初始条件真实值。本例为 u(0)=1。
            shape 为 [1, 1]

        lambda_ic:
            初始条件损失的权重。
            如果初始点拟合不好，可以适当增大这个值。

    返回：
        total_loss:
            用于反向传播的总损失，必须保持为 torch.Tensor。

        loss_pde:
            方程残差损失，供打印和记录。

        loss_ic:
            初始条件损失，供打印和记录。
    """

    # PINN 最关键的一步：让输入 t_pde 支持自动求导。
    # 后面我们要计算 du/dt，也就是网络输出对输入 t 的导数。
    t_pde = t_pde.clone().detach().requires_grad_(True)

    # 网络预测 u_theta(t)。
    u_pred = model(t_pde)

    # 用 PyTorch 自动微分计算 du/dt。
    # outputs=u_pred: 要被求导的量
    # inputs=t_pde: 对哪个变量求导
    # grad_outputs=ones_like(u_pred): 因为 u_pred 不是标量，所以需要指定权重
    # create_graph=True: 保留计算图，使得之后还能对 loss 反向传播
    du_dt = torch.autograd.grad(
        outputs=u_pred,
        inputs=t_pde,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
    )[0]

    # 对方程 du/dt = -u 移项，得到 residual = du/dt + u。
    # PINN 不直接需要解析解，而是最小化这个 residual。
    residual = du_dt + u_pred

    # PDE/ODE 残差损失：希望 residual 在所有内部点上都接近 0。
    loss_pde = torch.mean(residual**2)

    # 初始条件损失：希望网络在 t=0 的输出接近 1。
    u_ic_pred = model(t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)

    # 总损失：把物理方程约束和初始条件约束合在一起。
    total_loss = loss_pde + lambda_ic * loss_ic

    return total_loss, loss_pde.detach(), loss_ic.detach()


# =========================
# 4. 准备训练点
# =========================


def make_training_points(t_min=0.0, t_max=3.0, n_pde=200):
    """
    生成训练点。

    注意：
        这里没有生成大量带标签的数据点。
        PINN 的训练主要依赖：
            1. 微分方程在区间内部成立；
            2. 初始/边界条件成立。

    t_pde:
        在 [t_min, t_max] 内随机采样的点，用于计算方程残差。

    t_ic, u_ic:
        初始条件点和值。
    """

    # 区间内部点：随机采样。
    t_pde = torch.rand(n_pde, 1) * (t_max - t_min) + t_min

    # 初始条件：t=0, u=1。
    t_ic = torch.tensor([[0.0]])
    u_ic = torch.tensor([[1.0]])

    # 把所有张量放到同一个 device。
    # 如果模型在 GPU，输入也必须在 GPU；如果模型在 CPU，输入也必须在 CPU。
    return t_pde.to(device), t_ic.to(device), u_ic.to(device)


# =========================
# 5. 训练模型
# =========================


def train():
    """训练 PINN，并返回训练好的模型与损失历史。"""

    # 创建训练点。
    t_min, t_max = 0.0, 3.0
    t_pde, t_ic, u_ic = make_training_points(t_min=t_min, t_max=t_max, n_pde=200)

    print(f"PDE points shape: {tuple(t_pde.shape)}")
    print(f"IC point: t = {t_ic.item():.1f}, u = {u_ic.item():.1f}")

    # 创建模型，并放到 device 上。
    model = PINN(hidden_dim=20).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {n_params}")

    # Adam 是 PINN 入门阶段最常用的优化器。
    # 学习率太大会震荡，太小会收敛慢；本例 0.01 通常可以跑通。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 3000
    lambda_ic = 10.0

    # 用列表记录每轮损失，后面画损失曲线。
    history = {
        "total": [],
        "pde": [],
        "ic": [],
    }

    for epoch in range(1, n_epochs + 1):
        # 1. 清空上一轮累积的参数梯度。
        optimizer.zero_grad()

        # 2. 前向计算损失。
        total_loss, loss_pde, loss_ic = pinn_loss(
            model=model,
            t_pde=t_pde,
            t_ic=t_ic,
            u_ic=u_ic,
            lambda_ic=lambda_ic,
        )

        # 3. 反向传播：计算 total_loss 对网络参数的梯度。
        total_loss.backward()

        # 4. 优化器根据梯度更新网络参数。
        optimizer.step()

        # 5. 记录损失数值。
        history["total"].append(total_loss.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())

        # 每隔一段时间打印一次，观察是否收敛。
        if epoch == 1 or epoch % 500 == 0:
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"total = {total_loss.item():.6e} | "
                f"pde = {loss_pde.item():.6e} | "
                f"ic = {loss_ic.item():.6e}"
            )

    return model, history, (t_min, t_max), (t_pde, t_ic, u_ic)


# =========================
# 6. 评估与画图
# =========================


def evaluate_and_plot(model, history, domain, training_points):
    """
    与解析解比较，并保存结果图。

    对入门算例来说，解析解很重要：
        它可以帮助我们确认 PINN 是否真的学对了。

    复杂 PDE 或真实物理问题通常没有解析解，此时可对比：
        1. 传统数值解；
        2. 观测数据；
        3. 方程残差分布；
        4. 守恒量误差。
    """

    t_min, t_max = domain
    t_pde, t_ic, u_ic = training_points

    model.eval()

    # 构造均匀测试点，用于画曲线和计算误差。
    t_test = torch.linspace(t_min, t_max, 200).reshape(-1, 1).to(device)

    # 评估阶段不需要计算参数梯度，因此使用 torch.no_grad() 节省内存。
    with torch.no_grad():
        u_pred = model(t_test)

    # 转为 CPU 上的 numpy 数组，供 matplotlib 和 numpy 使用。
    t_np = t_test.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()

    # 解析解：u(t)=exp(-t)。
    u_exact_np = np.exp(-t_np)

    # 误差指标。
    # L2 相对误差越小越好，是 PINN/PDE 论文里常见指标。
    relative_l2_error = np.linalg.norm(u_pred_np - u_exact_np) / np.linalg.norm(u_exact_np)
    max_abs_error = np.max(np.abs(u_pred_np - u_exact_np))

    print("\nEvaluation:")
    print(f"Relative L2 error: {relative_l2_error:.6e}")
    print(f"Max absolute error: {max_abs_error:.6e}")

    # 画两张图：
    # 左：损失曲线
    # 右：PINN 预测值与解析解
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(history["total"], label="Total loss")
    ax.semilogy(history["pde"], label="ODE residual loss")
    ax.semilogy(history["ic"], label="Initial condition loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(t_np, u_exact_np, "b-", linewidth=2, label="Exact: exp(-t)")
    ax.plot(t_np, u_pred_np, "r--", linewidth=2, label="PINN prediction")

    # 标出 PDE 训练点的位置。纵坐标放在 0，只是为了看采样点分布。
    t_pde_np = t_pde.detach().cpu().numpy()
    ax.scatter(t_pde_np, np.zeros_like(t_pde_np), s=8, alpha=0.25, label="PDE points")

    # 标出初始条件点。
    ax.scatter(
        t_ic.detach().cpu().numpy(),
        u_ic.detach().cpu().numpy(),
        s=80,
        c="green",
        marker="*",
        label="Initial condition",
        zorder=5,
    )

    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.set_title("PINN vs exact solution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    figure_path = output_dir / "pinn_ode_intro_result.png"
    fig.savefig(figure_path, dpi=200)
    print(f"Figure saved to: {figure_path}")

    # 如果在本地有图形界面，也可以取消下面这一行的注释，直接弹出图窗。
    # plt.show()


def main():
    model, history, domain, training_points = train()
    evaluate_and_plot(model, history, domain, training_points)


if __name__ == "__main__":
    main()
