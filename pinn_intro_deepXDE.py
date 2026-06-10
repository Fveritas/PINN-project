"""
DeepXDE 入门代码：用 PINN 求解一阶常微分方程

问题：
    du/dt = -u,  t in [0, 3]
    u(0) = 1

解析解：
    u(t) = exp(-t)

运行方式：
    cd /home/guiyu/workspace/PINN
    source .venv/bin/activate
    python pinn_intro_deepXDE.py

这个脚本和 pinn_ode_intro_torch.py 求解同一个问题。

区别：
    pinn_ode_intro_torch.py  手写 PyTorch 训练循环、采样点和 autograd
    pinn_intro_deepXDE.py    使用 DeepXDE 封装几何区域、初始条件、采样和训练
"""

from pathlib import Path

import os

# DeepXDE 支持多个后端。这里强制使用 PyTorch 后端。
# 注意：必须在 import deepxde 之前设置。
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np


# =========================
# 1. 基本设置
# =========================

# 固定随机种子，方便复现实验结果。
dde.config.set_random_seed(42)

# 输出目录：结果图保存到 outputs/。
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)


# =========================
# 2. 定义微分方程 residual
# =========================


def ode_residual(t, u):
    """
    定义 ODE 残差。

    原方程：
        du/dt = -u

    移项得到 residual：
        du/dt + u = 0

    DeepXDE 会在采样点 t 上调用这个函数。

    参数：
        t:
            输入点，shape 通常为 [N, 1]

        u:
            神经网络预测值 u_theta(t)，shape 通常为 [N, 1]

    返回：
        residual:
            方程残差。训练目标是让 residual 尽量接近 0。
    """

    # dde.grad.jacobian 用来计算一阶导数。
    # 这里 u 只有一个输出维度，t 只有一个输入维度。
    # i=0 表示第 0 个输出分量 u。
    # j=0 表示对第 0 个输入分量 t 求导。
    du_dt = dde.grad.jacobian(u, t, i=0, j=0)

    return du_dt + u


# =========================
# 3. 定义初始条件
# =========================


def initial_value(t):
    """
    初始条件的函数值。

    本问题初始条件是：
        u(0) = 1

    DeepXDE 会把满足 initial_boundary 的点传进来。
    返回值 shape 要和网络输出一致，即 [N, 1]。
    """

    return np.ones((len(t), 1))


def initial_boundary(t, on_initial):
    """
    判断哪些点属于初始点。

    对 TimeDomain(0, 3) 来说，on_initial 表示 t=0。
    """

    return on_initial


# =========================
# 4. 构造 DeepXDE 问题
# =========================


# 时间区域 t in [0, 3]。
time_domain = dde.geometry.TimeDomain(0.0, 3.0)

# 初始条件 u(0)=1。
ic = dde.icbc.IC(
    geom=time_domain,
    func=initial_value,
    on_initial=initial_boundary,
)

# 训练数据对象。
# DeepXDE 里的 "data" 不一定是观测数据；这里主要是：
#   1. num_domain 个区间内部点，用来计算 ODE residual；
#   2. num_boundary 个边界/初始点，用来约束 u(0)=1；
#   3. solution 用于测试误差，不参与训练。
data = dde.data.PDE(
    geometry=time_domain,
    pde=ode_residual,
    bcs=[ic],
    num_domain=200,
    num_boundary=1,
    solution=lambda t: np.exp(-t),
    num_test=200,
)


# =========================
# 5. 定义神经网络
# =========================


# layer_sizes:
#   [1, 20, 20, 20, 1]
#
# 含义：
#   输入维度 1：t
#   隐藏层 3 层，每层 20 个神经元
#   输出维度 1：u(t)
net = dde.nn.FNN(
    layer_sizes=[1, 20, 20, 20, 1],
    activation="tanh",
    kernel_initializer="Glorot normal",
)


# =========================
# 6. 编译和训练模型
# =========================


model = dde.Model(data, net)

# 使用 Adam 优化器。
# metrics=["l2 relative error"] 表示训练过程中计算相对 L2 误差。
model.compile(
    optimizer="adam",
    lr=0.01,
    metrics=["l2 relative error"],
)

print("Start training DeepXDE PINN...")
loss_history, train_state = model.train(iterations=3000, display_every=500)
print("Training finished.")


# =========================
# 7. 评估与画图
# =========================


# 构造测试点。
t_test = np.linspace(0.0, 3.0, 200).reshape(-1, 1)

# DeepXDE 预测。
u_pred = model.predict(t_test)

# 解析解。
u_exact = np.exp(-t_test)

# 误差指标。
relative_l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
max_abs_error = np.max(np.abs(u_pred - u_exact))

print("\nEvaluation:")
print(f"Relative L2 error: {relative_l2_error:.6e}")
print(f"Max absolute error: {max_abs_error:.6e}")


# 画预测结果。
plt.figure(figsize=(7, 4))
plt.plot(t_test, u_exact, "b-", linewidth=2, label="Exact: exp(-t)")
plt.plot(t_test, u_pred, "r--", linewidth=2, label="DeepXDE PINN")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("DeepXDE PINN vs exact solution")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

figure_path = output_dir / "pinn_intro_deepXDE_result.png"
plt.savefig(figure_path, dpi=200)
print(f"Figure saved to: {figure_path}")

