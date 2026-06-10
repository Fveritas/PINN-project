"""
DeepXDE 快速验证：等温 Parker 太阳风方程

目标：
    用 PINN 求解无量纲等温 Parker 太阳风方程，快速验证 DeepXDE 能否
    学到穿过临界点的 transonic solar wind 分支。

无量纲变量：
    x = r / r_c
    y = v / c_s

其中：
    r_c = GM_sun / (2 c_s^2) 是 Parker 临界半径；
    c_s 是等温声速；
    y 同时也是 Mach number。

无量纲方程：
    (y - 1/y) dy/dx = 2/x - 2/x^2

残差形式：
    residual = (y - 1/y) dy/dx - (2/x - 2/x^2)

临界点条件：
    y(1) = 1

运行方式：
    cd /home/guiyu/workspace/PINN
    source .venv/bin/activate
    python pinn_parker_solar_wind_deepXDE.py

说明：
    这是快速验证脚本，不是科研级太阳风模型。当前版本只解速度/Mach 数，
    不包含密度、温度、磁场、真实单位换算或观测数据同化。
"""

from pathlib import Path

import os

# 必须在 import deepxde 之前设置后端。
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import brentq


# =========================
# 1. 基本设置
# =========================

dde.config.set_random_seed(42)

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 求解区间。x=1 是临界点；x<1 为亚声速区域，x>1 为超声速区域。
X_MIN = 0.2
X_MAX = 30.0

# 为了避免 1/y 数值问题，网络输出经过 softplus 后再加一个很小正数。
Y_EPS = 1e-3


# =========================
# 2. Parker 方程与参考解
# =========================


def parker_rhs(x):
    """无量纲 Parker 方程右端：2/x - 2/x^2。"""

    return 2.0 / x - 2.0 / (x**2)


def parker_residual(x, y):
    """
    DeepXDE 使用的 ODE residual。

    y 已经是经过 output_transform 的正值速度/Mach 数。
    """

    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)
    return (y - 1.0 / y) * dy_dx - (2.0 / x - 2.0 / (x**2))


def parker_implicit_rhs(x):
    """
    Transonic Parker 解的隐式方程右端。

    从无量纲方程积分可得：
        y^2 - ln(y^2) = 4 ln(x) + 4/x - 3

    常数 -3 来自临界条件 x=1, y=1。
    """

    return 4.0 * np.log(x) + 4.0 / x - 3.0


def parker_reference_y(x):
    """
    计算 transonic 分支的参考解。

    x < 1：取亚声速根 0 < y < 1
    x = 1：y = 1
    x > 1：取超声速根 y > 1

    这个参考解只用于画图和远端软约束，不参与 PDE residual 的构造。
    """

    x = float(x)
    if np.isclose(x, 1.0):
        return 1.0

    rhs = parker_implicit_rhs(x)

    def equation(y):
        return y**2 - np.log(y**2) - rhs

    if x < 1.0:
        return brentq(equation, 1e-6, 1.0 - 1e-8)
    return brentq(equation, 1.0 + 1e-8, 20.0)


# =========================
# 3. DeepXDE 数据和约束
# =========================


geom = dde.geometry.Interval(X_MIN, X_MAX)

# 临界点约束 y(1)=1。这个约束用于让 PINN 通过 sonic point。
critical_points = np.array([[1.0]], dtype=np.float32)
critical_values = np.array([[1.0]], dtype=np.float32)
critical_bc = dde.icbc.PointSetBC(critical_points, critical_values, component=0)

# 远端软约束帮助网络选择超声分支。目标值来自 transonic Parker 隐式解。
far_points = np.array([[X_MAX]], dtype=np.float32)
far_target = parker_reference_y(X_MAX)
far_values = np.array([[far_target]], dtype=np.float32)
far_bc = dde.icbc.PointSetBC(far_points, far_values, component=0)

data = dde.data.PDE(
    geometry=geom,
    pde=parker_residual,
    bcs=[critical_bc, far_bc],
    num_domain=800,
    num_boundary=0,
    anchors=np.array([[1.0], [X_MAX]], dtype=np.float32),
    num_test=500,
)


# =========================
# 4. 神经网络
# =========================


net = dde.nn.FNN(
    layer_sizes=[1, 64, 64, 64, 1],
    activation="tanh",
    kernel_initializer="Glorot normal",
)


def positive_output_transform(x, raw_y):
    """
    把网络原始输出 raw_y 映射为正速度 y。

    Parker 方程包含 1/y，因此训练过程中必须避免 y<=0。
    softplus(raw_y) + Y_EPS 是一个简单稳定的正值参数化。
    """

    return F.softplus(raw_y) + Y_EPS


net.apply_output_transform(positive_output_transform)

model = dde.Model(data, net)

# loss 顺序：
#   1. PDE residual
#   2. critical_bc: y(1)=1
#   3. far_bc: y(X_MAX)=far_target
#
# 临界点权重大一些，远端约束较弱，主要用于选中超声分支。
model.compile(
    optimizer="adam",
    lr=0.003,
    loss_weights=[1.0, 50.0, 5.0],
)


# =========================
# 5. 训练
# =========================


print("Start training DeepXDE Parker solar wind PINN...")
print(f"Domain: x in [{X_MIN}, {X_MAX}]")
print(f"Critical condition: y(1) = 1")
print(f"Far soft constraint: y({X_MAX}) = {far_target:.6f}")

loss_history, train_state = model.train(iterations=8000, display_every=1000)

# 用 L-BFGS 做一次后处理优化，通常能进一步降低 residual。
model.compile(
    optimizer="L-BFGS",
    loss_weights=[1.0, 50.0, 5.0],
)
loss_history, train_state = model.train()

print("Training finished.")


# =========================
# 6. 评估
# =========================


x_test = np.linspace(X_MIN, X_MAX, 600).reshape(-1, 1).astype(np.float32)
y_pred = model.predict(x_test)

# 用 DeepXDE 的 operator 直接计算 residual。
residual_pred = model.predict(x_test, operator=parker_residual)

y_ref = np.array([parker_reference_y(x[0]) for x in x_test]).reshape(-1, 1)

relative_l2_error = np.linalg.norm(y_pred - y_ref) / np.linalg.norm(y_ref)
max_abs_error = np.max(np.abs(y_pred - y_ref))
residual_mse = np.mean(residual_pred**2)
residual_max_abs = np.max(np.abs(residual_pred))

# 临界点 x=1 附近方程系数 (y - 1/y) 退化；最内侧 x=0.2 附近右端也很陡。
# 因此同时报告一个 core residual，用于判断主体区域是否满足方程。
core_mask = (x_test[:, 0] > 0.35) & (np.abs(x_test[:, 0] - 1.0) > 0.08)
core_residual = residual_pred[core_mask]
core_residual_mse = np.mean(core_residual**2)
core_residual_max_abs = np.max(np.abs(core_residual))

y_at_critical = model.predict(np.array([[1.0]], dtype=np.float32))[0, 0]
y_at_far = model.predict(np.array([[X_MAX]], dtype=np.float32))[0, 0]

print("\nEvaluation:")
print(f"Reference relative L2 error: {relative_l2_error:.6e}")
print(f"Reference max absolute error: {max_abs_error:.6e}")
print(f"Residual MSE, full domain: {residual_mse:.6e}")
print(f"Residual max absolute, full domain: {residual_max_abs:.6e}")
print(f"Residual MSE, core domain: {core_residual_mse:.6e}")
print(f"Residual max absolute, core domain: {core_residual_max_abs:.6e}")
print(f"y(1): {y_at_critical:.6f}")
print(f"y({X_MAX}): {y_at_far:.6f}")
print(f"Minimum predicted y: {np.min(y_pred):.6f}")


# =========================
# 7. 画图
# =========================


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(x_test, y_ref, "b-", linewidth=2, label="Transonic reference")
ax.plot(x_test, y_pred, "r--", linewidth=2, label="DeepXDE PINN")
ax.axvline(1.0, color="k", linestyle=":", linewidth=1.5, label="Critical point")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
ax.set_xlabel("x = r / r_c")
ax.set_ylabel("y = v / c_s")
ax.set_title("Parker solar wind Mach number")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(x_test, residual_pred, "m-", linewidth=1.5)
ax.axvline(1.0, color="k", linestyle=":", linewidth=1.5)
ax.set_xlabel("x = r / r_c")
ax.set_ylabel("ODE residual")
ax.set_title("Parker equation residual")
ax.grid(True, alpha=0.3)

fig.tight_layout()

figure_path = output_dir / "parker_solar_wind_deepXDE_result.png"
fig.savefig(figure_path, dpi=200)
print(f"Figure saved to: {figure_path}")
