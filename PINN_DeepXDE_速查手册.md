# PINN 与 DeepXDE 速查手册

这份文档用于快速串起三个层次：

1. 机器学习和 PyTorch 的最小基础；
2. PINN 的基本原理；
3. DeepXDE 的安装、后端设置和最小用法。

当前学习建议：

```text
手写 PyTorch ODE PINN
  -> 手写 PyTorch 热传导 PDE PINN
  -> DeepXDE 复现同一问题
  -> GPU 加速
  -> 太阳风方程
  -> MHD 表面输运模型
```

---

## 1. 机器学习最小基础

机器学习的核心流程可以概括为：

```text
模型预测 -> 计算损失 -> 反向传播 -> 更新参数
```

对应 PyTorch 代码：

```python
optimizer.zero_grad()   # 清空上一轮梯度
loss = loss_fn(...)     # 计算损失
loss.backward()         # 反向传播，计算梯度
optimizer.step()        # 根据梯度更新参数
```

常用概念：

| 概念 | 含义 | 在 PINN 中的对应 |
|---|---|---|
| 模型 model | 一个可训练函数 | 神经网络近似未知解 u(x,t) |
| 参数 parameters | 模型中可训练的权重和偏置 | 神经网络权重 theta |
| 损失函数 loss | 衡量预测有多差 | 方程残差 + 初始/边界条件误差 |
| 优化器 optimizer | 更新参数的方法 | Adam、L-BFGS |
| 训练点 | 用于计算 loss 的输入点 | PDE 区域点、边界点、初始点 |

监督学习需要大量输入和标签：

```text
t -> u(t)
```

PINN 通常不需要大量标签数据。它主要用物理方程提供训练信号，例如：

```text
du/dt + u = 0
```

但 PINN 仍然需要少量约束，例如初始条件或边界条件：

```text
u(0) = 1
```

所以 PINN 更准确地说是：

```text
物理方程约束 + 少量初边值/观测数据约束的学习方法
```

---

## 2. PyTorch 基础速查

### 2.1 nn.Module

`nn.Module` 是 PyTorch 神经网络的基类。

```python
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, t):
        return self.net(t)
```

关键点：

```python
model = PINN()
model.parameters()
```

只有继承 `nn.Module` 后，PyTorch 才能自动管理模型参数。

### 2.2 nn.Sequential

`nn.Sequential` 按顺序执行多个层：

```python
self.net = nn.Sequential(
    nn.Linear(1, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 1),
)
```

数据流：

```text
输入 t
  -> Linear(1 -> 20)
  -> Tanh
  -> Linear(20 -> 20)
  -> Tanh
  -> Linear(20 -> 1)
  -> 输出 u(t)
```

整体是一个多层感知机 MLP，不是多个 MLP。

### 2.3 nn.Linear

`nn.Linear(a, b)` 表示线性变换：

```text
输入维度 a -> 输出维度 b
```

数学形式：

```text
y = x W^T + b
```

例如：

```python
nn.Linear(1, 20)
```

表示输入一个数 `t`，输出 20 个隐藏特征。

### 2.4 激活函数

PINN 中常用 `Tanh`：

```python
nn.Tanh()
```

原因：

```text
PINN 经常需要对网络输出求一阶、二阶甚至更高阶导数。
Tanh 是光滑函数，适合自动微分。
```

### 2.5 Xavier 初始化

常见初始化代码：

```python
for layer in self.net:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
```

含义：

```text
遍历网络每一层；
如果这一层是 Linear，就初始化它的 weight 和 bias；
如果是 Tanh，就跳过。
```

`isinstance(layer, nn.Linear)` 用来判断当前层是不是线性层。

Xavier 初始化，也叫 Glorot 初始化，目标是让每一层输入和输出的数值尺度相对稳定，减少梯度爆炸或梯度消失。

### 2.6 自动微分

PINN 最关键的 PyTorch 代码：

```python
t.requires_grad_(True)
u = model(t)

du_dt = torch.autograd.grad(
    outputs=u,
    inputs=t,
    grad_outputs=torch.ones_like(u),
    create_graph=True,
)[0]
```

解释：

| 参数 | 含义 |
|---|---|
| `outputs=u` | 要被求导的量 |
| `inputs=t` | 对哪个变量求导 |
| `grad_outputs=torch.ones_like(u)` | 因为 `u` 不是标量，需要指定权重 |
| `create_graph=True` | 保留计算图，后面还能继续反向传播 |

二阶导数写法：

```python
du_dt = torch.autograd.grad(
    u, t,
    grad_outputs=torch.ones_like(u),
    create_graph=True,
)[0]

d2u_dt2 = torch.autograd.grad(
    du_dt, t,
    grad_outputs=torch.ones_like(du_dt),
    create_graph=True,
)[0]
```

---

## 3. PINN 基本原理

以一阶 ODE 为例：

```text
du/dt = -u,  t in [0, 3]
u(0) = 1
```

解析解是：

```text
u(t) = exp(-t)
```

PINN 的做法不是直接拿大量 `t -> u(t)` 数据训练，而是让神经网络表示未知函数：

```text
u_theta(t) = neural_network(t)
```

然后把微分方程写成残差：

```text
du/dt + u = 0
```

在训练点上计算：

```text
residual(t) = d u_theta / dt + u_theta(t)
```

如果网络满足方程，`residual` 应该接近 0。

### 3.1 PINN 总损失

本例损失函数：

```text
Loss = Loss_ODE + lambda_ic * Loss_IC
```

其中：

```text
Loss_ODE = mean((du/dt + u)^2)
Loss_IC  = mean((u(0) - 1)^2)
```

`lambda_ic` 是初始条件损失权重。入门例子中可以取：

```python
lambda_ic = 10.0
```

如果初始条件满足得不好，可以增大；如果初始条件主导训练、方程残差降不下去，可以减小。

### 3.2 手写 PyTorch PINN 最小模板

```python
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, t):
        return self.net(t)


def loss_fn(model, t_pde, t_ic, u_ic):
    t_pde = t_pde.clone().detach().requires_grad_(True)
    u = model(t_pde)

    du_dt = torch.autograd.grad(
        outputs=u,
        inputs=t_pde,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    loss_ode = torch.mean((du_dt + u) ** 2)
    loss_ic = torch.mean((model(t_ic) - u_ic) ** 2)
    return loss_ode + 10.0 * loss_ic


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

t_pde = torch.rand(200, 1) * 3.0
t_ic = torch.tensor([[0.0]])
u_ic = torch.tensor([[1.0]])

for epoch in range(3000):
    optimizer.zero_grad()
    loss = loss_fn(model, t_pde, t_ic, u_ic)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(epoch + 1, loss.item())
```

对应完整脚本：

```text
/home/guiyu/workspace/PINN/pinn_ode_intro_torch.py
```

---

## 4. 从 ODE 到 PDE

ODE 只有一个自变量，例如：

```text
u = u(t)
```

PDE 有多个自变量，例如热传导方程：

```text
u = u(x, t)
```

一维热传导方程：

```text
u_t = alpha * u_xx
```

残差写成：

```text
residual = u_t - alpha * u_xx
```

PyTorch 里需要分别对 `x` 和 `t` 求导。

典型输入：

```python
xt = torch.cat([x, t], dim=1)
u = model(xt)
```

一阶导：

```python
grads = torch.autograd.grad(
    u, xt,
    grad_outputs=torch.ones_like(u),
    create_graph=True,
)[0]

u_x = grads[:, 0:1]
u_t = grads[:, 1:2]
```

二阶空间导：

```python
u_x_grads = torch.autograd.grad(
    u_x, xt,
    grad_outputs=torch.ones_like(u_x),
    create_graph=True,
)[0]

u_xx = u_x_grads[:, 0:1]
```

热传导 residual：

```python
residual = u_t - alpha * u_xx
loss_pde = torch.mean(residual ** 2)
```

---

## 5. 安装 PyTorch

建议使用：

```bash
python3 -m pip ...
```

而不是直接：

```bash
pip ...
```

这样可以减少 `pip` 和 `python3` 指向不同环境的问题。

### 5.1 只跑入门示例：CPU 版

ODE 和简单热方程用 CPU 就够：

```bash
python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
```

普通依赖可以走国内镜像：

```bash
python3 -m pip install numpy matplotlib scipy pandas tqdm \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.2 GPU 版

先看显卡和驱动：

```bash
nvidia-smi
```

再按 PyTorch 官网选择对应 CUDA wheel。

常见 CUDA 12.1 示例：

```bash
python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

常见 CUDA 12.4 示例：

```bash
python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

安装后验证：

```bash
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

输出含义：

```text
torch.__version__         PyTorch 版本
torch.version.cuda        当前 torch wheel 对应的 CUDA 版本
torch.cuda.is_available() 是否能使用 GPU
```

### 5.3 国内 PyPI 镜像

常用镜像：

```bash
# 清华
-i https://pypi.tuna.tsinghua.edu.cn/simple

# 阿里云
-i https://mirrors.aliyun.com/pypi/simple

# 中科大
-i https://pypi.mirrors.ustc.edu.cn/simple
```

永久设置清华源：

```bash
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：

```text
PyTorch 建议优先使用官方 PyTorch wheel 源安装。
numpy、matplotlib、deepxde 等普通包可以使用国内 PyPI 镜像。
```

---

## 6. 安装 DeepXDE

DeepXDE 是一个 PINN 框架。它封装了：

| 内容 | DeepXDE 中的角色 |
|---|---|
| 几何区域 | `Interval`, `Rectangle`, `GeometryXTime` |
| 时间区域 | `TimeDomain` |
| PDE/ODE 残差 | 用户定义的函数 |
| 初始条件 | `IC` |
| 边界条件 | `DirichletBC`, `NeumannBC`, `PeriodicBC` |
| 神经网络 | `FNN` |
| 数据采样 | `PDE`, `TimePDE` |
| 训练流程 | `Model.compile`, `Model.train` |

安装：

```bash
python3 -m pip install deepxde \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

建议先安装 PyTorch，再安装 DeepXDE：

```bash
python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

python3 -m pip install deepxde matplotlib numpy \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```bash
python3 -c "import deepxde as dde; print(dde.__version__)"
```

---

## 7. DeepXDE 后端设置

DeepXDE 支持多个后端，例如 TensorFlow、PyTorch、JAX、Paddle。

这里建议使用 PyTorch 后端，和手写代码保持一致。

### 7.1 命令行设置

Linux/macOS/WSL：

```bash
export DDE_BACKEND=pytorch
python3 your_script.py
```

Windows PowerShell：

```powershell
$env:DDE_BACKEND="pytorch"
python your_script.py
```

### 7.2 Python 脚本内设置

必须写在 `import deepxde` 之前：

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
```

如果顺序反了：

```python
import deepxde as dde
os.environ["DDE_BACKEND"] = "pytorch"
```

可能不会生效。

---

## 8. DeepXDE 解同一个 ODE

问题：

```text
du/dt = -u
u(0) = 1
t in [0, 3]
```

DeepXDE 最小示例：

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np


def ode(t, u):
    du_dt = dde.grad.jacobian(u, t)
    return du_dt + u


def initial_boundary(t, on_initial):
    return on_initial


geom = dde.geometry.TimeDomain(0.0, 3.0)

ic = dde.icbc.IC(
    geom,
    lambda t: 1.0,
    initial_boundary,
)

data = dde.data.PDE(
    geometry=geom,
    pde=ode,
    bcs=[ic],
    num_domain=200,
    num_boundary=1,
    solution=lambda t: np.exp(-t),
    num_test=100,
)

net = dde.nn.FNN(
    layer_sizes=[1, 20, 20, 20, 1],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

model = dde.Model(data, net)
model.compile("adam", lr=0.01, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=3000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
```

和手写 PyTorch 的对应关系：

| 手写 PyTorch | DeepXDE |
|---|---|
| `PINN(nn.Module)` | `dde.nn.FNN` |
| 自己采样 `t_pde` | `num_domain` |
| 自己写初始条件 loss | `dde.icbc.IC` |
| 自己写 residual | `pde` 函数 |
| `torch.autograd.grad` | `dde.grad.jacobian` |
| 自己写训练循环 | `model.train` |

---

## 9. DeepXDE 解热传导方程的结构

一维热传导：

```text
u_t = alpha * u_xx
x in [0, 1], t in [0, 1]
u(0,t)=0, u(1,t)=0
u(x,0)=sin(pi x)
```

DeepXDE 中通常拆成：

```python
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
```

PDE residual：

```python
def pde(x, u):
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    return u_t - alpha * u_xx
```

边界条件：

```python
bc = dde.icbc.DirichletBC(
    geomtime,
    lambda x: 0,
    lambda x, on_boundary: on_boundary,
)
```

初始条件：

```python
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(np.pi * x[:, 0:1]),
    lambda x, on_initial: on_initial,
)
```

训练数据：

```python
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
)
```

网络：

```python
net = dde.nn.FNN([2, 20, 20, 20, 1], "tanh", "Glorot normal")
model = dde.Model(data, net)
```

---

## 10. GPU 使用速查

手写 PyTorch：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
t_pde = t_pde.to(device)
t_ic = t_ic.to(device)
u_ic = u_ic.to(device)
```

检查显存：

```python
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# train ...

if torch.cuda.is_available():
    print(torch.cuda.max_memory_allocated() / 1024**2, "MB")
```

常见问题：

| 问题 | 原因 |
|---|---|
| GPU 没变快 | 点数太少，网络太小 |
| 报 device mismatch | 模型和张量不在同一设备 |
| `torch.cuda.is_available()` 是 False | 装了 CPU 版 torch，或驱动/CUDA 不匹配 |
| 显存涨很快 | 高阶导数保留计算图，点数过多 |

PINN 的 GPU 加速通常在以下情况更明显：

```text
PDE 点很多；
网络较大；
需要大量高阶导数；
所有张量都留在 GPU 上。
```

---

## 11. 学习顺序建议

### 第一阶段：ODE

目标：

```text
理解 neural network 表示函数；
理解 autograd 求导；
理解 residual loss。
```

算例：

```text
du/dt = -u
u(0) = 1
```

### 第二阶段：热传导 PDE

目标：

```text
理解多输入网络 u(x,t)；
理解 u_t 和 u_xx；
理解初始条件和边界条件同时进入 loss。
```

算例：

```text
u_t = alpha * u_xx
```

### 第三阶段：Burgers 方程

目标：

```text
理解非线性 PDE；
理解 u * u_x 这类非线性项；
观察 PINN 在低粘性问题中的训练困难。
```

方程：

```text
u_t + u u_x = nu u_xx
```

### 第四阶段：Parker 太阳风方程

目标：

```text
从数学玩具问题进入真实物理模型；
学习非量纲化；
处理临界点和物理约束。
```

### 第五阶段：MHD 表面输运模型

目标：

```text
学习球坐标 PDE；
处理周期边界；
处理极区边界；
加入观测数据或源项。
```

---

## 12. 常见错误

### 12.1 忘记 requires_grad

错误：

```python
t = torch.rand(200, 1)
u = model(t)
du_dt = torch.autograd.grad(u, t)
```

正确：

```python
t = torch.rand(200, 1).requires_grad_(True)
```

或者：

```python
t = t.clone().detach().requires_grad_(True)
```

### 12.2 忘记 create_graph=True

PINN loss 后面还要反向传播，所以通常要写：

```python
create_graph=True
```

### 12.3 模型和数据不在同一设备

错误：

```text
model 在 cuda
t_pde 在 cpu
```

正确：

```python
model = model.to(device)
t_pde = t_pde.to(device)
```

### 12.4 DeepXDE 后端设置太晚

错误：

```python
import deepxde as dde
import os
os.environ["DDE_BACKEND"] = "pytorch"
```

正确：

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
```

### 12.5 把 PINN 当成万能 PDE 求解器

PINN 不是传统数值方法的直接替代品。

对于标准正问题，有限差分、有限元、有限体积通常更快、更稳定。

PINN 更适合：

```text
数据稀疏；
参数未知；
反问题；
观测数据和物理方程融合；
复杂区域上的连续函数表示。
```

---

## 13. 当前目录建议

建议保持：

```text
PINN/
  PINN_入门教程.ipynb
  pinn_ode_intro_torch.py
  PINN_DeepXDE_速查手册.md
  outputs/
```

后续可以继续增加：

```text
heat_equation_torch.py
heat_equation_deepxde.py
burgers_torch.py
parker_solar_wind_pinn.py
```

每个脚本都建议输出：

```text
loss 曲线；
预测解图；
解析解或数值解对比；
相对 L2 误差；
训练时间；
GPU 显存占用。
```

