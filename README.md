# PINN 学习项目

这个目录用于学习 Physics-Informed Neural Networks，当前包含一个 PyTorch 手写 ODE PINN 入门示例，以及 PINN/DeepXDE 速查文档。

## 当前环境

项目虚拟环境位于：

```bash
/home/guiyu/workspace/PINN/.venv
```

已验证的主要依赖：

```text
torch       2.12.0+cpu
DeepXDE     1.15.0
numpy       2.2.6
matplotlib  3.10.9
scipy       1.15.3
pandas      2.3.3
tqdm        4.68.1
```

PyTorch 是 CPU 版：

```text
torch.version.cuda = None
torch.cuda.is_available() = False
```

## 激活环境

```bash
cd /home/guiyu/workspace/PINN
source .venv/bin/activate
```

验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import deepxde as dde; print(dde.__version__)"
```

## 从依赖文件重新安装

如果已经有虚拟环境：

```bash
python -m pip install -r requirements.txt
```

如果从零开始：

```bash
cd /home/guiyu/workspace/PINN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

说明：

```text
普通 Python 包使用清华 PyPI 镜像。
torch 使用 PyTorch 官方 CPU wheel 源，避免下载 NVIDIA CUDA/cuDNN 依赖。
```

## 运行入门示例

```bash
cd /home/guiyu/workspace/PINN
source .venv/bin/activate
python pinn_ode_intro_torch.py
```

该脚本求解：

```text
du/dt = -u
u(0) = 1
```

解析解：

```text
u(t) = exp(-t)
```

运行后会输出训练损失和误差，并保存图像到：

```text
outputs/pinn_ode_intro_result.png
```

当前已跑通结果：

```text
Relative L2 error: 1.293063e-03
Max absolute error: 1.406372e-03
```

## 文件说明

```text
PINN_入门教程.ipynb          Jupyter Notebook 入门教程
pinn_ode_intro_torch.py      PyTorch 手写 PINN ODE 示例
PINN_DeepXDE_速查手册.md     机器学习、PINN、DeepXDE 速查文档
requirements.txt             pip 依赖文件
outputs/                     运行结果图像
```

## 后续学习顺序

建议按这个顺序扩展：

```text
1. 一阶 ODE：du/dt = -u
2. 二阶 ODE：u'' + u = 0
3. 一维热传导方程：u_t = alpha u_xx
4. Burgers 方程：u_t + u u_x = nu u_xx
5. Parker 太阳风方程
6. MHD 表面输运模型
```
