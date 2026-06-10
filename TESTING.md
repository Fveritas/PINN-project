# PINN 代码运行与测试方法

本文档说明如何运行当前 PyTorch PINN 入门代码，并判断结果是否正常。

当前可运行脚本：

```text
pinn_ode_intro_torch.py
```

它求解的一阶 ODE 是：

```text
du/dt = -u
u(0) = 1
t in [0, 3]
```

解析解：

```text
u(t) = exp(-t)
```

---

## 1. 进入项目目录

```bash
cd /home/guiyu/workspace/PINN
```

确认目录中有这些文件：

```bash
ls
```

应该至少看到：

```text
README.md
requirements.txt
pinn_ode_intro_torch.py
```

---

## 2. 激活虚拟环境

```bash
source .venv/bin/activate
```

激活后，终端前面通常会出现：

```text
(.venv)
```

确认当前 Python 来自项目虚拟环境：

```bash
which python
```

期望输出：

```text
/home/guiyu/workspace/PINN/.venv/bin/python
```

---

## 3. 检查依赖

运行：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

当前 CPU 环境的期望输出类似：

```text
2.12.0+cpu
None
False
```

含义：

```text
2.12.0+cpu  表示安装的是 CPU 版 PyTorch
None        表示这个 torch wheel 不带 CUDA
False       表示当前不会使用 GPU
```

再检查 DeepXDE：

```bash
python -c "import deepxde as dde; print(dde.__version__)"
```

期望输出包含：

```text
1.15.0
```

---

## 4. 运行 PINN 入门脚本

```bash
python pinn_ode_intro_torch.py
```

运行时会看到类似输出：

```text
PyTorch version: 2.12.0+cpu
Device: cpu
PDE points shape: (200, 1)
IC point: t = 0.0, u = 1.0
Number of trainable parameters: 901
Epoch    1/3000 | ...
Epoch  500/3000 | ...
...
Epoch 3000/3000 | ...

Evaluation:
Relative L2 error: ...
Max absolute error: ...
Figure saved to: /home/guiyu/workspace/PINN/outputs/pinn_ode_intro_result.png
```

运行时间通常在十几秒到几十秒之间，取决于机器性能。

---

## 5. 判断是否跑通

满足以下条件即可认为代码跑通：

```text
1. 没有 Python 报错；
2. 训练过程输出 Epoch 信息；
3. 最后输出 Relative L2 error 和 Max absolute error；
4. 生成 outputs/pinn_ode_intro_result.png。
```

检查输出图片：

```bash
ls -lh outputs/pinn_ode_intro_result.png
```

期望看到文件存在，且大小不是 0。

当前已验证的一次结果：

```text
Relative L2 error: 1.293063e-03
Max absolute error: 1.406372e-03
```

由于随机种子已经固定，结果一般应接近这个数值。如果误差在 `1e-2` 以内，入门测试可以认为正常。

---

## 6. 一键测试命令

如果只想快速确认环境和脚本是否能跑：

```bash
cd /home/guiyu/workspace/PINN
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python pinn_ode_intro_torch.py
ls -lh outputs/pinn_ode_intro_result.png
```

---

## 7. 常见问题

### 7.1 ModuleNotFoundError: No module named 'torch'

原因通常是没有激活虚拟环境。

解决：

```bash
cd /home/guiyu/workspace/PINN
source .venv/bin/activate
which python
```

确认 `which python` 输出的是：

```text
/home/guiyu/workspace/PINN/.venv/bin/python
```

### 7.2 找不到 .venv

如果 `.venv` 不存在，重新创建并安装依赖：

```bash
cd /home/guiyu/workspace/PINN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 7.3 torch.cuda.is_available() 是 False

当前项目安装的是 CPU 版 PyTorch，这是预期结果。

入门 ODE 示例不需要 GPU。后续做大规模 PDE 或 MHD 时，再单独安装 GPU 版 PyTorch。

### 7.4 图片没有弹出来

脚本默认保存图片，不弹窗。

结果图路径：

```text
outputs/pinn_ode_intro_result.png
```

这是为了兼容 WSL、服务器和无图形界面的终端环境。

### 7.5 误差比示例大

先确认脚本没有被改动。如果只是轻微波动，通常没问题。

入门测试建议标准：

```text
Relative L2 error < 1e-2
```

如果误差明显大于 `1e-2`，重点检查：

```text
1. 是否改动了学习率；
2. 是否改动了训练轮数；
3. 是否改动了网络结构；
4. 是否删除了随机种子；
5. 是否改动了 loss 函数。
```

---

## 8. 测试目标

这个测试不是为了证明 PINN 对所有问题都有效，而是确认：

```text
1. Python 环境正常；
2. PyTorch 自动微分正常；
3. PINN residual loss 可以下降；
4. 网络能学到 exp(-t)；
5. 结果图可以正常保存。
```

后续扩展热传导方程时，也应保留类似测试：

```text
1. loss 曲线；
2. 解析解或数值解对比；
3. 相对 L2 误差；
4. residual 分布；
5. 输出图片或数据文件。
```

