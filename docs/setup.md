# 环境配置指南

本文档介绍如何在不同环境中配置和运行教程代码。

## 支持的环境

| 环境 | Step 1-2 | Step 3-4 | Step 5-6 | 说明 |
|------|----------|----------|----------|------|
| MacBook M 系列 | ✅ | ✅ (小规模) | ⚠️ 较慢 | 本地开发 |
| CPU | ✅ | ✅ (小规模) | ⚠️ 较慢 | 学习理解 |
| NVIDIA GPU | ✅ | ✅ | ✅ | 推荐完整训练 |
| AutoDL 租用 | ✅ | ✅ | ✅ | 高性价比 |

---

## 1. 基础环境配置

### Python 版本

推荐 Python 3.10+

```bash
python --version  # 应该 >= 3.10
```

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/xxx/llm_tutorial.git
cd llm_tutorial

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

---

## 2. MacBook M 系列配置

### PyTorch MPS 后端

M1/M2/M3 芯片可以使用 MPS (Metal Performance Shaders) 加速：

```bash
# 安装支持 MPS 的 PyTorch
pip install torch torchvision

# 验证 MPS 可用
python -c "import torch; print(torch.backends.mps.is_available())"
# 应该输出 True
```

### 运行示例

```bash
# Step 1-2: 直接运行
cd step1_tokenizer && python tokenizer.py
cd step2_gpt_model && python model.py

# Step 3: 使用 MPS
cd step3_pretrain
python train.py --device mps --epochs 3 --batch_size 8

# Step 4-6: 同样使用 --device mps
```

### 注意事项

- M1/M2/M3 的 MPS 后端比 CUDA 慢，但比 CPU 快
- 建议减小 batch_size 以避免内存不足
- 某些操作可能不支持 MPS，会自动回退到 CPU

---

## 3. NVIDIA GPU 配置

### 安装 CUDA 版 PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 验证 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 运行示例

```bash
# 使用 CUDA
python train.py --device cuda --epochs 5 --batch_size 32
```

---

## 4. AutoDL 租用配置

[AutoDL](https://www.autodl.com/) 是国内的 GPU 租用平台，性价比高。

### 创建实例

1. 注册 AutoDL 账号
2. 选择镜像：`PyTorch 2.0 + Python 3.10 + CUDA 11.8`
3. 选择 GPU：推荐 `RTX 3090 (24GB)` 或 `RTX 4090 (24GB)`
4. 创建实例

### 连接实例

```bash
# SSH 连接（复制 AutoDL 提供的命令）
ssh -p xxxxx root@region-xxx.autodl.com

# 或使用 JupyterLab（AutoDL 控制台提供）
```

### 配置环境

```bash
# AutoDL 镜像通常已安装好 PyTorch，直接克隆仓库
cd /root/autodl-tmp  # 数据盘
git clone https://github.com/xxx/llm_tutorial.git
cd llm_tutorial

# 安装额外依赖
pip install sentencepiece tiktoken
```

### 运行训练

```bash
# Step 3: 预训练
cd step3_pretrain
python train.py --device cuda --epochs 5 --batch_size 32

# Step 4: SFT
cd step4_sft
python train_sft.py --device cuda --epochs 3

# Step 5: DPO
cd step5_rlhf
python train_dpo.py --device cuda --epochs 1

# Step 6: VLM
cd step6_vlm
python train.py --device cuda --epochs 3
```

### 成本估算

| GPU | 价格 | Step 3 | Step 4 | Step 5 | Step 6 | 总计 |
|-----|------|--------|--------|--------|--------|------|
| RTX 3090 | ~1.5元/小时 | 2小时 | 1小时 | 1小时 | 2小时 | ~9元 |
| RTX 4090 | ~2.5元/小时 | 1小时 | 0.5小时 | 0.5小时 | 1小时 | ~7.5元 |

---

## 5. 常见问题

### Q1: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 减小 `batch_size`
2. 减小 `block_size` 或 `max_length`
3. 使用梯度累积：`--accumulation_steps 4`

### Q2: MPS 不支持某些操作

```
NotImplementedError: The operator 'xxx' is not currently implemented for the MPS device
```

**解决方案**：
回退到 CPU：`--device cpu`

### Q3: 找不到模块

```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案**：
```bash
pip install xxx
```

### Q4: 训练太慢

**解决方案**：
1. 使用 GPU
2. 减小模型大小（n_embd, n_layer）
3. 使用混合精度训练（代码中已默认启用）

---

## 6. 推荐配置

### 快速体验（CPU/MPS）

```python
# 小模型配置
n_embd = 128
n_head = 4
n_layer = 4
batch_size = 8
block_size = 64
```

### 完整训练（GPU）

```python
# 标准配置
n_embd = 512
n_head = 8
n_layer = 8
batch_size = 32
block_size = 256
```

---

## 下一步

环境配置完成后，开始学习：

1. [Step 1: Tokenizer](../step1_tokenizer/)
2. [Step 2: GPT Model](../step2_gpt_model/)
3. ...
