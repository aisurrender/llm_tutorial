# Step 2: GPT Model - Transformer 架构

## 核心问题

如何让神经网络理解 token 序列之间的关系？

**答案**：Transformer 架构中的 Self-Attention 机制

## 学习方式

本教程采用**动手实践**的方式：

1. 打开 `tutorial.ipynb` 作为主学习界面
2. 去 `model_exercise.py` 完成 TODO 部分
3. 回到 Notebook 运行验证
4. 卡住了？查看 `model_solution.py`

## 文件结构

| 文件 | 说明 |
|------|------|
| `tutorial.ipynb` | **主学习界面** - 概念讲解、可视化、验证 |
| `model_exercise.py` | **练习文件** - 完成 TODO 部分 |
| `model_solution.py` | **参考答案** - 卡住时查看 |

## 学习目标

1. 理解 Embedding（词嵌入 + 位置嵌入）
2. **动手实现** Multi-Head Self-Attention
3. **动手实现** Feed-Forward Network (MLP)
4. **动手组装**完整的 GPT 模型

## GPT 架构图

```
输入 Token IDs: [1, 45, 23, 89, 12]
        ↓
┌─────────────────────────────────────┐
│  Token Embedding + Position Embedding │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│         Transformer Block × N         │
│  ┌─────────────────────────────────┐  │
│  │  LayerNorm → Attention → +残差   │  │
│  │  LayerNorm → MLP → +残差         │  │
│  └─────────────────────────────────┘  │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│    Final LayerNorm → LM Head          │
└─────────────────────────────────────┘
        ↓
输出 Logits: [vocab_size] 的概率分布
```

## 练习任务

### 基础练习

#### TODO 1: 实现 MLP forward（简单）
```python
def forward(self, x):
    # c_fc -> gelu -> c_proj -> dropout
```

#### TODO 2: 实现 TransformerBlock forward（简单）
```python
def forward(self, x):
    # x = x + attention(layernorm(x))
    # x = x + mlp(layernorm(x))
```

#### TODO 3: 实现 CausalSelfAttention（核心！）
- 3a: 计算注意力分数 `att = Q @ K^T / sqrt(d_k)`
- 3b: 应用因果掩码
- 3c: Softmax + Dropout + 与 V 相乘

#### TODO 4: 实现 GPT forward
- 4a: Token Embedding + Position Embedding
- 4b: 通过所有 Transformer Blocks
- 4c: Final LayerNorm + LM Head

### 进阶练习（选做）

#### TODO 5: 实现 RoPE 位置编码
挑战：用旋转位置编码（Rotary Position Embedding）替代绝对位置编码：
- RoPE 通过旋转向量来编码位置信息
- 这是 LLaMA、Qwen 等现代模型使用的方式
- 优势：更好的长度外推能力

```python
# RoPE 核心思想：将位置信息编码为旋转角度
# q_rotated = q * cos(θ) + rotate_half(q) * sin(θ)
```

#### TODO 6: 实现 KV Cache
挑战：实现推理时的 KV Cache 加速：
- 缓存已计算的 Key 和 Value
- 新 token 只需计算自己的 Q，复用之前的 KV
- 这是所有生产级 LLM 推理的标配

## 核心概念

### Self-Attention

```python
Q = x @ W_q   # Query: "我在找什么？"
K = x @ W_k   # Key: "我有什么？"
V = x @ W_v   # Value: "我的内容是什么？"

Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### 因果掩码（Causal Mask）

GPT 是自回归模型，每个 token 只能看到它之前的 token：

```
     pos0  pos1  pos2  pos3
pos0  ✓     ✗     ✗     ✗
pos1  ✓     ✓     ✗     ✗
pos2  ✓     ✓     ✓     ✗
pos3  ✓     ✓     ✓     ✓
```

## 验证你的实现

```bash
python model_exercise.py
```

## 关键超参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `vocab_size` | 词表大小 | 32000-50000 |
| `n_embd` | 嵌入维度 | 512-4096 |
| `n_head` | 注意力头数 | 8-32 |
| `n_layer` | Transformer 层数 | 6-32 |
| `block_size` | 最大序列长度 | 512-4096 |

**参数量估算**：约 12 × n_layer × n_embd² 个参数

## 验证清单

完成本步骤后，你应该能够：

**基础**
- [ ] 画出 GPT 的架构图
- [ ] 解释 Self-Attention 的计算过程
- [ ] 解释为什么需要因果掩码（Causal Mask）
- [ ] 实现 MLP、TransformerBlock、GPT
- [ ] 计算给定配置的模型参数量

**进阶**
- [ ] 解释 RoPE 相比绝对位置编码的优势
- [ ] 解释 KV Cache 如何加速推理
- [ ] 了解 GQA（Grouped Query Attention）的原理

## 推荐阅读

### GitHub 仓库

| 仓库 | 说明 |
|------|------|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 最简洁的 GPT 实现，附带训练脚本 |
| [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) | 论文逐行注释实现 |
| [karpathy/minGPT](https://github.com/karpathy/minGPT) | 教育性 GPT 实现 |

### 可视化教程

| 资源 | 说明 |
|------|------|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 图解 Transformer，强烈推荐 |
| [3Blue1Brown: Attention](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 可视化理解注意力机制 |
| [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 2小时从零实现 GPT |

### 现代架构演进

| 技术 | 说明 | 使用模型 |
|------|------|----------|
| **RoPE** | 旋转位置编码，更好的外推能力 | LLaMA, Qwen, Mistral |
| **GQA** | 分组查询注意力，减少 KV Cache | LLaMA 2, Mistral |
| **SwiGLU** | 改进的激活函数 | LLaMA, PaLM |
| **RMSNorm** | 简化的归一化，更快 | LLaMA, Qwen |

## 进入下一步

理解模型架构后，进入 [Step 3: Pretrain](../step3_pretrain/)，学习如何预训练这个模型。
