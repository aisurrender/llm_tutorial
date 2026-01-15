# Step 2: GPT Model - Transformer 架构

## 核心问题

如何让神经网络理解 token 序列之间的关系？

**答案**：Transformer 架构中的 Self-Attention 机制

## 学习目标

1. 理解 Embedding（词嵌入 + 位置嵌入）
2. 理解并实现 Multi-Head Self-Attention
3. 理解并实现 Feed-Forward Network (MLP)
4. 组装完整的 GPT 模型

## GPT 模型架构

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
│  │  LayerNorm                      │  │
│  │  Multi-Head Self-Attention      │  │
│  │  Residual Connection            │  │
│  ├─────────────────────────────────┤  │
│  │  LayerNorm                      │  │
│  │  Feed-Forward Network (MLP)     │  │
│  │  Residual Connection            │  │
│  └─────────────────────────────────┘  │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│           Final LayerNorm             │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│        Linear (LM Head)              │
└─────────────────────────────────────┘
        ↓
输出 Logits: [vocab_size] 的概率分布
```

## 核心概念

### 1. Embedding 层

```python
# Token Embedding: 将 token ID 映射为向量
tok_emb = nn.Embedding(vocab_size, n_embd)  # [vocab_size, n_embd]

# Position Embedding: 让模型知道 token 的位置
pos_emb = nn.Embedding(block_size, n_embd)  # [max_seq_len, n_embd]

# 最终输入 = Token Embedding + Position Embedding
x = tok_emb(tokens) + pos_emb(positions)
```

### 2. Self-Attention

Self-Attention 的核心：让每个 token "看到"序列中的其他 token

```
Q = x @ W_q   # Query: "我在找什么？"
K = x @ W_k   # Key: "我有什么？"
V = x @ W_v   # Value: "我的内容是什么？"

Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**因果掩码（Causal Mask）**：GPT 是自回归模型，每个 token 只能看到它之前的 token

### 3. Multi-Head Attention

将 Attention 分成多个"头"，每个头关注不同的特征：

```python
# 单头 attention
head_dim = n_embd // n_head

# 多头：每个头独立计算 attention，最后拼接
heads = [attention(Q_i, K_i, V_i) for i in range(n_head)]
output = concat(heads) @ W_o
```

### 4. Feed-Forward Network (MLP)

两层全连接，中间用 GELU 激活：

```python
# 先扩展，再压缩
x = Linear(n_embd, 4 * n_embd)(x)
x = GELU(x)
x = Linear(4 * n_embd, n_embd)(x)
```

### 5. LayerNorm 和 Residual Connection

```python
# Pre-LayerNorm + Residual
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

## 动手任务

### 任务 1：理解各个组件

```bash
python model.py --mode components
```

分别运行并观察每个组件的输入输出 shape。

### 任务 2：组装完整模型

```bash
python model.py --mode full
```

查看完整 GPT 模型的前向传播。

### 任务 3：计算参数量

```bash
python model.py --mode params
```

理解参数量是如何计算的。

## 代码文件

- `model.py` - GPT 模型的完整实现（约 200 行）

## 关键超参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `vocab_size` | 词表大小 | 32000-50000 |
| `n_embd` | 嵌入维度 | 512-4096 |
| `n_head` | 注意力头数 | 8-32 |
| `n_layer` | Transformer 层数 | 6-32 |
| `block_size` | 最大序列长度 | 512-4096 |

**参数量估算**：约 12 × n_layer × n_embd² 个参数

## 验证标准

完成本步骤后，你应该能够：

- [ ] 画出 GPT 的架构图
- [ ] 解释 Self-Attention 的计算过程
- [ ] 解释为什么需要因果掩码（Causal Mask）
- [ ] 计算给定配置的模型参数量
- [ ] 用自己的代码构建一个 GPT 模型

## 进入下一步

理解模型架构后，进入 [Step 3: Pretrain](../step3_pretrain/)，学习如何训练这个模型。
