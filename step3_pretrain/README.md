# Step 3: Pretrain - 预训练语言模型

## 核心问题

如何让模型学会语言？

**答案**：下一个词预测（Next Token Prediction）

```
输入: "今天天气"
目标: "天气真好"
Loss: -log P("天气真好" | "今天天气")
```

## 学习目标

1. 理解预训练的目标：语言建模
2. 实现训练循环
3. 理解学习率调度（Warmup + Cosine Decay）
4. 理解梯度裁剪、混合精度训练

## 核心概念

### 1. 语言建模目标

给定前文，预测下一个词的概率分布：

```
P(x_t | x_1, x_2, ..., x_{t-1})
```

损失函数是交叉熵：

```python
loss = -sum(log P(x_t | x_1, ..., x_{t-1}))
```

### 2. 训练循环

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播
        logits, loss = model(input_ids, targets=labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 学习率调度

**Warmup**：开始时学习率从 0 逐渐增加，避免初期不稳定

**Cosine Decay**：之后学习率按余弦曲线衰减

```
lr
↑
|   /\
|  /  \
| /    \___________
|/
+------------------→ step
  warmup   decay
```

### 4. 混合精度训练

使用 FP16/BF16 加速训练，减少显存占用：

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    logits, loss = model(input_ids, targets=labels)
```

## 动手任务

### 任务 1：在 Shakespeare 数据集上预训练

```bash
# 准备数据
python data.py --create_sample

# 开始训练（CPU/MPS 可跑，约 5-10 分钟）
python train.py --device cpu --epochs 5 --batch_size 8
```

### 任务 2：观察训练过程

- 观察 Loss 下降曲线
- 观察学习率变化
- 尝试不同的超参数

### 任务 3：测试生成效果

```bash
python train.py --mode generate --prompt "To be or not to be"
```

## 代码文件

- `data.py` - 数据加载和预处理
- `train.py` - 训练脚本
- `config.py` - 训练配置

## 关键超参数

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `batch_size` | 批大小 | 16-64 |
| `learning_rate` | 学习率 | 1e-4 ~ 6e-4 |
| `warmup_steps` | 预热步数 | 总步数的 5-10% |
| `grad_clip` | 梯度裁剪 | 1.0 |

## 验证标准

完成本步骤后，你应该能够：

- [ ] 解释"下一个词预测"的训练目标
- [ ] 解释为什么需要学习率调度
- [ ] 训练一个能生成连贯文本的小模型
- [ ] 观察并解释训练过程中的 Loss 变化

## 进入下一步

预训练完成后，进入 [Step 4: SFT](../step4_sft/)，学习如何让模型学会遵循指令。
