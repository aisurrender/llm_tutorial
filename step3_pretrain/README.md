# Step 3: Pretrain - 预训练语言模型

## 核心问题

如何让模型学会语言？

**答案**：下一个词预测（Next Token Prediction）

```
输入: "今天天气"
目标: "天气真好"
Loss: -log P("天气真好" | "今天天气")
```

## 学习方式

本教程采用**动手实践**的方式：

1. 打开 `tutorial.ipynb` 作为主学习界面
2. 去 `data_exercise.py` 完成数据集 TODO
3. 去 `train_exercise.py` 完成训练 TODO
4. 卡住了？查看 `*_solution.py`

## 文件结构

| 文件 | 说明 |
|------|------|
| `tutorial.ipynb` | **主学习界面** - 概念讲解、可视化 |
| `data_exercise.py` | **数据集练习** - 完成 TODO |
| `train_exercise.py` | **训练练习** - 完成 TODO |
| `data_solution.py` | 数据集参考答案 |
| `train_solution.py` | 训练参考答案 |

## 学习目标

1. 理解预训练的目标：语言建模
2. **动手实现**数据集构建
3. **动手实现**学习率调度
4. **动手实现**训练循环

## 练习任务

### TODO 1: 实现 `__getitem__`（data_exercise.py）

构建语言建模的输入-目标对：

```python
def __getitem__(self, idx):
    x = data[idx : idx + block_size]      # 输入
    y = data[idx + 1 : idx + block_size + 1]  # 目标（右移一位）
    return x, y
```

### TODO 2: 实现学习率调度（train_exercise.py）

- 2a: Warmup 阶段的线性增加
- 2b: Cosine Decay 阶段的衰减

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

### TODO 3: 实现训练循环（train_exercise.py）

- 3a: 更新学习率
- 3b: 前向传播 `logits, loss = model(x, targets=y)`
- 3c: 反向传播 + 梯度裁剪 + 优化器更新

## 核心概念

### 语言建模目标

给定前文，预测下一个词的概率分布：

```
P(x_t | x_1, x_2, ..., x_{t-1})
```

### 训练循环

```python
for batch in dataloader:
    logits, loss = model(input_ids, targets=labels)  # 前向
    loss.backward()                                   # 反向
    clip_grad_norm_(model.parameters(), 1.0)          # 梯度裁剪
    optimizer.step()                                  # 更新
    optimizer.zero_grad()                             # 清零
```

### 学习率调度

- **Warmup**：开始时学习率从 0 逐渐增加，避免初期不稳定
- **Cosine Decay**：之后学习率按余弦曲线衰减

## 验证你的实现

```bash
# 测试数据集
python data_exercise.py

# 测试学习率调度
python train_exercise.py --test

# 运行完整训练
python train_exercise.py --device cpu --epochs 3
```

## 关键超参数

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `batch_size` | 批大小 | 16-64 |
| `learning_rate` | 学习率 | 1e-4 ~ 6e-4 |
| `warmup_steps` | 预热步数 | 总步数的 5-10% |
| `grad_clip` | 梯度裁剪 | 1.0 |

## 验证清单

完成本步骤后，你应该能够：

- [ ] 解释"下一个词预测"的训练目标
- [ ] 实现数据集的 `__getitem__`
- [ ] 解释为什么需要学习率调度
- [ ] 实现学习率调度函数
- [ ] 实现完整的训练循环

## 进入下一步

预训练完成后，进入 [Step 4: SFT](../step4_sft/)，学习如何让模型学会遵循指令。
