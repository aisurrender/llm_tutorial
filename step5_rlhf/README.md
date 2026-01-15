# Step 5: RLHF - 人类偏好对齐

## 核心问题

SFT 后模型能对话了，但回答质量参差不齐。如何让模型输出更符合人类偏好？

**答案**：RLHF（Reinforcement Learning from Human Feedback）

## 学习目标

1. 理解人类偏好对齐的目标
2. 理解 DPO（Direct Preference Optimization）原理
3. 实现 DPO 训练

## 核心概念

### 1. 什么是人类偏好？

给定同一个问题，人类会偏好某些回答：

```
问题: "什么是AI？"

回答 A (chosen): "AI是人工智能的缩写，是计算机科学的一个分支..."
回答 B (rejected): "AI就是机器人。"

人类偏好: A > B
```

### 2. 传统 RLHF 流程

```
1. 收集偏好数据 → 2. 训练 Reward Model → 3. PPO 训练 → 对齐的模型
```

**问题**：需要训练额外的 Reward Model，PPO 训练不稳定。

### 3. DPO：更简单的方案

DPO（Direct Preference Optimization）直接从偏好数据学习，无需 Reward Model。

**核心公式**：

```
L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

其中:
- y_w: chosen（被偏好的回答）
- y_l: rejected（不被偏好的回答）
- π: 当前策略（训练中的模型）
- π_ref: 参考策略（冻结的原始模型）
- β: 温度系数
```

**直觉理解**：
- 增加 chosen 回答的概率
- 降低 rejected 回答的概率
- 同时不要偏离原始模型太远（用 π_ref 约束）

### 4. DPO vs PPO

| 方面 | PPO | DPO |
|------|-----|-----|
| Reward Model | 需要 | 不需要 |
| 训练稳定性 | 较差 | 较好 |
| 超参数 | 多 | 少 |
| 实现复杂度 | 高 | 低 |

## 动手任务

### 任务 1：准备偏好数据

```bash
python data.py --create_sample
```

数据格式：
```json
{
  "prompt": "问题",
  "chosen": "好的回答",
  "rejected": "差的回答"
}
```

### 任务 2：DPO 训练

```bash
python train_dpo.py --device cuda --epochs 1 --beta 0.1
```

### 任务 3：对比效果

比较 SFT 模型和 DPO 后模型的回答质量。

## 代码文件

- `data.py` - DPO 数据处理
- `train_dpo.py` - DPO 训练脚本

## DPO 关键参数

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `beta` | 温度系数 | 0.1 - 0.5 |
| `learning_rate` | 学习率 | 1e-6 ~ 5e-7 |

**注意**：DPO 的学习率要很小，避免模型"遗忘"。

## 验证标准

完成本步骤后，你应该能够：

- [ ] 解释 RLHF 的目标
- [ ] 解释 DPO 相比 PPO 的优势
- [ ] 实现 DPO 训练
- [ ] 对比 SFT 和 DPO 后的模型效果

## 进入下一步

完成 LLM 的全流程后，进入 [Step 6: VLM](../step6_vlm/)，学习如何扩展到多模态。
