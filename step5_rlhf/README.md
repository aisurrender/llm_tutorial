# Step 5: RLHF - 人类偏好对齐

## 核心问题

SFT 后模型能对话，但回答质量参差不齐。如何让模型输出更符合人类偏好？

**答案**：DPO（Direct Preference Optimization）

## 学习方式

1. 打开 `tutorial.ipynb` 作为主学习界面
2. 去 `train_dpo_exercise.py` 完成 TODO
3. 卡住了？查看 `*_solution.py`

## 文件结构

| 文件 | 说明 |
|------|------|
| `tutorial.ipynb` | **主学习界面** |
| `train_dpo_exercise.py` | **DPO 练习** - 完成 TODO |
| `train_dpo_solution.py` | DPO 完整实现 |
| `data_solution.py` | 数据处理参考 |

## 练习任务

### TODO 1: 实现 DPO Loss

```python
# DPO 核心公式
L_DPO = -log σ(β * (π_logratios - ref_logratios))

# 其中
π_logratios = log π(chosen) - log π(rejected)
ref_logratios = log π_ref(chosen) - log π_ref(rejected)
```

## 核心概念

### DPO vs PPO

| 方面 | PPO | DPO |
|------|-----|-----|
| Reward Model | 需要 | 不需要 |
| 训练稳定性 | 较差 | 较好 |
| 实现复杂度 | 高 | 低 |

### DPO 直觉理解

1. 增加 chosen（好回答）的概率
2. 降低 rejected（差回答）的概率
3. 不要偏离参考模型太远

## 验证你的实现

```bash
python train_dpo_exercise.py
```

## 验证清单

- [ ] 解释 RLHF 的目标
- [ ] 解释 DPO 相比 PPO 的优势
- [ ] 实现 DPO Loss
- [ ] 理解 β 参数的作用

## 进入下一步

进入 [Step 6: VLM](../step6_vlm/)，学习如何扩展到多模态。
