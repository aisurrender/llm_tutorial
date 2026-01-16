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
| `data_solution.py` | **偏好数据处理** - 构建 chosen/rejected 对 |

## 练习任务

### 基础练习

#### TODO 1: 实现 DPO Loss

```python
# DPO 核心公式
L_DPO = -log σ(β * (π_logratios - ref_logratios))

# 其中
π_logratios = log π(chosen) - log π(rejected)
ref_logratios = log π_ref(chosen) - log π_ref(rejected)
```

### 进阶练习（选做）

#### TODO 2: 理解偏好数据构建

查看 `data_solution.py`，理解如何构建偏好数据：

```bash
# 查看偏好数据处理代码
cat data_solution.py
```

思考：
- chosen 和 rejected 如何选择？
- 数据质量如何影响 DPO 效果？

#### TODO 3: 对比 PPO 与 DPO

挑战：理解两种方法的数学联系：

```
PPO: 需要训练 Reward Model，然后用 RL 优化
     max E[R(x,y)] - β * KL(π || π_ref)

DPO: 将 Reward Model 隐式化，直接优化偏好
     相当于 R(x,y) = β * log(π(y|x) / π_ref(y|x))
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

完成本步骤后，你应该能够：

**基础**
- [ ] 解释 RLHF 的目标（对齐人类偏好）
- [ ] 解释 DPO 相比 PPO 的优势
- [ ] 实现 DPO Loss 计算
- [ ] 理解 β 参数的作用（控制偏离程度）

**进阶**
- [ ] 解释 PPO 和 DPO 的数学联系
- [ ] 了解如何构建高质量偏好数据
- [ ] 了解 Reward Model 的作用

## 推荐阅读

### GitHub 仓库

| 仓库 | 说明 |
|------|------|
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 完整 RLHF 框架（PPO + DPO） |
| [huggingface/trl](https://github.com/huggingface/trl) | Hugging Face 的 RLHF 库 |
| [RLHFlow/Online-RLHF](https://github.com/RLHFlow/Online-RLHF) | 在线迭代 RLHF |

### 论文

| 论文 | 说明 |
|------|------|
| [DPO](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization 原始论文 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | OpenAI RLHF 开山之作 |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | Anthropic 的 RLAIF 方法 |

### 教程

| 资源 | 说明 |
|------|------|
| [RLHF in 2024 with DPO](https://www.philschmid.de/dpo-align-llms-in-2024-with-trl) | Hugging Face 实战教程 |
| [ICLR 2024: RLHF without RL](https://iclr-blogposts.github.io/2024/blog/rlhf-without-rl/) | DPO 原理深度解读 |

### RLHF 方法演进

| 方法 | 年份 | 特点 |
|------|------|------|
| **PPO** | 2022 | 需要 Reward Model + RL |
| **DPO** | 2023 | 无需 RM，直接优化偏好 |
| **KTO** | 2024 | 只需要 good/bad 标签 |
| **ORPO** | 2024 | SFT + 偏好对齐一步完成 |

## 进入下一步

进入 [Step 6: VLM](../step6_vlm/)，学习如何扩展到多模态。
