# Step 4: SFT - 指令微调

## 核心问题

预训练模型会"续写"，但不会"对话"。如何让它学会遵循指令？

**答案**：监督微调（Supervised Fine-Tuning, SFT）

```
预训练: "今天天气" → "真好，适合出游..."（续写）
SFT 后: "今天天气怎么样？" → "今天天气晴朗。"（回答）
```

## 学习方式

1. 打开 `tutorial.ipynb` 作为主学习界面
2. 去 `data_exercise.py` 完成 TODO
3. 卡住了？查看 `*_solution.py`

## 文件结构

| 文件 | 说明 |
|------|------|
| `tutorial.ipynb` | **主学习界面** |
| `data_exercise.py` | **数据处理练习** - 完成 TODO |
| `data_solution.py` | 数据处理答案 |
| `train_sft_solution.py` | SFT 训练参考 |
| `train_lora_solution.py` | LoRA 训练参考 |

## 练习任务

### TODO 1: 实现 _format_conversation

关键：只计算 Assistant 部分的 Loss

```python
# 输入: [system_tokens, user_tokens, assistant_tokens]
# 标签: [-100, -100, ..., -100, assistant_tokens]
#       ↑ 不计算 loss    ↑ 计算 loss
```

为什么？
- 我们希望模型学会"如何回答"，而不是"如何提问"
- User 的输入是上下文，不应该学习
- 只有 Assistant 的回答需要优化

## 核心概念

### LoRA（Low-Rank Adaptation）

不训练整个模型，只训练低秩"旁路"：

```
原始: y = Wx
LoRA: y = Wx + BAx  # 只训练 B 和 A

参数量对比（n=m=4096, r=8）:
- 原始: 16M 参数
- LoRA: 65K 参数（减少 99.6%）
```

## 验证你的实现

```bash
python data_exercise.py
```

## 验证清单

- [ ] 解释 SFT 和预训练的区别
- [ ] 解释为什么只计算 Assistant 部分的 Loss
- [ ] 实现 labels 的构建逻辑
- [ ] 解释 LoRA 的原理和优势

## 进入下一步

进入 [Step 5: RLHF](../step5_rlhf/)，学习如何让模型输出更符合人类偏好。
