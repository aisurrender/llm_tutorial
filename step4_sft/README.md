# Step 4: SFT - 指令微调

## 核心问题

预训练模型会"续写"，但不会"对话"。如何让它学会遵循指令？

**答案**：监督微调（Supervised Fine-Tuning, SFT）

```
预训练模型: "今天天气" → "真好，适合出游..."（续写）
SFT 后: "今天天气怎么样？" → "今天天气晴朗，温度适宜。"（回答）
```

## 学习目标

1. 理解 SFT 的数据格式
2. 理解为什么只计算 Assistant 部分的 Loss
3. 实现 SFT 训练
4. 理解并实现 LoRA 高效微调

## 核心概念

### 1. 指令数据格式

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
  ]
}
```

### 2. Chat Template

将对话转换为模型输入格式：

```
<|system|>你是一个有帮助的助手。<|endoftext|>
<|user|>什么是机器学习？<|endoftext|>
<|assistant|>机器学习是人工智能的一个分支...<|endoftext|>
```

### 3. 只计算 Assistant 部分的 Loss

**关键**：SFT 时，我们只希望模型学习"如何回答"，而不是学习"如何提问"。

```python
# 输入: [system_tokens, user_tokens, assistant_tokens]
# 目标: [-100, -100, ..., -100, assistant_tokens]
#       ↑ 不计算 loss    ↑ 计算 loss

loss = F.cross_entropy(logits, targets, ignore_index=-100)
```

### 4. LoRA（Low-Rank Adaptation）

不训练整个模型，只训练低秩的"旁路"矩阵：

```
原始: y = Wx
LoRA: y = Wx + BAx  # B: [n, r], A: [r, m], r << n, m

参数量: 原始需要 n×m 个参数
       LoRA 只需要 (n+m)×r 个参数
```

**优点**：
- 显存占用大幅减少
- 可以保留多个 LoRA 适配器
- 训练速度更快

## 动手任务

### 任务 1：准备 SFT 数据

```bash
python data.py --create_sample
```

### 任务 2：全参数 SFT

```bash
python train_sft.py --device cuda --epochs 3
```

### 任务 3：LoRA 微调

```bash
python train_lora.py --device cuda --epochs 3 --lora_r 8
```

### 任务 4：对话测试

```bash
python chat.py --checkpoint checkpoints/sft_model.pt
```

## 代码文件

- `data.py` - SFT 数据处理
- `train_sft.py` - 全参数 SFT 训练
- `train_lora.py` - LoRA 微调
- `chat.py` - 对话测试

## LoRA 关键参数

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `lora_r` | 低秩维度 | 8-64 |
| `lora_alpha` | 缩放系数 | 16-32 |
| `lora_dropout` | Dropout | 0.05-0.1 |
| `target_modules` | 目标模块 | q_proj, v_proj |

## 验证标准

完成本步骤后，你应该能够：

- [ ] 解释 SFT 和预训练的区别
- [ ] 解释为什么只计算 Assistant 部分的 Loss
- [ ] 实现并运行 SFT 训练
- [ ] 解释 LoRA 的原理和优势
- [ ] 训练一个能进行简单对话的模型

## 进入下一步

SFT 后模型能对话了，但回答质量可能参差不齐。进入 [Step 5: RLHF](../step5_rlhf/)，学习如何让模型输出更符合人类偏好。
