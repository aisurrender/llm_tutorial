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
| `train_sft_solution.py` | **SFT 训练参考** - 全参数微调 |
| `train_lora_solution.py` | **LoRA 训练参考** - 高效微调 |

## 练习任务

### 基础练习

#### TODO 1: 实现 _format_conversation

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

### 进阶练习（选做）

#### TODO 2: 自己实现 LoRA 层

挑战：不使用 PEFT 库，手动实现 LoRA：

```python
class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False  # 冻结原始权重

        # TODO: 实现低秩矩阵 A 和 B
        # A: (in_features, r), B: (r, out_features)
        # 初始化: A 用高斯，B 用零

    def forward(self, x):
        # TODO: y = original(x) + (x @ A @ B) * (alpha / r)
        pass
```

#### TODO 3: 运行 LoRA 训练

完成基础练习后，尝试用 LoRA 进行高效微调：

```bash
# 查看 LoRA 训练脚本
cat train_lora_solution.py

# 运行 LoRA 训练（显存需求大幅降低）
uv run python train_lora_solution.py
```

思考：对比全参数微调和 LoRA 微调的效果与效率差异

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

完成本步骤后，你应该能够：

**基础**
- [ ] 解释 SFT 和预训练的区别
- [ ] 解释为什么只计算 Assistant 部分的 Loss
- [ ] 实现 labels 的构建逻辑（-100 掩码）

**进阶**
- [ ] 解释 LoRA 的原理（低秩分解）
- [ ] 解释 LoRA 相比全参数微调的优势
- [ ] 了解 r 和 alpha 参数的作用

## 推荐阅读

### GitHub 仓库

| 仓库 | 说明 |
|------|------|
| [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) | 经典 LoRA 微调实现 |
| [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | 中文 LLaMA 微调 |
| [huggingface/peft](https://github.com/huggingface/peft) | 参数高效微调库（LoRA, QLoRA 等） |

### 论文

| 论文 | 说明 |
|------|------|
| [LoRA](https://arxiv.org/abs/2106.09685) | Low-Rank Adaptation 原始论文 |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 4-bit 量化 + LoRA，进一步降低显存 |
| [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) | Stanford 指令微调数据集 |

### 高效微调技术对比

| 方法 | 可训练参数 | 显存需求 | 效果 |
|------|-----------|---------|------|
| **全参数微调** | 100% | 高 | 最好 |
| **LoRA** | ~0.1% | 低 | 接近全参数 |
| **QLoRA** | ~0.1% | 更低 | 接近 LoRA |
| **Prefix Tuning** | <0.1% | 最低 | 略差 |

## 进入下一步

进入 [Step 5: RLHF](../step5_rlhf/)，学习如何让模型输出更符合人类偏好。
