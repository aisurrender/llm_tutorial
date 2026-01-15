# Step 6: VLM - 多模态扩展

## 核心问题

如何让语言模型"看懂"图片？

**答案**：VLM（Vision Language Model）= Vision Encoder + Projection + LLM

```
图片 → Vision Encoder → 图像特征 → Projection → "图像 Token" → LLM → 文本输出
```

## 学习目标

1. 理解 VLM 的整体架构
2. 理解 Vision Encoder 的作用
3. 理解 Modality Projection 的作用
4. 实现一个简单的 VLM

## 核心概念

### 1. VLM 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        VLM 架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   图片 (224×224×3)                                           │
│         ↓                                                    │
│   ┌─────────────────┐                                        │
│   │ Vision Encoder  │  CLIP-ViT: 图片 → 196×768 特征         │
│   │   (冻结)        │                                        │
│   └─────────────────┘                                        │
│         ↓                                                    │
│   ┌─────────────────┐                                        │
│   │   Projection    │  Linear: 768 → n_embd                  │
│   │   (可训练)      │  将视觉特征映射到文本空间               │
│   └─────────────────┘                                        │
│         ↓                                                    │
│   [img_1, img_2, ..., img_196, text_1, text_2, ...]         │
│         ↓                                                    │
│   ┌─────────────────┐                                        │
│   │      LLM        │  GPT: 接收图像+文本 token               │
│   │   (可训练)      │                                        │
│   └─────────────────┘                                        │
│         ↓                                                    │
│   文本输出                                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Vision Encoder

将图片编码为特征向量序列：

```python
# CLIP-ViT-B/16
# 输入: [B, 3, 224, 224] 图片
# 输出: [B, 196, 768] 特征（196 = 14×14 个 patch）

from transformers import CLIPVisionModel
vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
```

### 3. Modality Projection

将视觉特征映射到文本嵌入空间：

```python
# 简单的线性投影
projection = nn.Linear(vision_dim, text_dim)

# 或 MLP 投影
projection = nn.Sequential(
    nn.Linear(vision_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, text_dim),
)
```

### 4. 训练策略

**两阶段训练**：

| 阶段 | 训练目标 | Vision Encoder | Projection | LLM |
|------|----------|----------------|------------|-----|
| 预训练 | 图像描述 | 冻结 | 训练 | 训练最后几层 |
| SFT | 指令跟随 | 冻结 | 训练 | 全部训练 |

## 动手任务

### 任务 1：理解 VLM 架构

```bash
python model.py --demo
```

### 任务 2：VLM 训练

```bash
python train.py --device cuda --epochs 3
```

### 任务 3：图文对话测试

```bash
python demo.py --image test.jpg --prompt "描述这张图片"
```

## 代码文件

- `model.py` - VLM 模型实现
- `train.py` - VLM 训练脚本
- `demo.py` - 图文对话演示

## 关键点

1. **图片是"特殊的外语"**：Vision Encoder 相当于"翻译官"，把图片翻译成 LLM 能理解的 token
2. **Projection 是桥梁**：连接视觉空间和文本空间
3. **冻结 Vision Encoder**：利用预训练的视觉知识，减少训练成本

## 验证标准

完成本步骤后，你应该能够：

- [ ] 画出 VLM 的架构图
- [ ] 解释 Vision Encoder、Projection、LLM 各自的作用
- [ ] 解释为什么通常冻结 Vision Encoder
- [ ] 训练一个能进行简单图文对话的 VLM

## 恭喜完成！

你已经完成了 LLM/VLM 训练全流程的学习：

```
Step 1: Tokenizer    ✓ 文本 → Token
Step 2: GPT Model    ✓ Transformer 架构
Step 3: Pretrain     ✓ 下一个词预测
Step 4: SFT          ✓ 指令微调
Step 5: RLHF         ✓ 人类偏好对齐
Step 6: VLM          ✓ 多模态扩展
```

现在你已经掌握了大模型训练的核心知识，可以继续探索更多进阶主题！
