# Step 6: VLM - 多模态扩展

## 核心问题

如何让语言模型"看懂"图片？

**答案**：VLM = Vision Encoder + Projection + LLM

```
图片 → Vision Encoder → 图像特征 → Projection → "图像Token" → LLM → 文本
```

## 学习方式

1. 打开 `tutorial.ipynb` 作为主学习界面
2. 去 `model_exercise.py` 完成 TODO
3. 卡住了？查看 `*_solution.py`

## 文件结构

| 文件 | 说明 |
|------|------|
| `tutorial.ipynb` | **主学习界面** |
| `model_exercise.py` | **模型练习** - 完成 TODO |
| `model_solution.py` | 模型完整实现 |
| `train_solution.py` | 训练脚本参考 |

## 练习任务

### TODO 1: 实现 Projection Layer

将视觉特征映射到文本嵌入空间：

```python
# 简单线性层
self.projection = nn.Linear(vision_dim, text_dim)

# 或 MLP
self.projection = nn.Sequential(
    nn.Linear(vision_dim, text_dim * 2),
    nn.GELU(),
    nn.Linear(text_dim * 2, text_dim),
)
```

## 核心概念

### VLM 架构

```
Vision Encoder (冻结)   → 图片 → [196, 768] 特征
        ↓
Projection (训练)       → [196, text_dim]
        ↓
LLM (训练)              → 生成文本回复
```

### 各组件作用

| 组件 | 作用 | 训练状态 |
|------|------|----------|
| Vision Encoder | 图片→特征 | 冻结 |
| Projection | 维度对齐 | 训练 |
| LLM | 生成回复 | 训练 |

### 为什么冻结 Vision Encoder？

1. 利用预训练的视觉知识
2. 减少训练成本
3. 防止遗忘

## 验证清单

- [ ] 画出 VLM 的架构图
- [ ] 解释各组件的作用
- [ ] 解释为什么冻结 Vision Encoder
- [ ] 实现 Projection Layer

---

## 恭喜完成！

你已经完成了 LLM/VLM 训练全流程：

```
Step 1: Tokenizer    ✓ 文本 → Token
Step 2: GPT Model    ✓ Transformer 架构
Step 3: Pretrain     ✓ 下一个词预测
Step 4: SFT          ✓ 指令微调
Step 5: RLHF         ✓ 人类偏好对齐
Step 6: VLM          ✓ 多模态扩展
```
