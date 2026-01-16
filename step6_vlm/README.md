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
| `train_solution.py` | **训练脚本** - 两阶段训练参考 |

## 练习任务

### 基础练习

#### TODO 1: 实现 Projection Layer

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

### 进阶练习（选做）

#### TODO 2: 对比不同 Projection 设计

挑战：实现并对比不同的 Projection 设计：

| 设计 | 结构 | 特点 |
|------|------|------|
| Linear | `Linear(v, t)` | 最简单，参数少 |
| MLP | `Linear → GELU → Linear` | LLaVA 使用 |
| C-Abstractor | `Conv + Attention` | 压缩视觉 token 数量 |

#### TODO 3: 理解两阶段训练

查看 `train_solution.py`，理解 VLM 的两阶段训练：

```bash
cat train_solution.py
```

```
Stage 1: 预训练（特征对齐）
- 冻结：Vision Encoder + LLM
- 训练：Projection
- 数据：图文对（CC3M 等）

Stage 2: 指令微调
- 冻结：Vision Encoder
- 训练：Projection + LLM
- 数据：视觉问答、图像描述
```

#### TODO 4: 运行 VLM 训练

```bash
# 运行训练脚本
uv run python train_solution.py
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

完成本步骤后，你应该能够：

**基础**
- [ ] 画出 VLM 的架构图
- [ ] 解释各组件的作用（Vision Encoder, Projection, LLM）
- [ ] 解释为什么冻结 Vision Encoder
- [ ] 实现 Projection Layer

**进阶**
- [ ] 解释两阶段训练的目的
- [ ] 对比不同 Projection 设计的优劣
- [ ] 了解视觉 token 压缩技术

## 推荐阅读

### GitHub 仓库

| 仓库 | 说明 |
|------|------|
| [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) | 极简 VLM 实现（~750 行） |
| [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | 经典 VLM 实现 |
| [gokayfem/awesome-vlm-architectures](https://github.com/gokayfem/awesome-vlm-architectures) | VLM 架构大全 |
| [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) | 开源最强 VLM 之一 |

### 论文

| 论文 | 说明 |
|------|------|
| [LLaVA](https://arxiv.org/abs/2304.08485) | Visual Instruction Tuning |
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | 改进版，MLP Projection |
| [CLIP](https://arxiv.org/abs/2103.00020) | 视觉编码器基础 |

### 教程

| 资源 | 说明 |
|------|------|
| [nanoVLM 教程](https://huggingface.co/blog/nanovlm) | Hugging Face 官方教程 |
| [LLaVA 官网](https://llava-vl.github.io/) | 架构和训练详解 |

### VLM 架构演进

| 模型 | 年份 | Vision Encoder | Projection | 特点 |
|------|------|----------------|------------|------|
| **LLaVA** | 2023 | CLIP ViT-L | Linear | 开创性工作 |
| **LLaVA-1.5** | 2023 | CLIP ViT-L | MLP | 更好的对齐 |
| **Qwen-VL** | 2023 | ViT-bigG | C-Abstractor | 压缩视觉 token |
| **InternVL** | 2024 | InternViT | MLP | 更大视觉编码器 |

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

## 下一步

恭喜你完成了整个教程！接下来可以：

1. **实战项目**：用真实数据训练一个小型 LLM
2. **深入阅读**：研读推荐的论文和代码
3. **贡献开源**：参与 nanoGPT、nanoVLM 等项目
4. **关注前沿**：MoE、长上下文、Agent 等新方向
