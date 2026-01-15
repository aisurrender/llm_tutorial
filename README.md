# LLM 训练全流程教程

**从零开始，6 步掌握大模型训练全流程**

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM 训练全流程                            │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Tokenizer     文本 → Token ID                      │
│  Step 2: GPT Model     Transformer 架构                     │
│  Step 3: Pretrain      从零预训练语言模型                    │
│  Step 4: SFT           指令微调，让模型学会对话               │
│  Step 5: RLHF          人类偏好对齐 (DPO)                    │
│  Step 6: VLM           多模态扩展，图文理解                   │
└─────────────────────────────────────────────────────────────┘
```

## 设计理念

- **从简**：每个 Step 聚焦一个核心概念，代码精简到 100-200 行
- **连贯**：统一的代码风格，Step 之间自然衔接
- **干中学**：每个 Step 都有可运行的训练任务

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/xxx/llm_tutorial.git
cd llm_tutorial

# 安装依赖
pip install -r requirements.txt

# 从 Step 1 开始
cd step1_tokenizer
python tokenizer.py
```

## 学习路线

| Step | 内容 | 核心代码 | 硬件需求 | 学习时间 |
|------|------|----------|----------|----------|
| [Step 1](step1_tokenizer/) | Tokenizer - 文本如何变成数字 | ~100行 | CPU | 30分钟 |
| [Step 2](step2_gpt_model/) | GPT Model - Transformer 架构 | ~200行 | CPU | 1小时 |
| [Step 3](step3_pretrain/) | Pretrain - 预训练语言模型 | ~150行 | GPU推荐 | 2小时 |
| [Step 4](step4_sft/) | SFT - 指令微调 | ~100行 | GPU推荐 | 1小时 |
| [Step 5](step5_rlhf/) | RLHF - 人类偏好对齐 | ~100行 | GPU | 1小时 |
| [Step 6](step6_vlm/) | VLM - 多模态扩展 | ~200行 | GPU | 2小时 |

**总计**：约 8 小时完成全流程，GPU 成本约 10-20 元（AutoDL）

## 硬件建议

- **Step 1-2**：MacBook / 任意 CPU 即可
- **Step 3-4**：推荐 GPU，但 CPU/MPS 也能跑小规模实验
- **Step 5-6**：需要 GPU（推荐 RTX 3090 或 AutoDL 租用）

## 每个 Step 你会学到什么

### Step 1: Tokenizer
- 理解文本如何转换为模型输入
- 实现字符级和 BPE 分词器
- 使用 SentencePiece 训练中文分词器

### Step 2: GPT Model
- 从零实现 Multi-Head Attention
- 理解 LayerNorm、Residual Connection
- 组装完整的 GPT 模型

### Step 3: Pretrain
- 理解"下一个词预测"的训练目标
- 实现训练循环和学习率调度
- 在小数据集上预训练模型

### Step 4: SFT
- 理解指令微调的数据格式
- 实现 SFT 训练（只计算 Assistant 部分的 Loss）
- 可选：LoRA 高效微调

### Step 5: RLHF
- 理解人类偏好对齐的目标
- 实现 DPO（Direct Preference Optimization）
- 对比 SFT 和 DPO 后的效果

### Step 6: VLM
- 理解多模态模型架构
- 实现 Vision Encoder + Projection + LLM
- 训练一个能看图说话的模型

## 参考资源

本教程融合了以下优秀项目的精华：

- [nanoGPT](https://github.com/karpathy/nanoGPT) - 最简洁的 GPT 实现
- [MiniMind](https://github.com/jingyaogong/minimind) - 完整的中文 LLM 训练流程
- [Stanford CS336](https://stanford-cs336.github.io/spring2025/) - 系统化的 LLM 课程
- [nanoVLM](https://github.com/huggingface/nanoVLM) - 教育性的 VLM 框架
- [MiniMind-V](https://github.com/jingyaogong/minimind-v) - 轻量 VLM 实现

## 目录结构

```
llm_tutorial/
├── README.md                 # 本文件
├── requirements.txt          # 依赖包
│
├── step1_tokenizer/          # Step 1: 分词器
├── step2_gpt_model/          # Step 2: GPT 模型
├── step3_pretrain/           # Step 3: 预训练
├── step4_sft/                # Step 4: 指令微调
├── step5_rlhf/               # Step 5: 人类对齐
├── step6_vlm/                # Step 6: 多模态
│
├── shared/                   # 共享组件
├── data/                     # 数据目录
└── docs/                     # 文档
```

## License

MIT
