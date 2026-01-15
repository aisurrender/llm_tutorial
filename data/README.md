# 数据目录

本目录用于存放训练数据。

## 数据说明

各 Step 会自动创建示例数据用于演示。如果需要使用真实数据，请按以下格式准备：

### Step 3: 预训练数据

纯文本文件：

```
sample_data.txt
```

### Step 4: SFT 数据

JSONL 格式：

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Step 5: DPO 数据

JSONL 格式：

```json
{"prompt": "问题", "chosen": "好的回答", "rejected": "差的回答"}
```

### Step 6: VLM 数据

JSONL 格式：

```json
{"image": "path/to/image.jpg", "text": "图片描述"}
```

## 推荐数据集

- **预训练**: Wikipedia, Common Crawl
- **SFT**: Alpaca, ShareGPT
- **DPO**: HH-RLHF, UltraFeedback
- **VLM**: LLaVA-Instruct, ShareGPT4V
