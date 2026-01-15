# Step 1: Tokenizer - 文本如何变成数字

## 核心问题

神经网络只能处理数字，那么文本是如何变成数字的？

**答案**：Tokenizer（分词器）

```
"Hello world" → Tokenizer → [15496, 995]
```

## 学习目标

1. 理解 Tokenizer 的作用
2. 实现字符级 Tokenizer
3. 理解并实现 BPE（Byte Pair Encoding）算法
4. 使用 SentencePiece 训练中文分词器

## 核心概念

### 1. 为什么需要 Tokenizer？

- 神经网络输入必须是数字（张量）
- 需要一种方式将文本映射为数字序列
- 映射必须是**可逆的**（能从数字还原回文本）

### 2. 三种分词粒度

| 粒度 | 例子 | 优点 | 缺点 |
|------|------|------|------|
| 字符级 | "hello" → ["h","e","l","l","o"] | 词表小，无 OOV | 序列太长 |
| 词级 | "hello world" → ["hello", "world"] | 语义清晰 | 词表大，有 OOV |
| 子词级 (BPE) | "unhappiness" → ["un", "happi", "ness"] | 平衡 | 需要训练 |

### 3. BPE 算法原理

BPE 的核心思想：**高频字符对合并**

```
初始词表: a, b, c, d, ...
语料: "abab cdcd abab"

Step 1: 统计字符对频率
  "ab" 出现 4 次（最高）

Step 2: 合并 "ab" 为新 token "ab"
  词表: a, b, c, d, ..., ab
  语料: "ab ab cdcd ab ab"

Step 3: 重复直到词表大小达到目标
```

## 动手任务

### 任务 1：理解字符级 Tokenizer

```bash
python tokenizer.py --mode char
```

查看 `CharTokenizer` 类，理解：
- `encode()`: 文本 → token IDs
- `decode()`: token IDs → 文本

### 任务 2：实现 BPE 算法

```bash
python tokenizer.py --mode bpe
```

查看 `BPETokenizer` 类，理解：
- `train()`: 从语料训练词表
- 合并规则是如何学习的

### 任务 3：训练中文分词器

```bash
python train_bpe.py --data your_chinese_text.txt --vocab_size 6400
```

使用 SentencePiece 训练一个中文分词器。

## 代码文件

- `tokenizer.py` - 核心代码，包含字符级和 BPE 分词器实现
- `train_bpe.py` - 使用 SentencePiece 训练分词器

## 验证标准

完成本步骤后，你应该能够：

- [ ] 解释 Tokenizer 的作用
- [ ] 用自己的代码将文本编码为 token IDs
- [ ] 用自己的代码将 token IDs 解码回文本
- [ ] 理解 BPE 算法的合并过程
- [ ] 训练一个简单的中文分词器

## 进入下一步

理解 Tokenizer 后，进入 [Step 2: GPT Model](../step2_gpt_model/)，学习如何构建处理这些 token 的模型。
