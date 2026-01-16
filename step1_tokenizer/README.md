# Step 1: Tokenizer - 文本如何变成数字

## 核心问题

神经网络只能处理数字，那么文本是如何变成数字的？

**答案**：Tokenizer（分词器）

```
"Hello world" → Tokenizer → [15496, 995]
```

## 学习方式

本教程采用**动手实践**的方式：

1. 打开 `tutorial.ipynb` 作为主学习界面
2. 去 `tokenizer_exercise.py` 完成 TODO 部分
3. 回到 Notebook 运行验证
4. 卡住了？查看 `tokenizer_solution.py`

## 文件结构

| 文件 | 说明 |
|------|------|
| `tutorial.ipynb` | **主学习界面** - 概念讲解、运行代码、可视化 |
| `tokenizer_exercise.py` | **练习文件** - 完成 TODO 部分 |
| `tokenizer_solution.py` | **参考答案** - 卡住时查看 |
| `train_bpe.py` | **实战练习** - 使用 SentencePiece 训练工业级分词器 |
| `sample_data.txt` | 示例训练数据（中英文混合） |

## 学习目标

1. 理解 Tokenizer 的作用
2. **动手实现**字符级 Tokenizer（encode/decode）
3. **动手实现** BPE 算法核心（统计、合并）
4. （进阶）理解 Byte-level BPE 的优势
5. （实战）使用 SentencePiece 训练中文分词器

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

## 练习任务

### 基础练习

#### TODO 1: 实现 `encode` 方法（简单）
将文本转换为 token ID 列表

#### TODO 2: 实现 `decode` 方法（简单）
将 token ID 列表还原为文本

#### TODO 3: 实现 `_get_stats` 方法（中等）
统计相邻 token 对的频率

#### TODO 4: 完成 BPE 训练循环（较难）
- 4a: 获取 pair 频率
- 4b: 找到频率最高的 pair
- 4c: 合并这个 pair

### 进阶练习（选做）

#### TODO 5: 实现 `_merge_vocab` 方法
当前 `_merge_vocab` 已经实现，但你可以尝试：
1. 删除现有实现，自己重写
2. 理解为什么用字符串替换而不是遍历

#### TODO 6: 实现 Byte-level BPE
挑战：修改 BPETokenizer，使其在字节级别而非字符级别工作：
- 将文本先编码为 UTF-8 字节
- 初始词表为 256 个字节值
- 这是 GPT-2/3/4 实际使用的方式

### 实战练习：使用 SentencePiece

完成基础练习后，尝试用工业级工具训练分词器：

```bash
# 1. 查看示例数据
cat sample_data.txt

# 2. 训练 SentencePiece 分词器
uv run python train_bpe.py --data sample_data.txt --vocab_size 1000

# 3. 测试训练好的分词器
uv run python train_bpe.py --demo
```

思考：对比你手写的 BPE 和 SentencePiece 的结果有何不同？

## 验证你的实现

```bash
# 运行测试
python tokenizer_exercise.py

# 或在 Jupyter 中验证
jupyter notebook tutorial.ipynb
```

## 验证清单

完成本步骤后，你应该能够：

**基础**
- [ ] 解释 Tokenizer 的作用和为什么需要它
- [ ] 用自己的代码将文本编码为 token IDs
- [ ] 用自己的代码将 token IDs 解码回文本
- [ ] 理解并实现 BPE 算法的统计和合并过程

**进阶**
- [ ] 解释 Byte-level BPE 相比字符级 BPE 的优势
- [ ] 区分 BPE、WordPiece、Unigram 三种算法的核心差异
- [ ] 使用 SentencePiece 训练一个中文分词器

## 推荐阅读

想深入了解 Tokenizer？以下是优质学习资源：

### GitHub 仓库

| 仓库 | 说明 |
|------|------|
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | Karpathy 的极简 BPE 实现，附带视频讲解和练习 |
| [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | 工业级实现，支持 BPE/WordPiece/Unigram |
| [google/sentencepiece](https://github.com/google/sentencepiece) | Google 的无监督分词器，支持 BPE 和 Unigram |
| [openai/tiktoken](https://github.com/openai/tiktoken) | OpenAI 的快速 BPE 分词器（GPT-3.5/4 使用） |
| [gautierdag/bpeasy](https://github.com/gautierdag/bpeasy) | 快速 Byte-level BPE 训练器（Rust 实现） |

### 分词算法对比

| 算法 | 特点 | 代表模型 |
|------|------|----------|
| **BPE** | 基于频率合并，简单高效 | GPT-2/3/4, LLaMA |
| **WordPiece** | 基于似然合并，考虑上下文 | BERT, DistilBERT |
| **Unigram** | 概率模型，从大词表裁剪 | T5, ALBERT, XLNet |
| **Byte-level BPE** | 字节级处理，无 OOV | GPT-2/3/4, RoBERTa |

### 延伸阅读

- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) - Karpathy 2小时视频教程
- [Hugging Face Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary) - 各类分词器综述
- [BPE From Scratch (2025)](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) - Sebastian Raschka 的实现教程

## 进入下一步

理解 Tokenizer 后，进入 [Step 2: GPT Model](../step2_gpt_model/)，学习如何构建处理这些 token 的模型。
