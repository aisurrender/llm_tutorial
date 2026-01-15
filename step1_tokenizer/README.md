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
| `train_bpe.py` | 使用 SentencePiece 训练分词器 |

## 学习目标

1. 理解 Tokenizer 的作用
2. **动手实现**字符级 Tokenizer
3. **动手实现** BPE 算法核心
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

## 练习任务

### TODO 1: 实现 `encode` 方法（简单）
将文本转换为 token ID 列表

### TODO 2: 实现 `decode` 方法（简单）
将 token ID 列表还原为文本

### TODO 3: 实现 `_get_stats` 方法（中等）
统计相邻 token 对的频率

### TODO 4: 完成 BPE 训练循环（较难）
- 4a: 获取 pair 频率
- 4b: 找到频率最高的 pair
- 4c: 合并这个 pair

## 验证你的实现

```bash
# 运行测试
python tokenizer_exercise.py

# 或在 Jupyter 中验证
jupyter notebook tutorial.ipynb
```

## 验证清单

完成本步骤后，你应该能够：

- [ ] 解释 Tokenizer 的作用
- [ ] 用自己的代码将文本编码为 token IDs
- [ ] 用自己的代码将 token IDs 解码回文本
- [ ] 理解并实现 BPE 算法的合并过程
- [ ] 训练一个简单的中文分词器

## 进入下一步

理解 Tokenizer 后，进入 [Step 2: GPT Model](../step2_gpt_model/)，学习如何构建处理这些 token 的模型。
