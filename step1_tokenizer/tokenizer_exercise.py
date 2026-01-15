"""
Step 1: Tokenizer - 文本如何变成数字

练习文件：请完成标记为 TODO 的部分

运行测试: python -m pytest tokenizer_exercise.py -v
或在 tutorial.ipynb 中验证
"""

import re
from collections import defaultdict

# =============================================================================
# 1. 字符级 Tokenizer（最简单的实现）
# =============================================================================

class CharTokenizer:
    """
    字符级分词器：每个字符都是一个 token

    优点：词表小，无 OOV（Out of Vocabulary）问题
    缺点：序列太长，效率低

    示例:
        "hello" -> [7, 4, 11, 11, 14]  # 假设 h=7, e=4, l=11, o=14
    """

    def __init__(self):
        # 特殊 token
        self.special_tokens = {
            '<pad>': 0,   # 填充
            '<unk>': 1,   # 未知字符
            '<bos>': 2,   # 句子开始
            '<eos>': 3,   # 句子结束
        }
        self.vocab = dict(self.special_tokens)  # 字符 -> ID
        self.id_to_char = {v: k for k, v in self.vocab.items()}  # ID -> 字符

    def build_vocab(self, text: str):
        """从文本构建词表"""
        for char in text:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.id_to_char[idx] = char
        print(f"词表大小: {len(self.vocab)}")

    # =========================================================================
    # TODO 1: 实现 encode 方法（简单）
    # =========================================================================
    def encode(self, text: str) -> list:
        """
        文本 -> Token IDs

        任务：遍历 text 中的每个字符，查找其在 self.vocab 中的 ID

        提示：
        - 使用 self.vocab.get(char, default) 获取 ID
        - 如果字符不在词表中，返回 <unk> 的 ID
        - <unk> 的 ID 是 self.special_tokens['<unk>']

        示例：
            >>> tokenizer.build_vocab("hello")
            >>> tokenizer.encode("hello")
            [4, 5, 6, 6, 7]
        """
        # TODO: 实现这个方法（大约 1 行代码）
        # return [... for char in text]
        raise NotImplementedError("请实现 encode 方法")

    # =========================================================================
    # TODO 2: 实现 decode 方法（简单）
    # =========================================================================
    def decode(self, ids: list) -> str:
        """
        Token IDs -> 文本

        任务：遍历 ids 中的每个 ID，查找其对应的字符

        提示：
        - 使用 self.id_to_char.get(idx, default) 获取字符
        - 如果 ID 不在映射中，返回 '<unk>'
        - 使用 ''.join() 将字符列表合并为字符串

        示例：
            >>> tokenizer.decode([4, 5, 6, 6, 7])
            "hello"
        """
        # TODO: 实现这个方法（大约 1 行代码）
        # return ''.join([... for idx in ids])
        raise NotImplementedError("请实现 decode 方法")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# =============================================================================
# 2. BPE Tokenizer（子词级分词）
# =============================================================================

class BPETokenizer:
    """
    BPE (Byte Pair Encoding) 分词器

    核心思想：反复合并最高频的相邻 token 对

    训练过程：
    1. 初始词表 = 所有字符
    2. 统计相邻 token 对的频率
    3. 合并频率最高的 token 对，加入词表
    4. 重复 2-3 直到达到目标词表大小

    示例:
        训练语料: "low lower lowest"
        合并过程:
          l+o -> lo (高频)
          lo+w -> low (高频)
        最终: "lowest" -> ["low", "est"]
    """

    def __init__(self):
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        self.vocab = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = {}  # (token1, token2) -> merged_token 的合并规则

    # =========================================================================
    # TODO 3: 实现 _get_stats 方法（中等）
    # =========================================================================
    def _get_stats(self, words: list) -> dict:
        """
        统计相邻 token 对的频率

        Args:
            words: [(word_str, freq), ...] 列表
                   word_str 是空格分隔的 token 序列，如 "l o w"
                   freq 是这个词出现的次数

        Returns:
            dict: {(token1, token2): count, ...}

        示例：
            words = [("l o w", 5), ("l o w e r", 2)]
            返回: {('l', 'o'): 7, ('o', 'w'): 7, ('w', 'e'): 2, ('e', 'r'): 2}

        提示：
        - 遍历每个 (word, freq) 对
        - 用 word.split() 将 word 分割成 token 列表
        - 遍历相邻的 token 对，累加频率
        """
        pairs = defaultdict(int)

        # TODO: 实现统计逻辑（大约 4 行代码）
        # for word, freq in words:
        #     symbols = word.split()
        #     for i in range(...):
        #         pairs[...] += freq

        raise NotImplementedError("请实现 _get_stats 方法")

        return pairs

    def _merge_vocab(self, pair: tuple, words: list) -> list:
        """合并指定的 token 对（已实现）"""
        new_words = []
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in words:
            new_word = word.replace(bigram, replacement)
            new_words.append((new_word, freq))
        return new_words

    # =========================================================================
    # TODO 4: 完成 train 方法中的关键步骤（较难）
    # =========================================================================
    def train(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        """
        从文本训练 BPE 词表

        Args:
            text: 训练文本
            vocab_size: 目标词表大小
            verbose: 是否打印训练过程
        """
        # Step 1: 统计词频，每个字符用空格分开
        word_freqs = defaultdict(int)
        words = re.findall(r'\S+', text)  # 按空格分词
        for word in words:
            # 将词转换为字符序列，字符之间用空格分开
            char_seq = ' '.join(list(word))
            word_freqs[char_seq] += 1

        words = list(word_freqs.items())

        # 初始词表 = 特殊token + 所有字符
        for word, _ in words:
            for char in word.split():
                if char not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[char] = idx
                    self.id_to_token[idx] = char

        if verbose:
            print(f"初始词表大小: {len(self.vocab)}")

        # Step 2: 反复合并最高频的 token 对
        num_merges = vocab_size - len(self.vocab)

        for i in range(num_merges):
            # TODO 4a: 调用 _get_stats 获取所有 pair 的频率
            # pairs = ...
            pairs = self._get_stats(words)

            if not pairs:
                break

            # TODO 4b: 找到频率最高的 pair
            # 提示: 使用 max(pairs, key=pairs.get)
            # best_pair = ...
            raise NotImplementedError("请找到频率最高的 pair")

            # TODO 4c: 使用 _merge_vocab 合并这个 pair
            # words = self._merge_vocab(...)
            raise NotImplementedError("请合并 best_pair")

            # 添加新 token 到词表
            new_token = ''.join(best_pair)
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.id_to_token[idx] = new_token
                self.merges[best_pair] = new_token

            if verbose and (i + 1) % 100 == 0:
                print(f"合并 {i + 1}/{num_merges}: {best_pair} -> {new_token}")

        if verbose:
            print(f"最终词表大小: {len(self.vocab)}")

    def encode(self, text: str) -> list:
        """文本 -> Token IDs（已实现）"""
        tokens = []
        words = re.findall(r'\S+|\s+', text)  # 保留空格

        for word in words:
            if word.isspace():
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                continue

            # 将词分解为字符
            word_tokens = list(word)

            # 应用合并规则
            while len(word_tokens) > 1:
                pairs = [(word_tokens[i], word_tokens[i + 1])
                        for i in range(len(word_tokens) - 1)]
                mergeable = [(i, p) for i, p in enumerate(pairs) if p in self.merges]

                if not mergeable:
                    break

                idx, pair = mergeable[0]
                word_tokens = (word_tokens[:idx] +
                              [self.merges[pair]] +
                              word_tokens[idx + 2:])

            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.special_tokens['<unk>'])

        return tokens

    def decode(self, ids: list) -> str:
        """Token IDs -> 文本（已实现）"""
        return ''.join([self.id_to_token.get(idx, '<unk>') for idx in ids])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# =============================================================================
# 测试代码（用于验证你的实现）
# =============================================================================

def test_char_tokenizer():
    """测试字符级分词器"""
    tokenizer = CharTokenizer()
    tokenizer.build_vocab("hello world")

    # 测试 encode
    ids = tokenizer.encode("hello")
    assert isinstance(ids, list), "encode 应返回 list"
    assert len(ids) == 5, f"'hello' 应该编码为 5 个 token，得到 {len(ids)}"

    # 测试 decode
    decoded = tokenizer.decode(ids)
    assert decoded == "hello", f"解码结果应为 'hello'，得到 '{decoded}'"

    # 测试往返一致性
    text = "hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text

    print("CharTokenizer 测试通过!")


def test_bpe_stats():
    """测试 BPE 统计函数"""
    tokenizer = BPETokenizer()

    words = [("l o w", 5), ("l o w e r", 2)]
    stats = tokenizer._get_stats(words)

    assert stats[('l', 'o')] == 7, f"('l', 'o') 应出现 7 次，得到 {stats.get(('l', 'o'), 0)}"
    assert stats[('o', 'w')] == 7, f"('o', 'w') 应出现 7 次，得到 {stats.get(('o', 'w'), 0)}"

    print("BPE _get_stats 测试通过!")


def test_bpe_train():
    """测试 BPE 训练"""
    tokenizer = BPETokenizer()

    text = "low lower lowest " * 10
    tokenizer.train(text, vocab_size=50, verbose=False)

    # 检查是否学到了合并规则
    assert len(tokenizer.merges) > 0, "应该学到一些合并规则"

    # 检查编码解码一致性
    test_text = "lower"
    decoded = tokenizer.decode(tokenizer.encode(test_text))
    assert decoded == test_text, f"编码解码不一致: {test_text} -> {decoded}"

    print("BPE train 测试通过!")


if __name__ == "__main__":
    print("运行测试...\n")

    try:
        test_char_tokenizer()
    except NotImplementedError as e:
        print(f"CharTokenizer: {e}")
    except AssertionError as e:
        print(f"CharTokenizer 测试失败: {e}")

    print()

    try:
        test_bpe_stats()
    except NotImplementedError as e:
        print(f"BPE _get_stats: {e}")
    except AssertionError as e:
        print(f"BPE _get_stats 测试失败: {e}")

    print()

    try:
        test_bpe_train()
    except NotImplementedError as e:
        print(f"BPE train: {e}")
    except AssertionError as e:
        print(f"BPE train 测试失败: {e}")
