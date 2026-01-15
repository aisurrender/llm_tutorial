"""
Step 1: Tokenizer - 文本如何变成数字

本文件包含两种 Tokenizer 实现：
1. CharTokenizer - 字符级分词器（最简单）
2. BPETokenizer - BPE 分词器（实际使用）

运行: python tokenizer.py
"""

from collections import defaultdict
import re


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

    def encode(self, text: str) -> list:
        """文本 -> Token IDs"""
        return [self.vocab.get(char, self.special_tokens['<unk>']) for char in text]

    def decode(self, ids: list) -> str:
        """Token IDs -> 文本"""
        return ''.join([self.id_to_char.get(idx, '<unk>') for idx in ids])

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

    def _get_stats(self, words: list) -> dict:
        """统计相邻 token 对的频率"""
        pairs = defaultdict(int)
        for word, freq in words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: tuple, words: list) -> list:
        """合并指定的 token 对"""
        new_words = []
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in words:
            new_word = word.replace(bigram, replacement)
            new_words.append((new_word, freq))
        return new_words

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
            pairs = self._get_stats(words)
            if not pairs:
                break

            # 找到频率最高的 token 对
            best_pair = max(pairs, key=pairs.get)

            # 合并这个 token 对
            words = self._merge_vocab(best_pair, words)

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
        """文本 -> Token IDs"""
        tokens = []
        words = re.findall(r'\S+|\s+', text)  # 保留空格

        for word in words:
            if word.isspace():
                # 空格单独处理
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                continue

            # 将词分解为字符
            word_tokens = list(word)

            # 应用合并规则
            while len(word_tokens) > 1:
                # 找到可以合并的 pair
                pairs = [(word_tokens[i], word_tokens[i + 1])
                        for i in range(len(word_tokens) - 1)]

                # 找到在 merges 中存在的 pair
                mergeable = [(i, p) for i, p in enumerate(pairs) if p in self.merges]

                if not mergeable:
                    break

                # 合并第一个可合并的 pair
                idx, pair = mergeable[0]
                word_tokens = (word_tokens[:idx] +
                              [self.merges[pair]] +
                              word_tokens[idx + 2:])

            # 转换为 IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.special_tokens['<unk>'])

        return tokens

    def decode(self, ids: list) -> str:
        """Token IDs -> 文本"""
        return ''.join([self.id_to_token.get(idx, '<unk>') for idx in ids])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# =============================================================================
# 演示代码
# =============================================================================

def demo_char_tokenizer():
    """演示字符级分词器"""
    print("=" * 60)
    print("字符级 Tokenizer 演示")
    print("=" * 60)

    tokenizer = CharTokenizer()

    # 训练文本
    text = "Hello world! 你好世界！"
    tokenizer.build_vocab(text)

    # 编码
    test_text = "Hello 你好"
    ids = tokenizer.encode(test_text)
    print(f"\n原文: {test_text}")
    print(f"编码: {ids}")

    # 解码
    decoded = tokenizer.decode(ids)
    print(f"解码: {decoded}")

    # 验证
    print(f"\n编码-解码一致: {test_text == decoded}")
    print(f"词表大小: {tokenizer.vocab_size}")


def demo_bpe_tokenizer():
    """演示 BPE 分词器"""
    print("\n" + "=" * 60)
    print("BPE Tokenizer 演示")
    print("=" * 60)

    tokenizer = BPETokenizer()

    # 训练文本（重复一些词以展示 BPE 效果）
    text = """
    low lower lowest lowly
    new newer newest newly
    show showed showing shown
    the quick brown fox jumps over the lazy dog
    the the the quick quick brown brown
    """

    # 训练 BPE（小词表用于演示）
    tokenizer.train(text, vocab_size=100, verbose=True)

    # 编码
    test_text = "lower showing"
    ids = tokenizer.encode(test_text)
    print(f"\n原文: {test_text}")
    print(f"编码: {ids}")

    # 解码
    decoded = tokenizer.decode(ids)
    print(f"解码: {decoded}")

    # 查看一些学到的合并规则
    print(f"\n学到的部分合并规则:")
    for i, (pair, merged) in enumerate(list(tokenizer.merges.items())[:10]):
        print(f"  {pair} -> {merged}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenizer 演示")
    parser.add_argument("--mode", choices=["char", "bpe", "all"], default="all",
                       help="演示模式: char=字符级, bpe=BPE, all=全部")
    args = parser.parse_args()

    if args.mode in ["char", "all"]:
        demo_char_tokenizer()

    if args.mode in ["bpe", "all"]:
        demo_bpe_tokenizer()

    print("\n" + "=" * 60)
    print("下一步: 进入 step2_gpt_model/ 学习 GPT 模型架构")
    print("=" * 60)
