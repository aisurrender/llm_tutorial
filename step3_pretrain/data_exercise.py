"""
Step 3: 数据加载

练习文件：请完成标记为 TODO 的部分

运行测试: python data_exercise.py
"""

import os

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集

    语言建模的核心：给定前文，预测下一个 token
    输入: [t1, t2, t3, ..., tn]
    目标: [t2, t3, t4, ..., t(n+1)]

    例如 block_size=4:
        文本: "hello world"
        样本 0: 输入="hell", 目标="ello"
        样本 1: 输入="ello", 目标="llo "
        ...
    """

    def __init__(self, data_path: str, block_size: int = 256):
        """
        Args:
            data_path: 数据文件路径
            block_size: 序列长度
        """
        self.block_size = block_size

        # 读取文本
        with open(data_path, encoding='utf-8') as f:
            text = f.read()

        # 构建字符词表
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        # 编码
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

        print(f"数据集大小: {len(self.data)} tokens")
        print(f"词表大小: {self.vocab_size}")

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    # =========================================================================
    # TODO 1: 实现 __getitem__ 方法
    # =========================================================================
    def __getitem__(self, idx):
        """
        获取一个训练样本

        语言建模目标：给定 [t1, ..., tn]，预测 [t2, ..., t(n+1)]

        Args:
            idx: 样本索引（即起始位置）

        Returns:
            x: 输入序列 [idx : idx + block_size]
            y: 目标序列 [idx + 1 : idx + block_size + 1]

        示例（block_size=4, idx=0）:
            data = [0, 1, 2, 3, 4, 5, 6, ...]
            x = [0, 1, 2, 3]  # 位置 0-3
            y = [1, 2, 3, 4]  # 位置 1-4（每个位置预测下一个）

        提示:
            - 使用 self.data[start:end] 进行切片
            - x 从 idx 开始，长度为 block_size
            - y 从 idx+1 开始，长度为 block_size
        """
        # TODO: 实现这个方法（2 行代码）
        # x = self.data[idx : idx + self.block_size]
        # y = self.data[idx + 1 : idx + self.block_size + 1]
        # return x, y
        raise NotImplementedError("请实现 __getitem__ 方法")


def create_sample_data(output_path: str = "sample_data.txt"):
    """创建示例数据（莎士比亚风格）"""

    sample_text = """
ROMEO: But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.

JULIET: O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.

MACBETH: Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time,
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
""" * 100  # 重复以增加数据量

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)

    print(f"示例数据已保存到: {output_path}")
    print(f"数据大小: {len(sample_text)} 字符")


# =============================================================================
# 测试代码
# =============================================================================

def test_dataset():
    """测试数据集"""
    # 创建示例数据
    data_path = "sample_data.txt"
    if not os.path.exists(data_path):
        create_sample_data(data_path)

    dataset = PretrainDataset(data_path, block_size=64)

    try:
        x, y = dataset[0]

        # 检查 shape
        assert x.shape == (64,), f"输入 shape 应为 (64,)，得到 {x.shape}"
        assert y.shape == (64,), f"目标 shape 应为 (64,)，得到 {y.shape}"

        # 检查偏移关系：y 应该是 x 右移一位
        # x[1:] 应该等于 y[:-1]
        assert torch.all(x[1:] == y[:-1]), "目标序列应该是输入序列右移一位"

        print("✅ PretrainDataset 测试通过!")

        # 显示示例
        input_text = ''.join([dataset.idx_to_char[i.item()] for i in x[:20]])
        target_text = ''.join([dataset.idx_to_char[i.item()] for i in y[:20]])
        print("\n示例:")
        print(f"  输入: {repr(input_text)}")
        print(f"  目标: {repr(target_text)}")

        return True
    except NotImplementedError as e:
        print(f"⚠️ {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("数据集测试")
    print("=" * 60)
    print()
    test_dataset()
