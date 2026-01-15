"""
Step 3: 数据加载

本文件实现预训练数据的加载和预处理。

运行: python data.py --create_sample
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集

    将文本切分成固定长度的序列，用于语言建模训练。
    输入: [t1, t2, t3, ..., tn]
    目标: [t2, t3, t4, ..., t(n+1)]
    """

    def __init__(self, data_path: str, block_size: int = 256):
        """
        Args:
            data_path: 数据文件路径（纯文本或已编码的 .bin 文件）
            block_size: 序列长度
        """
        self.block_size = block_size

        if data_path.endswith('.bin'):
            # 已编码的数据
            self.data = torch.from_numpy(
                __import__('numpy').memmap(data_path, dtype=__import__('numpy').uint16, mode='r')
            ).long()
        else:
            # 纯文本数据，使用字符级编码
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

    def __getitem__(self, idx):
        # 输入: [idx, idx+block_size)
        # 目标: [idx+1, idx+block_size+1)
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def create_sample_data(output_path: str = "sample_data.txt"):
    """创建示例数据（莎士比亚风格）"""

    # 简单的示例文本
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

ROMEO: Shall I hear more, or shall I speak at this?

JULIET: 'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? it is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? that which we call a rose
By any other name would smell as sweet.

HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd.

MACBETH: Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day
To the last syllable of recorded time,
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more: it is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.
""" * 100  # 重复以增加数据量

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)

    print(f"示例数据已保存到: {output_path}")
    print(f"数据大小: {len(sample_text)} 字符")


def demo_dataset():
    """演示数据集"""
    print("=" * 60)
    print("数据集演示")
    print("=" * 60)

    # 创建示例数据
    data_path = "sample_data.txt"
    if not os.path.exists(data_path):
        create_sample_data(data_path)

    # 加载数据集
    dataset = PretrainDataset(data_path, block_size=64)

    print(f"\n数据集大小: {len(dataset)} 个样本")

    # 查看一个样本
    x, y = dataset[0]
    print("\n样本 0:")
    print(f"  输入 shape: {x.shape}")
    print(f"  目标 shape: {y.shape}")

    # 解码查看
    if hasattr(dataset, 'idx_to_char'):
        input_text = ''.join([dataset.idx_to_char[i.item()] for i in x[:50]])
        target_text = ''.join([dataset.idx_to_char[i.item()] for i in y[:50]])
        print(f"\n  输入文本: {repr(input_text)}")
        print(f"  目标文本: {repr(target_text)}")

    # 创建 DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print("\n批次 shape:")
    print(f"  输入: {batch_x.shape}")
    print(f"  目标: {batch_y.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据处理")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    parser.add_argument("--demo", action="store_true", help="演示数据集")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_data()
    elif args.demo or True:  # 默认演示
        demo_dataset()
