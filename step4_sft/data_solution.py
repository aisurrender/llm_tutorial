"""
Step 4: SFT 数据处理

本文件实现 SFT 数据的加载和预处理。
关键：只计算 Assistant 部分的 Loss。

运行: python data.py --create_sample
"""

import json
import torch
from torch.utils.data import Dataset


# =============================================================================
# 特殊 Token 定义
# =============================================================================

SPECIAL_TOKENS = {
    'bos': '<s>',
    'eos': '</s>',
    'pad': '<pad>',
    'system': '<|system|>',
    'user': '<|user|>',
    'assistant': '<|assistant|>',
}


# =============================================================================
# SFT 数据集
# =============================================================================

class SFTDataset(Dataset):
    """
    SFT 数据集

    数据格式（JSONL）:
    {"conversations": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

    关键：只计算 Assistant 部分的 Loss
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        Args:
            data_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item['conversations'])

        print(f"加载了 {len(self.data)} 条对话数据")

    def _format_conversation(self, conversations: list) -> tuple:
        """
        将对话格式化为模型输入

        Returns:
            input_ids: Token IDs
            labels: 标签（非 Assistant 部分为 -100）
        """
        input_ids = []
        labels = []

        for turn in conversations:
            role = turn['role']
            content = turn['content']

            # 添加角色标记
            if role == 'system':
                role_tokens = self.tokenizer.encode(SPECIAL_TOKENS['system'])
            elif role == 'user':
                role_tokens = self.tokenizer.encode(SPECIAL_TOKENS['user'])
            else:  # assistant
                role_tokens = self.tokenizer.encode(SPECIAL_TOKENS['assistant'])

            content_tokens = self.tokenizer.encode(content)
            eos_token = self.tokenizer.encode(SPECIAL_TOKENS['eos'])

            turn_tokens = role_tokens + content_tokens + eos_token

            input_ids.extend(turn_tokens)

            # 关键：只有 Assistant 的回复计算 Loss
            if role == 'assistant':
                # 角色标记不计算 loss，内容计算 loss
                labels.extend([-100] * len(role_tokens))
                labels.extend(content_tokens)
                labels.extend(eos_token)
            else:
                labels.extend([-100] * len(turn_tokens))

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return input_ids, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversations = self.data[idx]
        input_ids, labels = self._format_conversation(conversations)

        # 转换为 tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels


def collate_fn(batch, pad_token_id=0):
    """动态 padding"""
    input_ids_list, labels_list = zip(*batch)

    # 找最长序列
    max_len = max(len(ids) for ids in input_ids_list)

    # Padding
    padded_inputs = []
    padded_labels = []

    for input_ids, labels in zip(input_ids_list, labels_list):
        pad_len = max_len - len(input_ids)
        padded_inputs.append(torch.cat([input_ids, torch.full((pad_len,), pad_token_id)]))
        padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100)]))

    return torch.stack(padded_inputs), torch.stack(padded_labels)


# =============================================================================
# 简单的字符级 Tokenizer（用于演示）
# =============================================================================

class SimpleTokenizer:
    """简单的字符级分词器（用于演示）"""

    def __init__(self):
        # 基础字符
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars += list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \n\t")
        chars += list("你我他她它们的是了在有不这个上大中小国人年月日时分秒")
        chars += list("什么为什怎么样吗呢吧啊哦嗯好很")
        chars += list(SPECIAL_TOKENS.values())

        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list:
        return [self.char_to_idx.get(c, 1) for c in text]  # 1 = unk

    def decode(self, ids: list) -> str:
        return ''.join([self.idx_to_char.get(i, '?') for i in ids])


# =============================================================================
# 创建示例数据
# =============================================================================

def create_sample_data(output_path: str = "sft_data.jsonl"):
    """创建示例 SFT 数据"""

    samples = [
        {
            "conversations": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a branch of AI that enables computers to learn from data."}
            ]
        },
        {
            "conversations": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I am doing well, thank you for asking!"}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why do programmers prefer dark mode? Because light attracts bugs!"}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a popular programming language known for its simplicity."}
            ]
        },
    ] * 20  # 重复以增加数据量

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"示例数据已保存到: {output_path}")
    print(f"数据量: {len(samples)} 条对话")


def demo_dataset():
    """演示数据集"""
    import os

    print("=" * 60)
    print("SFT 数据集演示")
    print("=" * 60)

    # 创建示例数据
    data_path = "sft_data.jsonl"
    if not os.path.exists(data_path):
        create_sample_data(data_path)

    # 创建分词器和数据集
    tokenizer = SimpleTokenizer()
    dataset = SFTDataset(data_path, tokenizer, max_length=256)

    print(f"\n数据集大小: {len(dataset)} 条对话")
    print(f"词表大小: {tokenizer.vocab_size}")

    # 查看一个样本
    input_ids, labels = dataset[0]
    print(f"\n样本 0:")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Labels shape: {labels.shape}")

    # 统计有多少 token 计算 loss
    loss_tokens = (labels != -100).sum().item()
    total_tokens = len(labels)
    print(f"  计算 Loss 的 token 数: {loss_tokens}/{total_tokens} ({100*loss_tokens/total_tokens:.1f}%)")

    # 解码查看
    print(f"\n  解码后的输入:")
    print(f"  {tokenizer.decode(input_ids.tolist())[:200]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT 数据处理")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    parser.add_argument("--demo", action="store_true", help="演示数据集")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_data()
    elif args.demo or True:
        demo_dataset()
