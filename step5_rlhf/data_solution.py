"""
Step 5: DPO 数据处理

DPO 数据格式:
{
    "prompt": "用户问题",
    "chosen": "好的回答（被偏好）",
    "rejected": "差的回答（不被偏好）"
}

运行: python data.py --create_sample
"""

import json

import torch
from torch.utils.data import Dataset


class DPODataset(Dataset):
    """
    DPO 数据集

    每个样本包含:
    - prompt: 用户输入
    - chosen: 被偏好的回答
    - rejected: 不被偏好的回答
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"加载了 {len(self.data)} 条偏好数据")

    def _tokenize(self, prompt: str, response: str):
        """将 prompt + response 编码"""
        text = f"<|user|>{prompt}<|assistant|>{response}"
        input_ids = self.tokenizer.encode(text)

        # 创建 mask：只对 response 部分计算 loss
        prompt_text = f"<|user|>{prompt}<|assistant|>"
        prompt_len = len(self.tokenizer.encode(prompt_text))

        mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            mask = mask[:self.max_length]

        return input_ids, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 编码 chosen 和 rejected
        chosen_ids, chosen_mask = self._tokenize(item['prompt'], item['chosen'])
        rejected_ids, rejected_mask = self._tokenize(item['prompt'], item['rejected'])

        return {
            'chosen_ids': chosen_ids,
            'chosen_mask': chosen_mask,
            'rejected_ids': rejected_ids,
            'rejected_mask': rejected_mask,
        }


def collate_fn(batch, pad_token_id=0):
    """动态 padding"""
    # 找最长序列
    max_chosen_len = max(len(item['chosen_ids']) for item in batch)
    max_rejected_len = max(len(item['rejected_ids']) for item in batch)
    max_len = max(max_chosen_len, max_rejected_len)

    chosen_ids_list = []
    chosen_mask_list = []
    rejected_ids_list = []
    rejected_mask_list = []

    for item in batch:
        # Padding chosen
        pad_len = max_len - len(item['chosen_ids'])
        chosen_ids_list.append(item['chosen_ids'] + [pad_token_id] * pad_len)
        chosen_mask_list.append(item['chosen_mask'] + [0] * pad_len)

        # Padding rejected
        pad_len = max_len - len(item['rejected_ids'])
        rejected_ids_list.append(item['rejected_ids'] + [pad_token_id] * pad_len)
        rejected_mask_list.append(item['rejected_mask'] + [0] * pad_len)

    return {
        'chosen_ids': torch.tensor(chosen_ids_list, dtype=torch.long),
        'chosen_mask': torch.tensor(chosen_mask_list, dtype=torch.float),
        'rejected_ids': torch.tensor(rejected_ids_list, dtype=torch.long),
        'rejected_mask': torch.tensor(rejected_mask_list, dtype=torch.float),
    }


# =============================================================================
# 简单分词器
# =============================================================================

class SimpleTokenizer:
    """简单的字符级分词器"""

    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars += list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \n\t")
        chars += ["<|user|>", "<|assistant|>", "<|system|>", "<s>", "</s>", "<pad>"]

        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list:
        # 处理特殊 token
        for special in ["<|user|>", "<|assistant|>", "<|system|>", "<s>", "</s>", "<pad>"]:
            text = text.replace(special, f"\x00{special}\x00")

        result = []
        for part in text.split("\x00"):
            if part in self.char_to_idx:
                result.append(self.char_to_idx[part])
            else:
                for c in part:
                    result.append(self.char_to_idx.get(c, 1))
        return result

    def decode(self, ids: list) -> str:
        return ''.join([self.idx_to_char.get(i, '?') for i in ids])


# =============================================================================
# 创建示例数据
# =============================================================================

def create_sample_data(output_path: str = "dpo_data.jsonl"):
    """创建示例 DPO 数据"""

    samples = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris. Paris is known for the Eiffel Tower and its rich history.",
            "rejected": "France capital is paris."
        },
        {
            "prompt": "How do I learn programming?",
            "chosen": "To learn programming, start with fundamentals like variables and loops. Practice regularly with small projects. Python is a great beginner language.",
            "rejected": "Just read some books."
        },
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of AI that enables computers to learn patterns from data without explicit programming. It includes techniques like neural networks and decision trees.",
            "rejected": "Its like robots."
        },
        {
            "prompt": "Tell me about Python.",
            "chosen": "Python is a high-level programming language known for its readability and versatility. It is widely used in web development, data science, and automation.",
            "rejected": "Python is a snake."
        },
        {
            "prompt": "How to stay healthy?",
            "chosen": "To stay healthy, maintain a balanced diet, exercise regularly, get enough sleep, and manage stress. Regular health check-ups are also important.",
            "rejected": "Eat food."
        },
    ] * 20

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"示例数据已保存到: {output_path}")
    print(f"数据量: {len(samples)} 条偏好对")


def demo_dataset():
    """演示数据集"""
    import os

    print("=" * 60)
    print("DPO 数据集演示")
    print("=" * 60)

    data_path = "dpo_data.jsonl"
    if not os.path.exists(data_path):
        create_sample_data(data_path)

    tokenizer = SimpleTokenizer()
    dataset = DPODataset(data_path, tokenizer, max_length=256)

    print(f"\n数据集大小: {len(dataset)} 条偏好对")

    # 查看一个样本
    sample = dataset[0]
    print("\n样本 0:")
    print(f"  Chosen IDs 长度: {len(sample['chosen_ids'])}")
    print(f"  Rejected IDs 长度: {len(sample['rejected_ids'])}")
    print(f"  Chosen mask sum: {sum(sample['chosen_mask'])} (计算 loss 的 token 数)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DPO 数据处理")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_data()
    else:
        demo_dataset()
