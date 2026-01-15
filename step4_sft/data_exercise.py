"""
Step 4: SFT 数据处理

练习文件：请完成标记为 TODO 的部分

关键概念：SFT 只计算 Assistant 部分的 Loss
"""

import json

import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = {
    'bos': '<s>',
    'eos': '</s>',
    'pad': '<pad>',
    'system': '<|system|>',
    'user': '<|user|>',
    'assistant': '<|assistant|>',
}


class SFTDataset(Dataset):
    """
    SFT 数据集

    关键：只计算 Assistant 部分的 Loss！

    为什么？
    - 我们希望模型学会"如何回答"，而不是学会"如何提问"
    - User 的问题是输入，不应该计算 loss
    - 只有 Assistant 的回答需要学习
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item['conversations'])

        print(f"加载了 {len(self.data)} 条对话数据")

    # =========================================================================
    # TODO 1: 实现 _format_conversation 方法
    # =========================================================================
    def _format_conversation(self, conversations: list) -> tuple:
        """
        将对话格式化为模型输入

        关键任务：构建 labels，使得只有 Assistant 部分计算 loss

        输入格式:
            conversations = [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."}
            ]

        输出格式:
            input_ids: [user_tokens..., assistant_tokens...]
            labels:    [-100, -100, ..., assistant_content_tokens...]
                        ↑ 不计算 loss    ↑ 计算 loss

        -100 是 PyTorch CrossEntropyLoss 的 ignore_index

        Returns:
            input_ids: 完整的 token ID 列表
            labels: 标签列表（非 Assistant 部分为 -100）
        """
        input_ids = []
        labels = []

        for turn in conversations:
            role = turn['role']
            content = turn['content']

            # 获取角色标记的 token
            if role == 'system':
                role_tokens = self.tokenizer.encode(SPECIAL_TOKENS['system'])
            elif role == 'user':
                role_tokens = self.tokenizer.encode(SPECIAL_TOKENS['user'])
            else:  # assistant
                role_tokens = self.tokenizer.encode(SPECIAL_TOKENS['assistant'])

            content_tokens = self.tokenizer.encode(content)
            eos_token = self.tokenizer.encode(SPECIAL_TOKENS['eos'])

            turn_tokens = role_tokens + content_tokens + eos_token

            # 添加到 input_ids
            input_ids.extend(turn_tokens)

            # ================================================================
            # TODO: 构建 labels
            # ================================================================
            # 如果是 assistant 的回复：
            #   - 角色标记部分：labels 为 -100（不计算 loss）
            #   - 内容部分：labels 为 content_tokens（计算 loss）
            #   - EOS 部分：labels 为 eos_token（计算 loss）
            # 如果是其他角色（system/user）：
            #   - 全部为 -100（不计算 loss）
            #
            # 提示:
            # if role == 'assistant':
            #     labels.extend([-100] * len(role_tokens))  # 角色标记不计算
            #     labels.extend(content_tokens)             # 内容计算 loss
            #     labels.extend(eos_token)                  # EOS 计算 loss
            # else:
            #     labels.extend([-100] * len(turn_tokens))  # 全部不计算
            raise NotImplementedError("请实现 labels 的构建逻辑")

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
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch, pad_token_id=0):
    """动态 padding（已实现）"""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_inputs = []
    padded_labels = []

    for input_ids, labels in zip(input_ids_list, labels_list):
        pad_len = max_len - len(input_ids)
        padded_inputs.append(torch.cat([input_ids, torch.full((pad_len,), pad_token_id)]))
        padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100)]))

    return torch.stack(padded_inputs), torch.stack(padded_labels)


class SimpleTokenizer:
    """简单的字符级分词器（已实现）"""

    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars += list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \n\t")
        chars += list("你我他她它们的是了在有不这个上大中小国人")
        chars += list(SPECIAL_TOKENS.values())

        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list:
        return [self.char_to_idx.get(c, 1) for c in text]

    def decode(self, ids: list) -> str:
        return ''.join([self.idx_to_char.get(i, '?') for i in ids])


def create_sample_data(output_path: str = "sft_data.jsonl"):
    """创建示例数据"""
    samples = [
        {"conversations": [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."}
        ]},
        {"conversations": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you?"}
        ]},
    ] * 50

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"示例数据已保存到: {output_path}")


def test_sft_dataset():
    """测试 SFT 数据集"""
    import os

    data_path = "sft_data.jsonl"
    if not os.path.exists(data_path):
        create_sample_data(data_path)

    tokenizer = SimpleTokenizer()
    dataset = SFTDataset(data_path, tokenizer, max_length=256)

    try:
        input_ids, labels = dataset[0]

        # 检查只有部分 token 计算 loss
        loss_tokens = (labels != -100).sum().item()
        total_tokens = len(labels)

        assert loss_tokens < total_tokens, "应该只有部分 token 计算 loss"
        assert loss_tokens > 0, "至少应该有一些 token 计算 loss"

        print("✅ SFT 数据集测试通过!")
        print(f"   计算 Loss 的 token: {loss_tokens}/{total_tokens} ({100*loss_tokens/total_tokens:.1f}%)")
        return True
    except NotImplementedError as e:
        print(f"⚠️ {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SFT 数据集测试")
    print("=" * 60)
    print()
    test_sft_dataset()
