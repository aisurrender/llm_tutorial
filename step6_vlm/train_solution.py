"""
Step 6: VLM 训练脚本

两阶段训练:
1. 预训练: 学习图像描述（冻结 Vision Encoder）
2. SFT: 学习指令跟随

运行:
    python train.py --device cuda --epochs 3
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import VLM, VLMConfig

# =============================================================================
# 数据集
# =============================================================================

class VLMDataset(Dataset):
    """
    VLM 数据集

    数据格式 (JSONL):
    {"image": "path/to/image.jpg", "text": "图片描述文本"}
    """

    def __init__(self, data_path: str, tokenizer, image_size: int = 224, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 图像预处理
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载数据
        if os.path.exists(data_path):
            with open(data_path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))

        print(f"加载了 {len(self.data)} 条图文数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载图片（如果存在）
        if 'image' in item and os.path.exists(item['image']):
            image = Image.open(item['image']).convert('RGB')
            image = self.transform(image)
        else:
            # 使用随机图片（演示用）
            image = torch.randn(3, 224, 224)

        # 编码文本
        text = item.get('text', '')
        input_ids = self.tokenizer.encode(text)[:self.max_length]
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 目标（右移一位）
        labels = input_ids.clone()

        return {
            'image': image,
            'input_ids': input_ids,
            'labels': labels,
        }


def collate_fn(batch, pad_token_id=0):
    """动态 padding"""
    images = torch.stack([item['image'] for item in batch])

    max_len = max(len(item['input_ids']) for item in batch)

    input_ids_list = []
    labels_list = []

    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids_list.append(torch.cat([item['input_ids'], torch.full((pad_len,), pad_token_id)]))
        labels_list.append(torch.cat([item['labels'], torch.full((pad_len,), -100)]))

    return {
        'images': images,
        'input_ids': torch.stack(input_ids_list),
        'labels': torch.stack(labels_list),
    }


# =============================================================================
# 简单分词器
# =============================================================================

class SimpleTokenizer:
    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars += list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \n\t")
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list:
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, ids: list) -> str:
        return ''.join([self.idx_to_char.get(i, '?') for i in ids])


# =============================================================================
# 创建示例数据
# =============================================================================

def create_sample_data(output_path: str = "vlm_data.jsonl"):
    """创建示例 VLM 数据"""
    samples = [
        {"text": "A beautiful sunset over the ocean with orange and pink colors in the sky."},
        {"text": "A cute cat sitting on a windowsill looking outside."},
        {"text": "A busy city street with tall buildings and many people walking."},
        {"text": "A delicious pizza with pepperoni and melted cheese."},
        {"text": "A peaceful mountain landscape with snow-capped peaks."},
    ] * 20

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"示例数据已保存到: {output_path}")


# =============================================================================
# 训练
# =============================================================================

def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(args):
    """VLM 训练"""
    device = args.device
    device_type = "cuda" if "cuda" in device else ("mps" if device == "mps" else "cpu")
    print(f"使用设备: {device}")

    # 1. 准备数据
    data_path = args.data_path
    if not os.path.exists(data_path):
        print("数据文件不存在，创建示例数据...")
        create_sample_data(data_path)

    tokenizer = SimpleTokenizer()
    dataset = VLMDataset(data_path, tokenizer, image_size=args.image_size, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 2. 创建模型
    config = VLMConfig(
        vision_dim=args.vision_dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_patches=(args.image_size // args.patch_size) ** 2,
        vocab_size=tokenizer.vocab_size,
        block_size=args.max_length + (args.image_size // args.patch_size) ** 2 + 2,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = VLM(config).to(device)

    # 3. 冻结 Vision Encoder（可选）
    if args.freeze_vision:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        print("Vision Encoder 已冻结")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params/1e6:.2f}M")

    # 4. 优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # 5. 混合精度
    if device_type == "cuda":
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        ctx = nullcontext()
        scaler = None

    # 6. 训练循环
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    global_step = 0

    print("\n开始 VLM 训练:")
    print(f"  数据量: {len(dataset)} 条图文对")
    print(f"  总步数: {total_steps}")
    print()

    model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            lr = get_lr(global_step, warmup_steps, total_steps, args.learning_rate, args.learning_rate * 0.1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                logits, loss = model(images, input_ids, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if batch_idx % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{args.epochs} | Step {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")

        avg_loss = epoch_loss / num_batches
        print(f"\n>>> Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}\n")

    # 7. 保存
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "vlm_model.pt")
    torch.save({
        'model': model.state_dict(),
        'config': config,
    }, save_path)
    print(f"模型已保存到: {save_path}")

    print("\nVLM 训练完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM 训练")

    # 数据
    parser.add_argument("--data_path", type=str, default="vlm_data.jsonl")
    parser.add_argument("--max_length", type=int, default=64)

    # 模型
    parser.add_argument("--vision_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_vision", action="store_true", help="冻结 Vision Encoder")

    # 训练
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)
