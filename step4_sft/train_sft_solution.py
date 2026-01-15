"""
Step 4: SFT 训练脚本

本文件实现监督微调（Supervised Fine-Tuning）。

运行:
    python train_sft.py --device cuda --epochs 3
"""

import os
import sys
import math
import time
import argparse
from functools import partial
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_gpt_model.model import GPT, GPTConfig
from data import SFTDataset, SimpleTokenizer, collate_fn, create_sample_data


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """学习率调度：Warmup + Cosine Decay"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(args):
    """SFT 训练"""

    device = args.device
    device_type = "cuda" if "cuda" in device else ("mps" if device == "mps" else "cpu")
    print(f"使用设备: {device}")

    # 1. 准备数据
    data_path = args.data_path
    if not os.path.exists(data_path):
        print("数据文件不存在，创建示例数据...")
        create_sample_data(data_path)

    tokenizer = SimpleTokenizer()
    dataset = SFTDataset(data_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=0)
    )

    # 2. 创建/加载模型
    if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        print(f"加载预训练模型: {args.pretrain_checkpoint}")
        ckpt = torch.load(args.pretrain_checkpoint, map_location=device)
        config = ckpt['config']
        model = GPT(config).to(device)
        model.load_state_dict(ckpt['model'])
    else:
        print("从头创建模型...")
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.max_length,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
        )
        model = GPT(config).to(device)

    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # 4. 混合精度
    if device_type == "cuda":
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        ctx = nullcontext()
        scaler = None

    # 5. 训练循环
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    global_step = 0

    print(f"\n开始 SFT 训练:")
    print(f"  数据量: {len(dataset)} 条对话")
    print(f"  总步数: {total_steps}")
    print(f"  Batch size: {args.batch_size}")
    print()

    model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 更新学习率
            lr = get_lr(global_step, warmup_steps, total_steps, args.learning_rate, args.learning_rate * 0.1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播
            with ctx:
                logits, loss = model(input_ids, targets=labels)

            # 反向传播
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

    # 6. 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "sft_model.pt")
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'tokenizer_vocab': tokenizer.char_to_idx,
    }, save_path)
    print(f"模型已保存到: {save_path}")

    print("\nSFT 训练完成!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT 训练")

    # 数据
    parser.add_argument("--data_path", type=str, default="sft_data.jsonl")
    parser.add_argument("--max_length", type=int, default=256)

    # 模型
    parser.add_argument("--pretrain_checkpoint", type=str, default=None)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)
