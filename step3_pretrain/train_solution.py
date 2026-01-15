"""
Step 3: 预训练脚本

本文件实现完整的预训练流程，约 150 行核心代码。

运行:
    python train.py --device cpu --epochs 5    # CPU 训练
    python train.py --device mps --epochs 5    # Mac M 系列
    python train.py --device cuda --epochs 5   # NVIDIA GPU
"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_gpt_model.model import GPT, GPTConfig

from data import PretrainDataset, create_sample_data

# =============================================================================
# 学习率调度
# =============================================================================

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """
    学习率调度：Warmup + Cosine Decay

    1. Warmup 阶段：线性增加
    2. Decay 阶段：余弦衰减

    Args:
        step: 当前步数
        warmup_steps: 预热步数
        total_steps: 总步数
        max_lr: 最大学习率
        min_lr: 最小学习率
    """
    # Warmup 阶段
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Decay 阶段
    if step >= total_steps:
        return min_lr

    # Cosine decay
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# 训练函数
# =============================================================================

def train(args):
    """主训练函数"""

    # 1. 设置设备
    device = args.device
    device_type = "cuda" if "cuda" in device else ("mps" if device == "mps" else "cpu")

    print(f"使用设备: {device}")

    # 2. 准备数据
    data_path = args.data_path
    if not os.path.exists(data_path):
        print("数据文件不存在，创建示例数据...")
        create_sample_data(data_path)

    dataset = PretrainDataset(data_path, block_size=args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3. 创建模型
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # 5. 混合精度
    if device_type == "cuda":
        ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        ctx = nullcontext()
        scaler = None

    # 6. 训练循环
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    global_step = 0

    print("\n开始训练:")
    print(f"  总步数: {total_steps}")
    print(f"  Warmup 步数: {warmup_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Block size: {args.block_size}")
    print()

    model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # 更新学习率
            lr = get_lr(global_step, warmup_steps, total_steps, args.learning_rate, args.learning_rate * 0.1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播
            with ctx:
                logits, loss = model(x, targets=y)

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

            # 打印日志
            if batch_idx % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{args.epochs} | Step {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")

        avg_loss = epoch_loss / num_batches
        print(f"\n>>> Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}\n")

        # 保存 checkpoint
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch+1}.pt")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'vocab': dataset.char_to_idx if hasattr(dataset, 'char_to_idx') else None,
            }, ckpt_path)
            print(f"模型已保存到: {ckpt_path}")

    print("\n训练完成!")
    return model, dataset


# =============================================================================
# 生成函数
# =============================================================================

@torch.no_grad()
def generate(args):
    """加载模型并生成文本"""

    # 加载 checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint 不存在: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=args.device)
    config = ckpt['config']
    vocab = ckpt['vocab']

    # 创建模型
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 编码 prompt
    idx_to_char = {v: k for k, v in vocab.items()}
    prompt_ids = [vocab.get(c, 0) for c in args.prompt]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    # 生成
    print(f"\nPrompt: {args.prompt}")
    print("生成中...")

    y = model.generate(x, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)

    # 解码
    generated = ''.join([idx_to_char.get(i.item(), '?') for i in y[0]])
    print(f"\n生成结果:\n{generated}")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT 预训练")

    # 模式
    parser.add_argument("--mode", choices=["train", "generate"], default="train")

    # 数据
    parser.add_argument("--data_path", type=str, default="sample_data.txt")
    parser.add_argument("--block_size", type=int, default=128)

    # 模型
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    # 生成
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_epoch3.pt")
    parser.add_argument("--prompt", type=str, default="To be or not to be")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        generate(args)

    print("\n" + "=" * 60)
    print("下一步: 进入 step4_sft/ 学习指令微调")
    print("=" * 60)
