"""
Step 3: 预训练脚本

练习文件：请完成标记为 TODO 的部分

运行:
    python train_exercise.py --device cpu --epochs 3
"""

import os
import sys
import math
import time
import argparse
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_gpt_model.model_solution import GPT, GPTConfig
from data_exercise import PretrainDataset, create_sample_data


# =============================================================================
# TODO 2: 学习率调度
# =============================================================================

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """
    学习率调度：Warmup + Cosine Decay

    训练过程中学习率的变化：
    1. Warmup 阶段（step < warmup_steps）：
       - 从 0 线性增加到 max_lr
       - 公式: lr = max_lr * (step + 1) / warmup_steps

    2. Decay 阶段（step >= warmup_steps）：
       - 按余弦曲线从 max_lr 衰减到 min_lr
       - 公式: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
       - progress = (step - warmup_steps) / (total_steps - warmup_steps)

    为什么需要 Warmup？
    - 训练初期，模型参数是随机的，梯度可能很大
    - 如果一开始就用大学习率，容易发散
    - Warmup 让模型"热身"，逐渐适应

    为什么需要 Cosine Decay？
    - 训练后期需要更小的学习率来精细调整
    - 余弦衰减比线性衰减更平滑

    学习率曲线:
        lr
        ↑
        |   /\\
        |  /  \\
        | /    \\___________
        |/
        +------------------→ step
          warmup   decay

    Args:
        step: 当前步数
        warmup_steps: 预热步数
        total_steps: 总步数
        max_lr: 最大学习率
        min_lr: 最小学习率

    Returns:
        当前步的学习率
    """
    # TODO 2a: Warmup 阶段
    if step < warmup_steps:
        # 线性增加: 0 -> max_lr
        # 公式: lr = max_lr * (step + 1) / warmup_steps
        # return ...
        raise NotImplementedError("请实现 warmup 阶段的学习率计算")

    # 训练结束后返回最小学习率
    if step >= total_steps:
        return min_lr

    # TODO 2b: Cosine Decay 阶段
    # 1. 计算 decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    # 2. 计算 coeff = 0.5 * (1 + cos(pi * decay_ratio))
    # 3. 返回 min_lr + coeff * (max_lr - min_lr)
    #
    # 提示: 使用 math.cos 和 math.pi
    # decay_ratio = ...
    # coeff = ...
    # return min_lr + coeff * (max_lr - min_lr)
    raise NotImplementedError("请实现 cosine decay 阶段的学习率计算")


# =============================================================================
# TODO 3: 训练循环
# =============================================================================

def train(args):
    """主训练函数"""

    # 1. 设置设备
    device = args.device
    print(f"使用设备: {device}")

    # 2. 准备数据
    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"数据文件不存在，创建示例数据...")
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

    # 5. 训练设置
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    global_step = 0

    print(f"\n开始训练:")
    print(f"  总步数: {total_steps}")
    print(f"  Warmup 步数: {warmup_steps}")
    print(f"  Batch size: {args.batch_size}")
    print()

    model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # ================================================================
            # TODO 3a: 更新学习率
            # ================================================================
            # 提示:
            # lr = get_lr(global_step, warmup_steps, total_steps, args.learning_rate, args.learning_rate * 0.1)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            try:
                lr = get_lr(global_step, warmup_steps, total_steps, args.learning_rate, args.learning_rate * 0.1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            except NotImplementedError:
                lr = args.learning_rate  # 使用固定学习率作为后备

            # ================================================================
            # TODO 3b: 前向传播
            # ================================================================
            # 提示: logits, loss = model(x, targets=y)
            # logits, loss = ...
            raise NotImplementedError("请实现前向传播")

            # ================================================================
            # TODO 3c: 反向传播 + 梯度裁剪 + 优化器更新
            # ================================================================
            # 步骤:
            # 1. loss.backward()  # 计算梯度
            # 2. torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            # 3. optimizer.step()  # 更新参数
            # 4. optimizer.zero_grad(set_to_none=True)  # 清空梯度
            raise NotImplementedError("请实现反向传播和优化器更新")

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

    print("\n训练完成!")
    return model, dataset


# =============================================================================
# 测试代码
# =============================================================================

def test_lr_schedule():
    """测试学习率调度"""
    print("测试学习率调度...")

    warmup_steps = 100
    total_steps = 1000
    max_lr = 1e-3
    min_lr = 1e-4

    try:
        # 测试 warmup 阶段
        lr_0 = get_lr(0, warmup_steps, total_steps, max_lr, min_lr)
        lr_50 = get_lr(50, warmup_steps, total_steps, max_lr, min_lr)
        lr_99 = get_lr(99, warmup_steps, total_steps, max_lr, min_lr)

        # Warmup 阶段应该线性增加
        assert lr_0 < lr_50 < lr_99, "Warmup 阶段学习率应该递增"
        assert abs(lr_99 - max_lr) < 0.1 * max_lr, "Warmup 结束时应接近 max_lr"

        # 测试 decay 阶段
        lr_100 = get_lr(100, warmup_steps, total_steps, max_lr, min_lr)
        lr_500 = get_lr(500, warmup_steps, total_steps, max_lr, min_lr)
        lr_999 = get_lr(999, warmup_steps, total_steps, max_lr, min_lr)

        # Decay 阶段应该递减
        assert lr_100 > lr_500 > lr_999, "Decay 阶段学习率应该递减"
        assert abs(lr_999 - min_lr) < 0.1 * min_lr, "Decay 结束时应接近 min_lr"

        print("✅ 学习率调度测试通过!")

        # 可视化
        print("\n学习率曲线（采样）:")
        for step in [0, 50, 100, 200, 500, 800, 999]:
            lr = get_lr(step, warmup_steps, total_steps, max_lr, min_lr)
            bar = "█" * int(lr / max_lr * 20)
            print(f"  Step {step:4d}: {lr:.2e} {bar}")

        return True
    except NotImplementedError as e:
        print(f"⚠️ {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT 预训练")

    # 测试模式
    parser.add_argument("--test", action="store_true", help="运行测试")

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

    args = parser.parse_args()

    if args.test:
        print("=" * 60)
        print("预训练测试")
        print("=" * 60)
        print()
        test_lr_schedule()
    else:
        train(args)
