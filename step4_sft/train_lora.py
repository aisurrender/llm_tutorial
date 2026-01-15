"""
Step 4: LoRA 微调脚本

LoRA (Low-Rank Adaptation) 是一种高效微调方法：
- 冻结原始模型参数
- 只训练低秩分解的旁路矩阵
- 大幅减少训练参数量

运行:
    python train_lora.py --device cuda --epochs 3 --lora_r 8
"""

import os
import sys
import math
import time
import argparse
from functools import partial
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_gpt_model.model import GPT, GPTConfig
from data import SFTDataset, SimpleTokenizer, collate_fn, create_sample_data


# =============================================================================
# LoRA 实现
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA 线性层

    原始: y = Wx
    LoRA: y = Wx + (B @ A)x = Wx + BAx

    其中:
    - W: 原始权重 [out_features, in_features]（冻结）
    - A: 低秩矩阵 [r, in_features]
    - B: 低秩矩阵 [out_features, r]
    - r: 秩（远小于 in_features 和 out_features）

    参数量对比:
    - 原始: out_features × in_features
    - LoRA: (out_features + in_features) × r
    """

    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: float = 16, dropout: float = 0.05):
        """
        Args:
            original_linear: 原始的 nn.Linear 层
            r: LoRA 的秩
            alpha: 缩放系数
            dropout: Dropout 概率
        """
        super().__init__()

        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 冻结原始权重
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # 初始时 LoRA 输出为 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_output = self.original_linear(x)

        # LoRA 输出: (B @ A) @ x * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        return original_output + lora_output


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16, target_modules: list = None):
    """
    将 LoRA 应用到模型的指定层

    Args:
        model: 原始模型
        r: LoRA 秩
        alpha: 缩放系数
        target_modules: 要应用 LoRA 的模块名（如 ['c_attn', 'c_proj']）
    """
    if target_modules is None:
        target_modules = ['c_attn', 'c_proj']  # 默认应用到 attention 层

    lora_params = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是目标模块
            if any(target in name for target in target_modules):
                # 创建 LoRA 层
                lora_linear = LoRALinear(module, r=r, alpha=alpha)

                # 替换原始层
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, child_name, lora_linear)

                # 收集 LoRA 参数
                lora_params.extend([lora_linear.lora_A, lora_linear.lora_B])
                print(f"  应用 LoRA 到: {name}")

    # 返回 LoRA 参数
    return lora_params


def count_parameters(model: nn.Module) -> tuple:
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# 训练
# =============================================================================

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(args):
    """LoRA 训练"""

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

    # 2. 创建模型
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.max_length,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=0.0,  # LoRA 时关闭原始 dropout
    )
    model = GPT(config).to(device)

    # 3. 应用 LoRA
    print(f"\n应用 LoRA (r={args.lora_r}, alpha={args.lora_alpha}):")
    lora_params = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    total, trainable = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  总参数量: {total/1e6:.2f}M")
    print(f"  可训练参数量: {trainable/1e6:.4f}M ({100*trainable/total:.2f}%)")

    # 4. 优化器（只优化 LoRA 参数）
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=0.01)

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

    print(f"\n开始 LoRA 训练:")
    print(f"  数据量: {len(dataset)} 条对话")
    print(f"  总步数: {total_steps}")
    print()

    model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            lr = get_lr(global_step, warmup_steps, total_steps, args.learning_rate, args.learning_rate * 0.1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                logits, loss = model(input_ids, targets=labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
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
    save_path = os.path.join(args.save_dir, "lora_model.pt")

    # 只保存 LoRA 参数
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data

    torch.save({
        'lora_state_dict': lora_state_dict,
        'config': config,
        'lora_config': {'r': args.lora_r, 'alpha': args.lora_alpha},
    }, save_path)
    print(f"LoRA 参数已保存到: {save_path}")

    print("\nLoRA 训练完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 微调")

    # 数据
    parser.add_argument("--data_path", type=str, default="sft_data.jsonl")
    parser.add_argument("--max_length", type=int, default=256)

    # 模型
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=6)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA 缩放系数")

    # 训练
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # LoRA 可以用更大的学习率
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)
