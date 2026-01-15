"""
Step 5: DPO 训练脚本

DPO (Direct Preference Optimization) 直接从偏好数据学习，无需 Reward Model。

运行:
    python train_dpo.py --device cuda --epochs 1 --beta 0.1
"""

import os
import sys
import copy
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2_gpt_model.model import GPT, GPTConfig
from data import DPODataset, SimpleTokenizer, collate_fn, create_sample_data


# =============================================================================
# DPO 核心函数
# =============================================================================

def get_log_probs(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算每个 token 的 log probability

    Args:
        model: 语言模型
        input_ids: 输入 token IDs [batch, seq_len]
        labels: 标签（用于取对应位置的 log prob）[batch, seq_len]

    Returns:
        log_probs: 每个位置的 log probability [batch, seq_len]
    """
    logits, _ = model(input_ids)  # [batch, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

    # 取出对应 label 的 log prob
    # labels 需要右移一位（因为 logits[i] 预测的是 token[i+1]）
    labels_shifted = labels[:, 1:].unsqueeze(-1)  # [batch, seq_len-1, 1]
    log_probs_shifted = log_probs[:, :-1, :]  # [batch, seq_len-1, vocab_size]

    token_log_probs = torch.gather(log_probs_shifted, dim=-1, index=labels_shifted).squeeze(-1)
    # [batch, seq_len-1]

    # 补齐第一个位置（设为 0）
    token_log_probs = F.pad(token_log_probs, (1, 0), value=0)  # [batch, seq_len]

    return token_log_probs


def dpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    ref_chosen_log_probs: torch.Tensor,
    ref_rejected_log_probs: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    计算 DPO Loss

    L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

    Args:
        policy_chosen_log_probs: 策略模型对 chosen 的 log prob
        policy_rejected_log_probs: 策略模型对 rejected 的 log prob
        ref_chosen_log_probs: 参考模型对 chosen 的 log prob
        ref_rejected_log_probs: 参考模型对 rejected 的 log prob
        chosen_mask: chosen 的 mask（只对 response 部分计算）
        rejected_mask: rejected 的 mask
        beta: 温度系数

    Returns:
        loss: DPO loss
    """
    # 对 response 部分求和（按 mask）
    def masked_mean(log_probs, mask):
        masked = log_probs * mask
        return masked.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)

    # 计算每个样本的平均 log prob
    policy_chosen = masked_mean(policy_chosen_log_probs, chosen_mask)
    policy_rejected = masked_mean(policy_rejected_log_probs, rejected_mask)
    ref_chosen = masked_mean(ref_chosen_log_probs, chosen_mask)
    ref_rejected = masked_mean(ref_rejected_log_probs, rejected_mask)

    # DPO loss
    # log(π/π_ref) = log π - log π_ref
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected

    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    return loss


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
    """DPO 训练"""

    device = args.device
    device_type = "cuda" if "cuda" in device else ("mps" if device == "mps" else "cpu")
    print(f"使用设备: {device}")

    # 1. 准备数据
    data_path = args.data_path
    if not os.path.exists(data_path):
        print("数据文件不存在，创建示例数据...")
        create_sample_data(data_path)

    tokenizer = SimpleTokenizer()
    dataset = DPODataset(data_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 2. 创建策略模型
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.max_length,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=0.0,
    )
    policy_model = GPT(config).to(device)

    # 3. 创建参考模型（冻结）
    ref_model = GPT(config).to(device)
    ref_model.load_state_dict(policy_model.state_dict())  # 复制权重
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"策略模型参数量: {sum(p.numel() for p in policy_model.parameters())/1e6:.2f}M")
    print(f"参考模型参数量: {sum(p.numel() for p in ref_model.parameters())/1e6:.2f}M (冻结)")

    # 4. 优化器（注意：DPO 学习率要很小）
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate, weight_decay=0.01)

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

    print(f"\n开始 DPO 训练:")
    print(f"  数据量: {len(dataset)} 条偏好对")
    print(f"  总步数: {total_steps}")
    print(f"  Beta: {args.beta}")
    print()

    policy_model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            chosen_ids = batch['chosen_ids'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            chosen_mask = batch['chosen_mask'].to(device)
            rejected_mask = batch['rejected_mask'].to(device)

            # 更新学习率
            lr = get_lr(global_step, warmup_steps, total_steps, args.learning_rate, args.learning_rate * 0.1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                # 计算参考模型的 log probs（不需要梯度）
                with torch.no_grad():
                    ref_chosen_log_probs = get_log_probs(ref_model, chosen_ids, chosen_ids)
                    ref_rejected_log_probs = get_log_probs(ref_model, rejected_ids, rejected_ids)

                # 计算策略模型的 log probs
                policy_chosen_log_probs = get_log_probs(policy_model, chosen_ids, chosen_ids)
                policy_rejected_log_probs = get_log_probs(policy_model, rejected_ids, rejected_ids)

                # 计算 DPO loss
                loss = dpo_loss(
                    policy_chosen_log_probs,
                    policy_rejected_log_probs,
                    ref_chosen_log_probs,
                    ref_rejected_log_probs,
                    chosen_mask,
                    rejected_mask,
                    beta=args.beta
                )

            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if batch_idx % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{args.epochs} | Step {batch_idx}/{len(dataloader)} | "
                      f"DPO Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")

        avg_loss = epoch_loss / num_batches
        print(f"\n>>> Epoch {epoch+1} 完成, 平均 DPO Loss: {avg_loss:.4f}\n")

    # 7. 保存
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "dpo_model.pt")
    torch.save({
        'model': policy_model.state_dict(),
        'config': config,
    }, save_path)
    print(f"模型已保存到: {save_path}")

    print("\nDPO 训练完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO 训练")

    # 数据
    parser.add_argument("--data_path", type=str, default="dpo_data.jsonl")
    parser.add_argument("--max_length", type=int, default=256)

    # 模型
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=6)

    # DPO
    parser.add_argument("--beta", type=float, default=0.1, help="DPO 温度系数")

    # 训练
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)  # DPO 用很小的学习率
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)
