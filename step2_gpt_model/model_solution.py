"""
Step 2: GPT Model - Transformer 架构

本文件实现一个完整的 GPT 模型，约 200 行核心代码。
参考: nanoGPT (https://github.com/karpathy/nanoGPT)

运行: python model.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 模型配置
# =============================================================================

@dataclass
class GPTConfig:
    """GPT 模型配置"""
    vocab_size: int = 6400       # 词表大小
    block_size: int = 512        # 最大序列长度
    n_embd: int = 512            # 嵌入维度
    n_head: int = 8              # 注意力头数
    n_layer: int = 8             # Transformer 层数
    dropout: float = 0.1         # Dropout 概率
    bias: bool = False           # 是否使用偏置


# =============================================================================
# 核心组件
# =============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization

    对每个样本的特征维度进行归一化，使训练更稳定。
    公式: y = (x - mean) / sqrt(var + eps) * weight + bias
    """

    def __init__(self, n_embd: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力（Causal Self-Attention）

    这是 GPT 的核心！每个 token 只能看到它之前的 token（包括自己）。

    计算过程:
    1. 将输入 x 投影为 Q, K, V
    2. 计算 Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
    3. 应用因果掩码，使每个位置只能看到之前的位置
    4. 多头注意力：将上述过程在多个"头"上并行进行
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V 投影（合并为一个矩阵以提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # 检查是否支持 Flash Attention（PyTorch 2.0+）
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch_size, seq_len, n_embd

        # 1. 计算 Q, K, V
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)  # 各 [B, T, C]

        # 2. 分成多个头: [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. 计算 Attention
        if self.flash:
            # PyTorch 2.0+ 使用高效的 Flash Attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=True  # 因果掩码
            )
        else:
            # 手动实现 Attention（用于理解原理）
            # Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            # 因果掩码：上三角矩阵设为 -inf
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # 4. 合并多个头: [B, n_head, T, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    前馈神经网络（Feed-Forward Network）

    结构: Linear -> GELU -> Linear
    先扩展到 4 倍，再压缩回来
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)      # [B, T, 4*C]
        x = self.gelu(x)      # 激活函数
        x = self.c_proj(x)    # [B, T, C]
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block

    结构（Pre-LayerNorm）:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    残差连接使深层网络更容易训练。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # 残差连接 + 注意力
        x = x + self.mlp(self.ln_2(x))   # 残差连接 + MLP
        return x


# =============================================================================
# 完整的 GPT 模型
# =============================================================================

class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) 模型

    结构:
    1. Token Embedding + Position Embedding
    2. N 个 Transformer Block
    3. Final LayerNorm
    4. LM Head（输出词表大小的 logits）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Embedding 层
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # 最后的 LayerNorm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # LM Head: 将隐藏状态映射到词表
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：Token Embedding 和 LM Head 共享权重
        self.tok_emb.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

        # 打印参数量
        print(f"GPT 模型参数量: {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple:
        """
        前向传播

        Args:
            idx: token IDs, shape [batch_size, seq_len]
            targets: 目标 token IDs（用于计算 loss）

        Returns:
            logits: shape [batch_size, seq_len, vocab_size]
            loss: 交叉熵损失（如果提供了 targets）
        """
        B, T = idx.size()
        assert self.config.block_size >= T, f"序列长度 {T} 超过最大长度 {self.config.block_size}"

        # 1. Embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # [T]
        tok_emb = self.tok_emb(idx)   # [B, T, C]
        pos_emb = self.pos_emb(pos)   # [T, C]
        x = self.drop(tok_emb + pos_emb)  # [B, T, C]

        # 2. Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final LayerNorm
        x = self.ln_f(x)

        # 4. LM Head
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # 5. 计算 Loss（如果提供了 targets）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        """
        自回归生成

        Args:
            idx: 起始 token IDs, shape [batch_size, seq_len]
            max_new_tokens: 要生成的 token 数量
            temperature: 采样温度（越高越随机）
            top_k: 只从概率最高的 k 个 token 中采样
        """
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置

            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# =============================================================================
# 演示代码
# =============================================================================

def demo_components():
    """演示各个组件"""
    print("=" * 60)
    print("组件演示")
    print("=" * 60)

    config = GPTConfig(vocab_size=1000, block_size=64, n_embd=128, n_head=4, n_layer=2)

    # 模拟输入
    B, T = 2, 32  # batch_size=2, seq_len=32
    x = torch.randn(B, T, config.n_embd)

    # 1. LayerNorm
    ln = LayerNorm(config.n_embd)
    out = ln(x)
    print("\n1. LayerNorm")
    print(f"   输入: {x.shape} -> 输出: {out.shape}")

    # 2. Attention
    attn = CausalSelfAttention(config)
    out = attn(x)
    print("\n2. CausalSelfAttention")
    print(f"   输入: {x.shape} -> 输出: {out.shape}")

    # 3. MLP
    mlp = MLP(config)
    out = mlp(x)
    print("\n3. MLP")
    print(f"   输入: {x.shape} -> 输出: {out.shape}")

    # 4. Transformer Block
    block = TransformerBlock(config)
    out = block(x)
    print("\n4. TransformerBlock")
    print(f"   输入: {x.shape} -> 输出: {out.shape}")


def demo_full_model():
    """演示完整模型"""
    print("\n" + "=" * 60)
    print("完整模型演示")
    print("=" * 60)

    config = GPTConfig(
        vocab_size=1000,
        block_size=64,
        n_embd=128,
        n_head=4,
        n_layer=4
    )

    model = GPT(config)

    # 模拟输入
    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    # 前向传播
    logits, loss = model(idx, targets)
    print("\n前向传播:")
    print(f"  输入 shape: {idx.shape}")
    print(f"  输出 logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # 生成
    print("\n生成演示:")
    start_ids = torch.randint(0, config.vocab_size, (1, 5))
    print(f"  起始 tokens: {start_ids.tolist()}")
    generated = model.generate(start_ids, max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"  生成 tokens: {generated.tolist()}")


def demo_params():
    """演示参数量计算"""
    print("\n" + "=" * 60)
    print("参数量计算")
    print("=" * 60)

    configs = [
        GPTConfig(vocab_size=6400, n_embd=512, n_head=8, n_layer=8),     # ~26M
        GPTConfig(vocab_size=6400, n_embd=768, n_head=12, n_layer=12),   # ~85M
        GPTConfig(vocab_size=6400, n_embd=1024, n_head=16, n_layer=24),  # ~300M
    ]

    for i, config in enumerate(configs):
        model = GPT(config)
        params = model.get_num_params()
        print(f"\n配置 {i+1}:")
        print(f"  n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")
        print(f"  参数量: {params/1e6:.2f}M")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT 模型演示")
    parser.add_argument("--mode", choices=["components", "full", "params", "all"], default="all",
                       help="演示模式")
    args = parser.parse_args()

    if args.mode in ["components", "all"]:
        demo_components()

    if args.mode in ["full", "all"]:
        demo_full_model()

    if args.mode in ["params", "all"]:
        demo_params()

    print("\n" + "=" * 60)
    print("下一步: 进入 step3_pretrain/ 学习如何预训练模型")
    print("=" * 60)
