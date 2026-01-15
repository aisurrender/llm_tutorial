"""
Step 2: GPT Model - Transformer 架构

练习文件：请完成标记为 TODO 的部分

运行测试: python model_exercise.py
或在 tutorial.ipynb 中验证
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 模型配置（已实现）
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
# Layer Normalization（已实现）
# =============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization"""

    def __init__(self, n_embd: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


# =============================================================================
# TODO 1: MLP（前馈神经网络）
# =============================================================================

class MLP(nn.Module):
    """
    前馈神经网络（Feed-Forward Network）

    结构: Linear(n_embd -> 4*n_embd) -> GELU -> Linear(4*n_embd -> n_embd) -> Dropout

    这是 Transformer 中的"思考"部分，对每个位置独立处理。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # 这些层已经定义好了
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        任务：依次通过 c_fc -> gelu -> c_proj -> dropout

        Args:
            x: 输入张量 [batch_size, seq_len, n_embd]

        Returns:
            输出张量 [batch_size, seq_len, n_embd]

        示例:
            x: [2, 32, 512] -> [2, 32, 2048] -> [2, 32, 2048] -> [2, 32, 512] -> [2, 32, 512]
        """
        # TODO: 实现前向传播（大约 4 行代码）
        # x = self.c_fc(x)
        # x = self.gelu(x)
        # x = ...
        # x = ...
        # return x
        raise NotImplementedError("请实现 MLP 的 forward 方法")


# =============================================================================
# TODO 2: Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer Block

    结构（Pre-LayerNorm + Residual Connection）:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    残差连接使深层网络更容易训练（梯度可以直接流过）。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        任务：实现 Pre-LayerNorm + Residual Connection

        步骤：
        1. x = x + self.attn(self.ln_1(x))  # 注意力 + 残差
        2. x = x + self.mlp(self.ln_2(x))   # MLP + 残差

        Args:
            x: 输入张量 [batch_size, seq_len, n_embd]

        Returns:
            输出张量 [batch_size, seq_len, n_embd]
        """
        # TODO: 实现前向传播（2 行代码）
        # 提示: 残差连接就是 x = x + something(x)
        raise NotImplementedError("请实现 TransformerBlock 的 forward 方法")


# =============================================================================
# TODO 3: Causal Self-Attention（核心！）
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    因果自注意力（Causal Self-Attention）

    这是 GPT 的核心！让每个 token 能"看到"它之前的所有 token。

    计算过程:
    1. 将输入 x 投影为 Q, K, V
    2. 分成多个头
    3. 计算 Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
    4. 应用因果掩码（每个位置只能看到之前的位置）
    5. 合并多个头
    6. 输出投影
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V 投影（合并为一个矩阵）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [B, T, C] (batch_size, seq_len, n_embd)

        Returns:
            输出张量 [B, T, C]
        """
        B, T, C = x.size()

        # =====================================================================
        # Step 1: 计算 Q, K, V
        # =====================================================================
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)  # 各 [B, T, C]

        # =====================================================================
        # Step 2: 分成多个头
        # [B, T, C] -> [B, n_head, T, head_dim]
        # =====================================================================
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # =====================================================================
        # TODO 3a: 计算注意力分数
        # 公式: att = (Q @ K^T) / sqrt(head_dim)
        # =====================================================================
        # 提示:
        # - 使用 @ 进行矩阵乘法
        # - K^T 用 k.transpose(-2, -1) 表示
        # - sqrt(head_dim) 用 math.sqrt(self.head_dim)
        # att = ...  # shape: [B, n_head, T, T]
        raise NotImplementedError("请计算注意力分数 att = Q @ K^T / sqrt(d_k)")

        # =====================================================================
        # TODO 3b: 应用因果掩码
        # 让每个位置只能看到之前的位置（包括自己）
        # =====================================================================
        # 因果掩码是一个上三角矩阵，上三角部分设为 -inf
        # 提示:
        # - torch.triu(torch.ones(T, T), diagonal=1) 创建上三角矩阵
        # - att.masked_fill(mask, float('-inf')) 将 mask 为 True 的位置填充为 -inf
        # mask = ...
        # att = att.masked_fill(mask, float('-inf'))
        raise NotImplementedError("请应用因果掩码")

        # =====================================================================
        # TODO 3c: Softmax + Dropout + 与 V 相乘
        # =====================================================================
        # 提示:
        # - F.softmax(att, dim=-1) 在最后一个维度上做 softmax
        # - self.attn_dropout(att) 应用 dropout
        # - att @ v 得到输出
        # att = F.softmax(...)
        # att = self.attn_dropout(att)
        # y = att @ v  # [B, n_head, T, head_dim]
        raise NotImplementedError("请完成 softmax 和与 V 的乘法")

        # =====================================================================
        # Step 3: 合并多个头
        # [B, n_head, T, head_dim] -> [B, T, C]
        # =====================================================================
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # =====================================================================
        # Step 4: 输出投影
        # =====================================================================
        y = self.resid_dropout(self.c_proj(y))
        return y


# =============================================================================
# TODO 4: 完整的 GPT 模型
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

        # LM Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享
        self.tok_emb.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple:
        """
        前向传播

        Args:
            idx: token IDs [batch_size, seq_len]
            targets: 目标 token IDs（可选，用于计算 loss）

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: 交叉熵损失（如果提供了 targets）
        """
        B, T = idx.size()
        assert self.config.block_size >= T, f"序列长度 {T} 超过最大长度 {self.config.block_size}"

        # =================================================================
        # TODO 4a: Token Embedding + Position Embedding
        # =================================================================
        # 提示:
        # - pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # - tok_emb = self.tok_emb(idx)  # [B, T, C]
        # - pos_emb = self.pos_emb(pos)  # [T, C]
        # - x = self.drop(tok_emb + pos_emb)
        raise NotImplementedError("请实现 Embedding 层")

        # =================================================================
        # TODO 4b: 通过所有 Transformer Blocks
        # =================================================================
        # 提示: for block in self.blocks: x = block(x)
        raise NotImplementedError("请通过 Transformer Blocks")

        # =================================================================
        # TODO 4c: Final LayerNorm + LM Head
        # =================================================================
        # 提示:
        # - x = self.ln_f(x)
        # - logits = self.lm_head(x)
        raise NotImplementedError("请实现最后的 LayerNorm 和 LM Head")

        # 计算 Loss
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
        """自回归生成（已实现）"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# =============================================================================
# 测试代码
# =============================================================================

def test_mlp():
    """测试 MLP"""
    config = GPTConfig(n_embd=128)
    mlp = MLP(config)

    x = torch.randn(2, 32, 128)
    try:
        out = mlp(x)
        assert out.shape == x.shape, f"输出 shape 应为 {x.shape}，得到 {out.shape}"
        print("✅ MLP 测试通过!")
        return True
    except NotImplementedError as e:
        print(f"⚠️ MLP: {e}")
        return False


def test_transformer_block():
    """测试 Transformer Block"""
    config = GPTConfig(n_embd=128, n_head=4)

    # 先测试依赖项
    try:
        mlp = MLP(config)
        _ = mlp(torch.randn(2, 32, 128))
    except NotImplementedError:
        print("⚠️ TransformerBlock: 请先完成 MLP")
        return False

    try:
        attn = CausalSelfAttention(config)
        _ = attn(torch.randn(2, 32, 128))
    except NotImplementedError:
        print("⚠️ TransformerBlock: 请先完成 CausalSelfAttention")
        return False

    block = TransformerBlock(config)
    x = torch.randn(2, 32, 128)
    try:
        out = block(x)
        assert out.shape == x.shape, f"输出 shape 应为 {x.shape}，得到 {out.shape}"
        print("✅ TransformerBlock 测试通过!")
        return True
    except NotImplementedError as e:
        print(f"⚠️ TransformerBlock: {e}")
        return False


def test_attention():
    """测试 Self-Attention"""
    config = GPTConfig(n_embd=128, n_head=4)
    attn = CausalSelfAttention(config)

    x = torch.randn(2, 32, 128)
    try:
        out = attn(x)
        assert out.shape == x.shape, f"输出 shape 应为 {x.shape}，得到 {out.shape}"
        print("✅ CausalSelfAttention 测试通过!")
        return True
    except NotImplementedError as e:
        print(f"⚠️ CausalSelfAttention: {e}")
        return False


def test_gpt():
    """测试完整 GPT 模型"""
    config = GPTConfig(vocab_size=1000, block_size=64, n_embd=128, n_head=4, n_layer=2)

    # 检查依赖项
    try:
        block = TransformerBlock(config)
        _ = block(torch.randn(2, 32, 128))
    except NotImplementedError:
        print("⚠️ GPT: 请先完成 TransformerBlock")
        return False

    try:
        model = GPT(config)
        idx = torch.randint(0, 1000, (2, 32))
        logits, loss = model(idx, idx)

        assert logits.shape == (2, 32, 1000), f"logits shape 错误: {logits.shape}"
        assert loss is not None, "loss 应该不为 None"

        print("✅ GPT 测试通过!")
        print(f"   参数量: {model.get_num_params()/1e6:.2f}M")
        return True
    except NotImplementedError as e:
        print(f"⚠️ GPT: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("GPT 模型测试")
    print("=" * 60)
    print()

    test_mlp()
    print()
    test_attention()
    print()
    test_transformer_block()
    print()
    test_gpt()
