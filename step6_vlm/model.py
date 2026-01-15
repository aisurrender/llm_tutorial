"""
Step 6: VLM 模型实现

VLM = Vision Encoder + Projection + LLM

运行: python model.py --demo
"""

import sys
import os
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step2_gpt_model.model import GPT, GPTConfig


# =============================================================================
# 配置
# =============================================================================

@dataclass
class VLMConfig:
    """VLM 配置"""
    # Vision Encoder
    vision_dim: int = 768           # Vision Encoder 输出维度
    image_size: int = 224           # 图片大小
    patch_size: int = 16            # Patch 大小
    num_patches: int = 196          # Patch 数量 (224/16)^2

    # LLM
    vocab_size: int = 6400
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.1

    # Projection
    projection_type: str = "linear"  # "linear" or "mlp"


# =============================================================================
# Vision Encoder（简化版）
# =============================================================================

class SimpleViT(nn.Module):
    """
    简化版 Vision Transformer

    将图片分成 patches，然后用 Transformer 编码。
    实际使用时，通常用预训练的 CLIP-ViT。
    """

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        num_patches = (config.image_size // config.patch_size) ** 2

        # Patch embedding: 将每个 patch 投影到 vision_dim
        self.patch_embed = nn.Conv2d(
            3, config.vision_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.vision_dim))

        # Transformer layers
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.vision_dim,
                nhead=8,
                dim_feedforward=config.vision_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(4)  # 4 层，简化版
        ])

        self.norm = nn.LayerNorm(config.vision_dim)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图片 [B, 3, H, W]

        Returns:
            features: 图像特征 [B, num_patches, vision_dim]
        """
        # Patch embedding: [B, 3, 224, 224] -> [B, vision_dim, 14, 14]
        x = self.patch_embed(x)

        # Reshape: [B, vision_dim, 14, 14] -> [B, 196, vision_dim]
        x = x.flatten(2).transpose(1, 2)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


# =============================================================================
# Modality Projection
# =============================================================================

class ModalityProjection(nn.Module):
    """
    模态投影：将视觉特征映射到文本嵌入空间

    这是 VLM 的关键组件，起到"桥梁"作用。
    """

    def __init__(self, config: VLMConfig):
        super().__init__()

        if config.projection_type == "linear":
            # 简单线性投影
            self.proj = nn.Linear(config.vision_dim, config.n_embd)
        else:
            # MLP 投影
            self.proj = nn.Sequential(
                nn.Linear(config.vision_dim, config.n_embd * 2),
                nn.GELU(),
                nn.Linear(config.n_embd * 2, config.n_embd),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 视觉特征 [B, num_patches, vision_dim]

        Returns:
            projected: 投影后的特征 [B, num_patches, n_embd]
        """
        return self.proj(x)


# =============================================================================
# VLM 模型
# =============================================================================

class VLM(nn.Module):
    """
    Vision Language Model

    架构: Vision Encoder → Projection → LLM

    输入:
    - 图片: [B, 3, 224, 224]
    - 文本 token IDs: [B, seq_len]

    输出:
    - logits: [B, num_patches + seq_len, vocab_size]
    """

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        # 1. Vision Encoder
        self.vision_encoder = SimpleViT(config)

        # 2. Modality Projection
        self.projection = ModalityProjection(config)

        # 3. LLM
        llm_config = GPTConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            dropout=config.dropout,
        )
        self.llm = GPT(llm_config)

        # 特殊 token embedding（图片占位符）
        self.image_start_token = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)
        self.image_end_token = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)

        print(f"VLM 参数量: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图片

        Args:
            images: [B, 3, H, W]

        Returns:
            image_embeds: [B, num_patches + 2, n_embd]（包含起始和结束 token）
        """
        B = images.shape[0]

        # Vision Encoder
        vision_features = self.vision_encoder(images)  # [B, 196, 768]

        # Projection
        image_embeds = self.projection(vision_features)  # [B, 196, n_embd]

        # 添加图片起始和结束 token
        start = self.image_start_token.expand(B, -1, -1)
        end = self.image_end_token.expand(B, -1, -1)
        image_embeds = torch.cat([start, image_embeds, end], dim=1)

        return image_embeds

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        前向传播

        Args:
            images: 图片 [B, 3, H, W]
            input_ids: 文本 token IDs [B, seq_len]
            targets: 目标（用于计算 loss）

        Returns:
            logits, loss
        """
        embeddings_list = []

        # 1. 编码图片（如果有）
        if images is not None:
            image_embeds = self.encode_image(images)
            embeddings_list.append(image_embeds)

        # 2. 编码文本（如果有）
        if input_ids is not None:
            text_embeds = self.llm.tok_emb(input_ids)
            # 添加位置编码
            pos = torch.arange(0, input_ids.size(1), device=input_ids.device)
            text_embeds = text_embeds + self.llm.pos_emb(pos)
            embeddings_list.append(text_embeds)

        # 3. 拼接
        if len(embeddings_list) == 0:
            raise ValueError("需要提供 images 或 input_ids")

        x = torch.cat(embeddings_list, dim=1)
        x = self.llm.drop(x)

        # 4. Transformer
        for block in self.llm.blocks:
            x = block(x)
        x = self.llm.ln_f(x)

        # 5. LM Head
        logits = self.llm.lm_head(x)

        # 6. 计算 Loss
        loss = None
        if targets is not None:
            # 只对文本部分计算 loss
            if images is not None:
                # 图片部分不计算 loss
                num_image_tokens = self.config.num_patches + 2
                text_logits = logits[:, num_image_tokens:, :]
            else:
                text_logits = logits

            loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        图文生成

        Args:
            images: 图片 [B, 3, H, W]
            prompt_ids: 提示文本 [B, seq_len]
            max_new_tokens: 生成的 token 数量
        """
        B = images.shape[0]
        device = images.device

        # 编码图片
        image_embeds = self.encode_image(images)

        # 初始化
        if prompt_ids is not None:
            generated_ids = prompt_ids.clone()
        else:
            generated_ids = torch.zeros((B, 0), dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # 获取文本 embedding
            if generated_ids.size(1) > 0:
                text_embeds = self.llm.tok_emb(generated_ids)
                pos = torch.arange(0, generated_ids.size(1), device=device)
                text_embeds = text_embeds + self.llm.pos_emb(pos)
                x = torch.cat([image_embeds, text_embeds], dim=1)
            else:
                x = image_embeds

            x = self.llm.drop(x)

            # Transformer
            for block in self.llm.blocks:
                x = block(x)
            x = self.llm.ln_f(x)

            # 取最后一个位置的 logits
            logits = self.llm.lm_head(x[:, -1, :]) / temperature

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return generated_ids


# =============================================================================
# 演示
# =============================================================================

def demo():
    """演示 VLM 模型"""
    print("=" * 60)
    print("VLM 模型演示")
    print("=" * 60)

    config = VLMConfig(
        vision_dim=256,
        n_embd=256,
        n_head=8,
        n_layer=4,
        num_patches=49,  # 7x7
        image_size=112,
        patch_size=16,
    )

    model = VLM(config)

    # 模拟输入
    B = 2
    images = torch.randn(B, 3, config.image_size, config.image_size)
    input_ids = torch.randint(0, config.vocab_size, (B, 20))
    targets = torch.randint(0, config.vocab_size, (B, 20))

    print(f"\n输入:")
    print(f"  图片: {images.shape}")
    print(f"  文本: {input_ids.shape}")

    # 前向传播
    logits, loss = model(images, input_ids, targets)

    print(f"\n输出:")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # 参数统计
    vision_params = sum(p.numel() for p in model.vision_encoder.parameters())
    proj_params = sum(p.numel() for p in model.projection.parameters())
    llm_params = sum(p.numel() for p in model.llm.parameters())

    print(f"\n参数分布:")
    print(f"  Vision Encoder: {vision_params/1e6:.2f}M")
    print(f"  Projection: {proj_params/1e6:.4f}M")
    print(f"  LLM: {llm_params/1e6:.2f}M")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", default=True)
    args = parser.parse_args()

    if args.demo:
        demo()
