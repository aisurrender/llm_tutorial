"""
Step 6: VLM 模型

练习文件：请完成标记为 TODO 的部分

VLM 架构：Vision Encoder + Projection + LLM
"""

import torch
import torch.nn as nn


class VLM(nn.Module):
    """
    Vision Language Model

    架构:
    1. Vision Encoder: 图片 → 图像特征 [B, num_patches, vision_dim]
    2. Projection: 图像特征 → 文本空间 [B, num_patches, text_dim]
    3. LLM: 拼接图像和文本 token → 生成回复

    图片相当于"特殊的外语"，Vision Encoder 是"翻译官"
    """

    def __init__(
        self,
        llm,                    # 语言模型
        vision_dim: int = 768,  # Vision Encoder 输出维度
        text_dim: int = 512,    # LLM 嵌入维度
        num_patches: int = 196, # 图像 patch 数量 (14x14)
    ):
        super().__init__()

        self.llm = llm
        self.num_patches = num_patches

        # =====================================================================
        # TODO 1: 实现 Projection Layer
        # =====================================================================
        # Projection 的作用：将视觉特征映射到文本嵌入空间
        #
        # 为什么需要 Projection？
        # - Vision Encoder 输出的特征维度是 vision_dim（如 768）
        # - LLM 的嵌入维度是 text_dim（如 512）
        # - 两者维度不同，需要一个"桥梁"来转换
        #
        # 实现方式：
        # 方式 1: 简单线性层
        #   self.projection = nn.Linear(vision_dim, text_dim)
        #
        # 方式 2: MLP（更强的表达能力）
        #   self.projection = nn.Sequential(
        #       nn.Linear(vision_dim, text_dim * 2),
        #       nn.GELU(),
        #       nn.Linear(text_dim * 2, text_dim),
        #   )
        #
        # 请选择一种方式实现:
        # self.projection = ...
        raise NotImplementedError("请实现 Projection Layer")

    # =========================================================================
    # TODO 2: 实现 forward 方法
    # =========================================================================
    def forward(
        self,
        image_features: torch.Tensor,  # [B, num_patches, vision_dim]
        text_ids: torch.Tensor,        # [B, text_len]
        labels: torch.Tensor = None,   # [B, text_len] 用于计算 loss
    ):
        """
        前向传播

        步骤：
        1. 将图像特征通过 Projection 映射到文本空间
        2. 获取文本的嵌入
        3. 拼接图像嵌入和文本嵌入
        4. 通过 LLM 生成输出

        Args:
            image_features: Vision Encoder 输出的图像特征
            text_ids: 文本 token IDs
            labels: 训练时的标签

        Returns:
            logits: 预测的 logits
            loss: 如果提供了 labels
        """
        B = image_features.shape[0]

        # Step 1: Projection
        # image_embeds = self.projection(image_features)  # [B, num_patches, text_dim]

        # Step 2: 获取文本嵌入
        # text_embeds = self.llm.tok_emb(text_ids)  # [B, text_len, text_dim]

        # Step 3: 拼接（图像在前，文本在后）
        # combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        # [B, num_patches + text_len, text_dim]

        # Step 4: 通过 LLM（跳过嵌入层，直接输入 embedding）
        # 这里简化处理，实际需要修改 LLM 支持直接输入 embedding

        raise NotImplementedError("请实现 forward 方法")


# =============================================================================
# 测试代码
# =============================================================================

def test_projection():
    """测试 Projection Layer"""
    print("测试 Projection Layer...")

    vision_dim = 768
    text_dim = 512

    # 简单的 mock LLM
    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(1000, text_dim)

    try:
        vlm = VLM(
            llm=MockLLM(),
            vision_dim=vision_dim,
            text_dim=text_dim,
        )

        # 测试 projection
        image_features = torch.randn(2, 196, vision_dim)
        projected = vlm.projection(image_features)

        assert projected.shape == (2, 196, text_dim), \
            f"Projection 输出形状错误: {projected.shape}"

        print("✅ Projection Layer 测试通过!")
        print(f"   输入: {image_features.shape}")
        print(f"   输出: {projected.shape}")
        return True
    except NotImplementedError as e:
        print(f"⚠️ {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("VLM 测试")
    print("=" * 60)
    print()
    test_projection()
