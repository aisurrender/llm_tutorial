"""Tests for Step 2: GPT Model implementation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from step2_gpt_model.model_solution import (
    GPT,
    MLP,
    CausalSelfAttention,
    GPTConfig,
    LayerNorm,
    TransformerBlock,
)


@pytest.fixture
def small_config():
    return GPTConfig(
        vocab_size=1000,
        block_size=64,
        n_embd=128,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )


class TestLayerNorm:
    def test_output_shape(self, small_config):
        ln = LayerNorm(small_config.n_embd)
        x = torch.randn(2, 32, small_config.n_embd)
        out = ln(x)
        assert out.shape == x.shape

    def test_normalized_stats(self, small_config):
        ln = LayerNorm(small_config.n_embd)
        x = torch.randn(2, 32, small_config.n_embd)
        out = ln(x)
        assert out.mean(dim=-1).abs().max() < 0.1
        assert (out.std(dim=-1) - 1).abs().max() < 0.1


class TestMLP:
    def test_output_shape(self, small_config):
        mlp = MLP(small_config)
        x = torch.randn(2, 32, small_config.n_embd)
        out = mlp(x)
        assert out.shape == x.shape


class TestCausalSelfAttention:
    def test_output_shape(self, small_config):
        attn = CausalSelfAttention(small_config)
        x = torch.randn(2, 32, small_config.n_embd)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_masking(self, small_config):
        attn = CausalSelfAttention(small_config)
        attn.flash = False

        x1 = torch.randn(1, 10, small_config.n_embd)
        x2 = x1.clone()
        x2[0, 5:, :] = torch.randn(5, small_config.n_embd)

        out1 = attn(x1)
        out2 = attn(x2)

        assert torch.allclose(out1[0, :5, :], out2[0, :5, :], atol=1e-5)


class TestTransformerBlock:
    def test_output_shape(self, small_config):
        block = TransformerBlock(small_config)
        x = torch.randn(2, 32, small_config.n_embd)
        out = block(x)
        assert out.shape == x.shape


class TestGPT:
    def test_output_shapes(self, small_config):
        model = GPT(small_config)
        idx = torch.randint(0, small_config.vocab_size, (2, 32))
        targets = torch.randint(0, small_config.vocab_size, (2, 32))

        logits, loss = model(idx, targets)

        assert logits.shape == (2, 32, small_config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # scalar

    def test_forward_without_targets(self, small_config):
        model = GPT(small_config)
        idx = torch.randint(0, small_config.vocab_size, (2, 32))

        logits, loss = model(idx)

        assert logits.shape == (2, 32, small_config.vocab_size)
        assert loss is None

    def test_generate(self, small_config):
        model = GPT(small_config)
        model.eval()
        start = torch.randint(0, small_config.vocab_size, (1, 5))

        generated = model.generate(start, max_new_tokens=10, temperature=1.0, top_k=50)

        assert generated.shape == (1, 15)  # 5 start + 10 generated

    def test_weight_tying(self, small_config):
        model = GPT(small_config)
        assert model.tok_emb.weight is model.lm_head.weight
