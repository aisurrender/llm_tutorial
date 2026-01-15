"""Tests for shared utilities."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import get_device, get_lr, set_seed


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = torch.rand(10)
        set_seed(42)
        b = torch.rand(10)
        assert torch.allclose(a, b)

    def test_different_seeds_produce_different_results(self):
        set_seed(42)
        a = torch.rand(10)
        set_seed(123)
        b = torch.rand(10)
        assert not torch.allclose(a, b)


class TestGetDevice:
    def test_cpu_always_available(self):
        assert get_device("cpu") == "cpu"

    def test_auto_returns_valid_device(self):
        device = get_device("auto")
        assert device in ["cuda", "mps", "cpu"]


class TestGetLR:
    def test_warmup_starts_near_zero(self):
        lr = get_lr(step=0, warmup_steps=100, total_steps=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr < 1e-3
        assert lr > 0

    def test_warmup_reaches_max_lr(self):
        lr = get_lr(step=99, warmup_steps=100, total_steps=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr == pytest.approx(1e-3, rel=0.01)

    def test_decay_reaches_min_lr(self):
        lr = get_lr(step=1000, warmup_steps=100, total_steps=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr == 1e-4

    def test_lr_decreases_during_decay(self):
        lr1 = get_lr(step=200, warmup_steps=100, total_steps=1000, max_lr=1e-3, min_lr=1e-4)
        lr2 = get_lr(step=500, warmup_steps=100, total_steps=1000, max_lr=1e-3, min_lr=1e-4)
        lr3 = get_lr(step=900, warmup_steps=100, total_steps=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr1 > lr2 > lr3

    def test_cosine_shape(self):
        lr = get_lr(step=550, warmup_steps=100, total_steps=1000, max_lr=1e-3, min_lr=1e-4)
        expected = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * 0.5))
        assert lr == pytest.approx(expected, rel=0.01)
