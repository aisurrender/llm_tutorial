"""Tests for Step 1: Tokenizer implementations."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1_tokenizer.tokenizer_solution import BPETokenizer, CharTokenizer


class TestCharTokenizer:
    def test_build_vocab(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hello")
        assert tokenizer.vocab_size >= 4 + 4  # 4 special tokens + "helo"

    def test_encode_known_text(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hello")
        ids = tokenizer.encode("hello")
        assert isinstance(ids, list)
        assert len(ids) == 5

    def test_decode_reverses_encode(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hello world")
        text = "hello"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_unknown_char_returns_unk(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("abc")
        ids = tokenizer.encode("xyz")
        assert all(i == tokenizer.special_tokens["<unk>"] for i in ids)


class TestBPETokenizer:
    def test_train_creates_vocab(self):
        tokenizer = BPETokenizer()
        tokenizer.train("low lower lowest " * 10, vocab_size=50, verbose=False)
        assert tokenizer.vocab_size > 4  # More than just special tokens

    def test_train_creates_merges(self):
        tokenizer = BPETokenizer()
        tokenizer.train("low lower lowest " * 10, vocab_size=50, verbose=False)
        assert len(tokenizer.merges) > 0

    def test_encode_decode_consistency(self):
        tokenizer = BPETokenizer()
        tokenizer.train("low lower lowest " * 10, vocab_size=50, verbose=False)
        text = "lower"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_frequent_pairs_merged(self):
        tokenizer = BPETokenizer()
        tokenizer.train("ab ab ab ab ab ab ab ab ab ab", vocab_size=20, verbose=False)
        assert ("a", "b") in tokenizer.merges or "ab" in tokenizer.vocab
