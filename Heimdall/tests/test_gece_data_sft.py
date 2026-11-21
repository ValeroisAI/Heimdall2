"""
Unit tests for night run data preparation, SFT loss masking, and evaluation scripts.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from heimdall.io import DEFAULT_TOKENIZER
from bench.sft_egit import format_chatml_and_mask, SFTDataset, nullcontext, NONFINITE_EXIT
from bench.ornek_uret import PROMPTS, sample_token


def test_sft_format_and_loss_masking():
    tok = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    im_start = tok.token_to_id("<|im_start|>")
    im_end = tok.token_to_id("<|im_end|>")
    assert im_start is not None and im_end is not None

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]

    inp, tgt = format_chatml_and_mask(messages, tok, max_len=512)
    assert len(inp) == len(tgt)
    assert len(inp) > 0

    # Ensure system and user are masked (-100)
    # The first portion of targets must be -100
    assert tgt[0] == -100

    # Ensure assistant response tokens are NOT -100
    active_targets = [t for t in tgt if t != -100]
    assert len(active_targets) > 0

    # Verify that the active targets end with im_end token
    assert active_targets[-1] == im_end or im_end in active_targets


def test_sft_dataset_batching():
    tok = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    pad_id = tok.token_to_id("<pad>") or 0

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "data.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"User question {i}"},
                        {"role": "assistant", "content": f"Assistant answer {i} with some explanation."}
                    ]
                }) + "\n")

        ds = SFTDataset(str(jsonl_path), tok, max_len=128)
        assert len(ds) == 10

        x, y = ds.next_batch(4, pad_id=pad_id)
        assert x.shape == y.shape
        assert x.shape[0] == 4
        # Target must contain active tokens and -100 tokens
        assert (y != -100).sum() > 0
        assert (y == -100).sum() > 0


def test_ornek_uret_prompts_and_sampling():
    assert len(PROMPTS) == 10
    for cat, title, prompt in PROMPTS:
        assert len(cat) > 0
        assert len(title) > 0
        assert len(prompt) > 10

    # Test greedy sampling
    logits = torch.tensor([1.0, 5.0, 2.0])
    token = sample_token(logits, temp=0.0)
    assert token == 1

    # Test stochastic sampling
    token_stoch = sample_token(logits, temp=0.7, top_p=0.9)
    assert token_stoch in (0, 1, 2)


def test_nullcontext_import_and_nonfinite_exit():
    ctx = nullcontext()
    with ctx:
        pass
    assert NONFINITE_EXIT == 75
