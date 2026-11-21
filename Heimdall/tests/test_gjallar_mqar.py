"""
test_gjallar_mqar.py — Unit test suite for bench/gjallar_mqar.py.

Verifies:
  1. batch() output shapes under MIXED=2.
  2. Balanced 16 easy + 16 hard queries per sequence.
  3. Ground-truth target integrity (targets match latest written values).
  4. Distance invariant: hard queries lie >= WIN tokens from write; easy queries lie < WIN.
  5. Net forward pass execution for modes: none, always, oraclehard, gated (with RES=0 and RES=32).

Execution time is well under 30 seconds on CPU.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Configure environment before importing bench.gjallar_mqar
os.environ["MIXED"] = "2"
os.environ["D"] = "16"
os.environ["PAIRS"] = "48"

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench.gjallar_mqar as mqar


def test_batch_shapes():
    """Verify batch() returns tensors of identical matching shapes [B, L]."""
    rng = np.random.default_rng(42)
    B, gap = 4, 32
    ids, tg, hd = mqar.batch(rng, B, gap)

    assert ids.shape == tg.shape == hd.shape, f"Shape mismatch: ids={ids.shape}, tg={tg.shape}, hd={hd.shape}"
    assert ids.dim() == 2
    assert ids.shape[0] == B
    # Expected length:
    # 48 archived pairs * 2 = 96 tokens
    # gap tokens = 32
    # 32 query episodes: each has a pair (2 tokens) + 1 query token = 3 * 32 = 96 tokens
    # Total = 96 + 32 + 96 = 224 tokens
    expected_len = mqar.PAIRS * 2 + gap + 32 * 3
    assert ids.shape[1] == expected_len, f"Expected length {expected_len}, got {ids.shape[1]}"
    print("PASS: test_batch_shapes")


def test_mixed2_query_counts():
    """Verify MIXED=2 produces exactly 16 easy and 16 hard queries per sequence."""
    rng = np.random.default_rng(123)
    ids, tg, hd = mqar.batch(rng, 8, 64)
    for b in range(8):
        m = tg[b] != -100
        total_queries = m.sum().item()
        assert total_queries == 32, f"Batch {b}: expected 32 total queries, got {total_queries}"

        easy_count = (m & ~hd[b]).sum().item()
        hard_count = (m & hd[b]).sum().item()
        assert easy_count == 16, f"Batch {b}: expected 16 easy queries, got {easy_count}"
        assert hard_count == 16, f"Batch {b}: expected 16 hard queries, got {hard_count}"
    print("PASS: test_mixed2_query_counts")


def test_target_equals_latest_written_value():
    """Verify every target in tg corresponds exactly to the most recent written value for that key."""
    rng = np.random.default_rng(456)
    ids, tg, hd = mqar.batch(rng, 4, 16)
    for b in range(4):
        seq = ids[b].tolist()
        targets = tg[b].tolist()
        written = {}

        t = 0
        while t < len(seq):
            token = seq[t]
            target = targets[t]
            if target != -100:
                # Query token
                key_idx = token - mqar.K0
                assert key_idx in written, f"Batch {b}, pos {t}: queried key {key_idx} was never written!"
                expected_val = mqar.V0 + written[key_idx]
                assert target == expected_val, (
                    f"Batch {b}, pos {t}: target {target} != expected {expected_val} for key {key_idx}"
                )
                t += 1
            else:
                # Value token write
                if mqar.V0 <= token < mqar.V0 + mqar.NV and t > 0:
                    prev_token = seq[t - 1]
                    if mqar.K0 <= prev_token < mqar.K0 + mqar.NK:
                        key_idx = prev_token - mqar.K0
                        val_idx = token - mqar.V0
                        written[key_idx] = val_idx
                t += 1
    print("PASS: test_target_equals_latest_written_value")


def test_query_distance_bounds():
    """Verify hard queries lie >= WIN tokens after their write, and easy queries lie < WIN tokens."""
    rng = np.random.default_rng(789)
    ids, tg, hd = mqar.batch(rng, 4, 64)
    for b in range(4):
        seq = ids[b].tolist()
        targets = tg[b].tolist()
        is_hard = hd[b].tolist()
        last_write_pos = {}

        for pos, token in enumerate(seq):
            target = targets[pos]
            if target != -100:
                # Query position
                key_idx = token - mqar.K0
                assert key_idx in last_write_pos, f"Batch {b}, pos {pos}: key {key_idx} queried without prior write"
                write_pos = last_write_pos[key_idx]
                dist = pos - write_pos

                if is_hard[pos]:
                    assert dist >= mqar.WIN, (
                        f"Batch {b}, pos {pos}: hard query distance {dist} < WIN {mqar.WIN} (write was at {write_pos})"
                    )
                else:
                    assert dist < mqar.WIN, (
                        f"Batch {b}, pos {pos}: easy query distance {dist} >= WIN {mqar.WIN} (write was at {write_pos})"
                    )
            else:
                # Track position of value token writes
                if mqar.V0 <= token < mqar.V0 + mqar.NV and pos > 0:
                    prev_token = seq[pos - 1]
                    if mqar.K0 <= prev_token < mqar.K0 + mqar.NK:
                        key_idx = prev_token - mqar.K0
                        last_write_pos[key_idx] = pos
    print("PASS: test_query_distance_bounds")


def test_net_forward_all_modes():
    """Verify Net forward pass runs cleanly for modes none, always, oraclehard, and gated with RES=0 and RES=32."""
    rng = np.random.default_rng(999)
    B, gap = 2, 16
    ids, tg, hd = mqar.batch(rng, B, gap)

    modes = ["none", "always", "oraclehard", "gated"]

    # 1. Test RES = 0
    mqar.RES = 0
    for mode in modes:
        torch.manual_seed(0)
        net = mqar.Net(mode)
        logits, gates = net(ids, hardmask=hd)
        assert logits.shape == (B, ids.shape[1], mqar.VOCAB), f"RES=0, mode={mode}: logits shape mismatch {logits.shape}"
        assert gates.shape == (B, ids.shape[1], 1), f"RES=0, mode={mode}: gates shape mismatch {gates.shape}"
        assert not torch.isnan(logits).any(), f"RES=0, mode={mode}: logits contain NaNs"
        assert not torch.isnan(gates).any(), f"RES=0, mode={mode}: gates contain NaNs"

    # 2. Test RES = 32
    mqar.RES = 32
    for mode in modes:
        torch.manual_seed(0)
        net = mqar.Net(mode)
        logits, gates = net(ids, hardmask=hd)
        assert logits.shape == (B, ids.shape[1], mqar.VOCAB), f"RES=32, mode={mode}: logits shape mismatch {logits.shape}"
        assert gates.shape == (B, ids.shape[1], 1), f"RES=32, mode={mode}: gates shape mismatch {gates.shape}"
        assert not torch.isnan(logits).any(), f"RES=32, mode={mode}: logits contain NaNs"
        assert not torch.isnan(gates).any(), f"RES=32, mode={mode}: gates contain NaNs"

    print("PASS: test_net_forward_all_modes")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING GJALLAR MQAR UNIT TESTS")
    print("=" * 60)
    test_batch_shapes()
    test_mixed2_query_counts()
    test_target_equals_latest_written_value()
    test_query_distance_bounds()
    test_net_forward_all_modes()
    print("=" * 60)
    print("ALL 5 TESTS PASSED SUCCESSFULLY (< 10s CPU runtime)!")
    print("=" * 60)
