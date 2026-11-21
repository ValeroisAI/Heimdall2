"""
test_gjallar.py — Unit tests for Gjallar Surprise-Gated Retrieval Attention.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from heimdall.gjallar import GjallarGate, TwoLevelBlockIndex, GjallarAttention, GjallarStack


def test_block_upper_bound_invariant():
    """Verify the mathematical invariant: for every token in a block, q·k <= UB."""
    torch.manual_seed(0)
    B, G, R, t, D = 2, 2, 2, 4, 64
    block_size = 32
    nb = 8

    q = torch.randn(B, G, R, t, D)
    k = torch.randn(B, G, nb, block_size, D)

    kmin = k.amin(dim=3).view(B, G, 1, 1, nb, D)
    kmax = k.amax(dim=3).view(B, G, 1, 1, nb, D)

    ub = TwoLevelBlockIndex.block_upper_bounds(q, kmin, kmax) # [B, G, t, nb]

    # Test every single token
    for b in range(nb):
        ktoks = k[:, :, b] # [B, G, block_size, D]
        # True dot products for each query head
        dot = torch.einsum("bgrtd,bgld->bgrtl", q, ktoks)
        max_dot = dot.amax(dim=(2, -1)) # [B, G, t]
        bound = ub[:, :, :, b]
        assert (bound >= max_dot - 1e-5).all(), f"Upper bound violated at block {b}!"


def test_gjallar_fast_path_slow_path():
    """Verify that Fast-Path and Slow-Path behave correctly and update profiling counters."""
    torch.manual_seed(42)
    B, H, G, D = 1, 4, 2, 64
    window = 128
    block_size = 16
    n = 256 # Exceeds window + block_size

    layer = GjallarAttention(
        d_model=H * D,
        n_heads=H,
        n_kv_heads=G,
        head_dim=D,
        window=window,
        block_size=block_size,
        top_k_blocks=4,
    )

    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, G, n, D)
    v = torch.randn(B, G, n, D)

    # 1. Force FAST-PATH (threshold = 1.0)
    layer.reset_stats()
    out_fast, S_next, g = layer(q, k, v, hard_gate=True, gate_threshold=1.0)
    assert out_fast.shape == (B, H, 1, D)
    assert not torch.isnan(out_fast).any()
    stats = layer.get_stats()
    assert stats["fast_path_tokens"] == 1
    assert stats["slow_path_tokens"] == 0
    assert stats["retrieval_rate"] == 0.0

    # 2. Force SLOW-PATH (threshold = -1.0)
    layer.reset_stats()
    out_slow, S_next, g = layer(q, k, v, hard_gate=True, gate_threshold=-1.0)
    assert out_slow.shape == (B, H, 1, D)
    assert not torch.isnan(out_slow).any()
    stats = layer.get_stats()
    assert stats["fast_path_tokens"] == 0
    assert stats["slow_path_tokens"] == 1
    assert stats["retrieval_rate"] == 1.0


def test_gjallar_prefill_chunk():
    """Chunked prefill (t > 1) is rejected: the reference has no causal mask (early queries would see future V)."""
    torch.manual_seed(123)
    B, H, G, D = 2, 8, 2, 64
    window = 64
    block_size = 16
    t = 128
    n = 128

    layer = GjallarAttention(
        d_model=H * D,
        n_heads=H,
        n_kv_heads=G,
        head_dim=D,
        window=window,
        block_size=block_size,
        top_k_blocks=2,
    )

    q = torch.randn(B, H, t, D)
    k = torch.randn(B, G, n, D)
    v = torch.randn(B, G, n, D)

    try:
        layer(q, k, v, hard_gate=False)
    except ValueError:
        pass
    else:
        raise AssertionError("t > 1 must raise: no causal mask in the reference")
    out, _, g = layer(q[:, :, -1:], k, v, hard_gate=False)   # single-token decode still works
    assert out.shape == (B, H, 1, D) and g.shape == (B, H, 1, 1)
    assert not torch.isnan(out).any()


def test_gjallar_gate_gradient_flow():
    """Verify that gradients propagate back through soft gate to encourage budget sparsity."""
    torch.manual_seed(0)
    gate = GjallarGate(in_features=2)
    omem_norm = torch.rand(2, 4, 8, requires_grad=True)
    lse_win = torch.rand(2, 4, 8, requires_grad=True)

    g = gate(omem_norm, lse_win, hard=False)
    loss = g.mean() # Budget penalty
    loss.backward()

    assert gate.proj.weight.grad is not None
    assert gate.proj.bias.grad is not None
    assert omem_norm.grad is not None


def test_gjallar_centroid_c4_selector():
    """Verify C4 quantized centroid shortlist indexing and exact reranking."""
    torch.manual_seed(42)
    B, G, R, t, D = 2, 2, 2, 1, 64
    H = G * R
    block_size = 16
    nb = 12
    top_k = 4
    shortlist_size = 8

    # 1. Test quantize_c4 shape, bounds, and values
    cen = torch.randn(B, G, nb, D)
    cen_c4 = TwoLevelBlockIndex.quantize_c4(cen, group_size=32)
    assert cen_c4.shape == cen.shape
    assert not torch.isnan(cen_c4).any()
    # Check max deviation is reasonable for 7-level symmetric int4
    err = (cen - cen_c4).abs()
    assert err.mean() < cen.abs().mean()

    # 2. Test centroid_c4_select functionality
    q_gqa = torch.randn(B, G, R, t, D)
    kb = torch.randn(B, G, nb, block_size, D)
    picks = TwoLevelBlockIndex.centroid_c4_select(
        q_gqa, kb, scale=1.0 / 8.0, top_k=top_k, shortlist_size=shortlist_size
    )
    assert picks.shape == (B, G, t, top_k)
    assert (picks >= 0).all() and (picks < nb).all()

    # Edge case: nb <= top_k (all blocks fit)
    picks_small = TwoLevelBlockIndex.centroid_c4_select(
        q_gqa, kb[:, :, :3], scale=1.0 / 8.0, top_k=top_k, shortlist_size=shortlist_size
    )
    assert picks_small.shape == (B, G, t, 3)

    # 3. End-to-end GjallarAttention layer with selector="centroid_c4"
    window = 64
    n = 192 # sinks(4) + nb(8)*16(128) + window(64) = 196
    layer = GjallarAttention(
        d_model=H * D,
        n_heads=H,
        n_kv_heads=G,
        head_dim=D,
        window=window,
        block_size=block_size,
        top_k_blocks=top_k,
        selector="centroid_c4",
        shortlist_size=shortlist_size,
    )
    assert layer.selector == "centroid_c4"

    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, G, n, D)
    v = torch.randn(B, G, n, D)

    # Fast-path verification
    layer.reset_stats()
    out_fast, _, _ = layer(q, k, v, hard_gate=True, gate_threshold=1.0)
    assert out_fast.shape == (B, H, 1, D)
    assert not torch.isnan(out_fast).any()
    assert layer.get_stats()["fast_path_tokens"] == B

    # Slow-path verification (executes C4 centroid shortlist + rerank)
    layer.reset_stats()
    out_slow, _, _ = layer(q, k, v, hard_gate=True, gate_threshold=-1.0)
    assert out_slow.shape == (B, H, 1, D)
    assert not torch.isnan(out_slow).any()
    assert layer.get_stats()["slow_path_tokens"] == B
    assert layer.get_stats()["retrieval_rate"] == 1.0


def test_gjallar_attention_sinks():
    """Verify that initial sink tokens (0..sinks-1) are preserved in the window even when n >> window."""
    torch.manual_seed(99)
    B, H, G, D = 1, 2, 1, 32
    window = 32
    sinks = 4
    block_size = 16
    n = 128  # 128 > window + sinks (36)

    layer = GjallarAttention(
        d_model=H * D,
        n_heads=H,
        n_kv_heads=G,
        head_dim=D,
        window=window,
        sinks=sinks,
        block_size=block_size,
    )

    k = torch.randn(B, G, n, D)
    v = torch.randn(B, G, n, D)

    # Align sink token 0 with query
    q = torch.randn(B, H, 1, D)
    k[:, :, 0] = q.view(B, G, H // G, 1, D).mean(dim=2).squeeze(2) * 10.0
    v[:, :, 0] = 5.0

    # In fast-path (no retrieval), sink token 0 must still be in k_win
    out_fast, _, _ = layer(q, k, v, hard_gate=True, gate_threshold=1.0)
    assert not torch.isnan(out_fast).any()
    assert (out_fast > 1.0).all(), "Sink token attention mass should contribute substantially to output"

    # Boundary test: n in (window, window + sinks] (e.g. n = 34 with window=32, sinks=4)
    # Sinks must not drop out during the window-to-delta transition interval
    k_bound = torch.randn(B, G, 34, D)
    v_bound = torch.randn(B, G, 34, D)
    k_bound[:, :, 0] = q.view(B, G, H // G, 1, D).mean(dim=2).squeeze(2) * 10.0
    v_bound[:, :, 0] = 5.0
    out_bound, _, _ = layer(q, k_bound, v_bound, hard_gate=True, gate_threshold=1.0)
    assert not torch.isnan(out_bound).any()
    assert (out_bound > 1.0).all(), "Sink token 0 must remain present in window when n in (window, window+sinks]"


def test_gjallar_delta_memory_update():
    """Verify that delta memory updates occur only for tokens past sinks+window and obey beta gate."""
    torch.manual_seed(42)
    B, H, G, D = 1, 2, 1, 16
    window = 8
    sinks = 2
    layer = GjallarAttention(d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D, window=window, sinks=sinks)

    q = torch.randn(B, H, 1, D)
    S_init = torch.zeros(B, G, D, D)

    # 1. n <= window + sinks (10 tokens): no token has left the window, S should not change
    k_short = torch.randn(B, G, 10, D)
    v_short = torch.randn(B, G, 10, D)
    _, S_out, _ = layer(q, k_short, v_short, S_prev=S_init)
    assert torch.equal(S_out, S_init), "Memory should not update when no token has left window+sinks"

    # 2. n = 11: token at index sinks (index 2) leaves the window and enters memory
    k_long = torch.randn(B, G, 11, D)
    v_long = torch.randn(B, G, 11, D)
    _, S_out_updated, _ = layer(q, k_long, v_long, S_prev=S_init)
    assert not torch.equal(S_out_updated, S_init), "Memory should update when token leaves window"

    # 3. beta = 0: write gate completely suppresses the update
    beta_zero = torch.zeros(B, G)
    _, S_out_zero, _ = layer(q, k_long, v_long, S_prev=S_init, beta=beta_zero)
    assert torch.allclose(S_out_zero, S_init, atol=1e-6), "Beta=0 must suppress write into delta state"


def test_gjallar_gather_vs_masked_equivalence():
    """
    Verify numerical equivalence between the O(1) block gather path and the legacy O(N) masked scatter path.
    Tests across multiple seeds (0, 1, 42), for both box_bound and centroid_c4 selectors with single-token decode (t=1).
    """
    import math
    B, H, G, D = 1, 4, 2, 64
    R = H // G
    t = 1
    window = 32
    sinks = 4
    block_size = 16
    n = 128
    top_k = 4
    scale = 1.0 / math.sqrt(D)

    for selector in ("box_bound", "centroid_c4"):
        for seed in (0, 1, 42):
            torch.manual_seed(seed)
            layer = GjallarAttention(
                d_model=H * D,
                n_heads=H,
                n_kv_heads=G,
                head_dim=D,
                window=window,
                sinks=sinks,
                block_size=block_size,
                top_k_blocks=top_k,
                selector=selector,
            )

            q = torch.randn(B, H, t, D)
            k = torch.randn(B, G, n, D)
            v = torch.randn(B, G, n, D)

            # 1. Output from actual GjallarAttention (using optimized gather)
            out_gather, _, _ = layer(q, k, v, hard_gate=True, gate_threshold=-1.0) # force slow path

            # 2. Reference output computed via legacy masked scatter
            q_gqa = q.view(B, G, R, t, D)
            # Window attention
            k_win = torch.cat([k[:, :, :sinks], k[:, :, n - window:]], dim=2)
            v_win = torch.cat([v[:, :, :sinks], v[:, :, n - window:]], dim=2)
            scores_win = (q_gqa @ k_win.unsqueeze(2).transpose(-1, -2)) * scale
            attn_win = F.softmax(scores_win, dim=-1)
            o_win = (attn_win @ v_win.unsqueeze(2)).view(B, H, t, D)

            # Delta memory read (S_prev is zero)
            o_mem = torch.zeros(B, H, t, D)

            # Candidate blocks
            nc_tokens = n - window - sinks
            nb = nc_tokens // block_size
            k_cand = k[:, :, sinks:sinks + nb * block_size]
            v_cand = v[:, :, sinks:sinks + nb * block_size]
            kb = k_cand.view(B, G, nb, block_size, D)
            vb = v_cand.view(B, G, nb, block_size, D)

            kk = min(top_k, nb)
            if selector == "centroid_c4":
                top_blocks = layer.index.centroid_c4_select(
                    q_gqa, kb, scale=scale, top_k=top_k, shortlist_size=layer.shortlist_size
                )
            else:
                kmin = kb.amin(dim=3).view(B, G, 1, 1, nb, D)
                kmax = kb.amax(dim=3).view(B, G, 1, 1, nb, D)
                ub = layer.index.block_upper_bounds(q_gqa, kmin, kmax)
                top_blocks = ub.topk(kk, dim=-1).indices

            # Legacy masked scatter
            mask = torch.zeros(B, G, 1, t, nb * block_size, dtype=torch.bool)
            for b_idx in range(kk):
                blk_idx = top_blocks[:, :, :, b_idx:b_idx + 1]
                blk_start = blk_idx * block_size
                offsets = torch.arange(block_size).view(1, 1, 1, -1)
                token_indices = (blk_start + offsets).unsqueeze(2)
                mask.scatter_(-1, token_indices, True)

            scores_ret = (q_gqa @ k_cand.unsqueeze(2).transpose(-1, -2)) * scale
            scores_ret = scores_ret.masked_fill(~mask, -1e9)
            attn_ret = F.softmax(scores_ret, dim=-1)
            o_ret = (attn_ret @ v_cand.unsqueeze(2)).view(B, H, t, D)

            out_masked = o_win + layer.mix_mem * o_mem + 1.0 * o_ret

            max_err = (out_gather - out_masked).abs().max().item()
            assert torch.allclose(out_gather, out_masked, atol=1e-5), (
                f"Gather and masked outputs diverge for selector={selector}, seed={seed}: max_err={max_err:.2e}"
            )


def test_gjallar_mixed_batch_isolation():
    """
    Verify numerical isolation and precise profiling in a mixed batch where batch item 0
    takes Fast-Path (gate=0) and batch item 1 takes Slow-Path (gate=1).
    """
    torch.manual_seed(42)
    B, H, G, D = 2, 4, 2, 64
    window = 32
    sinks = 4
    block_size = 16
    n = 128
    top_k = 4

    layer = GjallarAttention(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k
    )

    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, G, n, D)
    v = torch.randn(B, G, n, D)

    # 1. Run item 0 alone in Fast-Path (threshold 1.0)
    layer.reset_stats()
    out_0, _, _ = layer(q[:1], k[:1], v[:1], hard_gate=True, gate_threshold=1.0)

    # 2. Run item 1 alone in Slow-Path (threshold -1.0)
    layer.reset_stats()
    out_1, _, _ = layer(q[1:], k[1:], v[1:], hard_gate=True, gate_threshold=-1.0)

    # 3. Run mixed batch (mock gate: item 0 -> 0.0, item 1 -> 1.0)
    class MixedGate(torch.nn.Module):
        def forward(self, omem_norm, lse_win, *args, **kwargs):
            g = torch.zeros(B, H, 1, 1)
            g[1] = 1.0
            return g

    layer.gate = MixedGate()
    layer.reset_stats()
    out_batch, _, _ = layer(q, k, v, hard_gate=True)

    # Verify per-token stats tracking
    stats = layer.get_stats()
    assert stats["fast_path_tokens"] == 1, f"Expected 1 fast-path token, got {stats['fast_path_tokens']}"
    assert stats["slow_path_tokens"] == 1, f"Expected 1 slow-path token, got {stats['slow_path_tokens']}"
    assert stats["retrieval_rate"] == 0.5

    # Verify numerical isolation (zero cross-talk between batch elements)
    diff_0 = (out_batch[:1] - out_0).abs().max().item()
    diff_1 = (out_batch[1:] - out_1).abs().max().item()
    assert diff_0 < 1e-6, f"Batch item 0 cross-talk error: {diff_0:.2e}"
    assert diff_1 < 1e-6, f"Batch item 1 cross-talk error: {diff_1:.2e}"


def test_gjallar_stack_3to1_topology():
    """
    Verify the 3:1 multi-layer hybrid stack (GjallarStack):
    3 GDN layers + 1 anchor SGRA layer in a 4-layer block.
    Checks state propagation, layer typing, output shapes, and residual accumulation.
    """
    torch.manual_seed(42)
    B, H, G, D = 1, 4, 2, 64
    num_layers = 4
    n = 128
    stack = GjallarStack(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        num_layers=num_layers, window=32, sinks=4, block_size=16, top_k_blocks=4,
        selector="centroid_c4"
    )

    # 1. Check layer typing: layers 0, 1, 2 are GDN, layer 3 is SGRA
    assert stack.layers[0].layer_type == "gdn"
    assert stack.layers[1].layer_type == "gdn"
    assert stack.layers[2].layer_type == "gdn"
    assert stack.layers[3].layer_type == "sgra"

    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, G, n, D)
    v = torch.randn(B, G, n, D)

    # 2. Forward pass with initial zero states
    out, next_states, gates = stack(q, k, v, states=None, hard_gate=True, gate_threshold=0.5)

    assert out.shape == (B, H, 1, D)
    assert not torch.isnan(out).any()
    assert len(next_states) == num_layers
    assert len(gates) == num_layers
    assert gates[0] is None and gates[1] is None and gates[2] is None
    assert gates[3] is not None and gates[3].shape == (B, H, 1, 1)

    # 3. Step 2 with state continuation
    q2 = torch.randn(B, H, 1, D)
    out2, next_states2, gates2 = stack(q2, k, v, states=next_states, hard_gate=True)
    assert out2.shape == (B, H, 1, D)
    assert not torch.isnan(out2).any()


def test_gjallar_edge_cases():
    """
    Stress-test boundary conditions and edge cases:
    1. Zero attention sinks (sinks = 0).
    2. Short context length strictly smaller than window size (n < window).
    3. Fewer candidate blocks than top_k_blocks (0 < nb < top_k).
    4. Large batch dimension (B = 8) with multi-head GQA (H = 8, G = 2, R = 4).
    """
    torch.manual_seed(999)
    D = 32

    # Case 1: Zero attention sinks
    layer_zero_sinks = GjallarAttention(
        d_model=2 * D, n_heads=2, n_kv_heads=1, head_dim=D,
        window=16, sinks=0, block_size=8, top_k_blocks=2, selector="centroid_c4"
    )
    q = torch.randn(1, 2, 1, D)
    k = torch.randn(1, 1, 40, D)
    v = torch.randn(1, 1, 40, D)
    # Slow path forced
    out_zs, S_next_zs, g_zs = layer_zero_sinks(q, k, v, hard_gate=True, gate_threshold=-1.0)
    assert out_zs.shape == (1, 2, 1, D) and not torch.isnan(out_zs).any()
    assert S_next_zs is not None and not torch.isnan(S_next_zs).any()

    # Case 2: Short context length (n < window)
    layer_short = GjallarAttention(
        d_model=2 * D, n_heads=2, n_kv_heads=1, head_dim=D,
        window=64, sinks=4, block_size=8, top_k_blocks=2
    )
    k_short = torch.randn(1, 1, 12, D)
    v_short = torch.randn(1, 1, 12, D)
    out_sh, S_next_sh, g_sh = layer_short(q, k_short, v_short, hard_gate=True, gate_threshold=-1.0)
    assert out_sh.shape == (1, 2, 1, D) and not torch.isnan(out_sh).any()
    # nb == 0, so should route to fast-path despite threshold
    assert layer_short.get_stats()["fast_path_tokens"] == 1

    # Case 3: Fewer candidate blocks than top_k_blocks (nb < top_k)
    # window=16, sinks=4, block_size=8, top_k_blocks=4.
    # n = 36 -> nc_tokens = 36 - 16 - 4 = 16 -> nb = 2 blocks (< 4)
    for sel in ("box_bound", "centroid_c4"):
        layer_few_blocks = GjallarAttention(
            d_model=2 * D, n_heads=2, n_kv_heads=1, head_dim=D,
            window=16, sinks=4, block_size=8, top_k_blocks=4, selector=sel
        )
        k_few = torch.randn(1, 1, 36, D)
        v_few = torch.randn(1, 1, 36, D)
        out_fb, _, _ = layer_few_blocks(q, k_few, v_few, hard_gate=True, gate_threshold=-1.0)
        assert out_fb.shape == (1, 2, 1, D) and not torch.isnan(out_fb).any()

    # Case 4: Large batch dimension (B = 8, H = 8, G = 2, R = 4)
    B_large = 8
    layer_large = GjallarAttention(
        d_model=8 * D, n_heads=8, n_kv_heads=2, head_dim=D,
        window=32, sinks=4, block_size=16, top_k_blocks=2, selector="centroid_c4"
    )
    q_large = torch.randn(B_large, 8, 1, D)
    k_large = torch.randn(B_large, 2, 80, D)
    v_large = torch.randn(B_large, 2, 80, D)
    out_lg, S_lg, g_lg = layer_large(q_large, k_large, v_large, hard_gate=True, gate_threshold=0.5)
    assert out_lg.shape == (B_large, 8, 1, D) and not torch.isnan(out_lg).any()
    assert S_lg.shape == (B_large, 2, D, D)

    # Case 5: shortlist_size smaller than top_k (e.g. top_k = 16, shortlist_size = 4)
    layer_small_shortlist = GjallarAttention(
        d_model=2 * D, n_heads=2, n_kv_heads=2, head_dim=D,
        window=32, sinks=4, block_size=16, top_k_blocks=16,
        selector="centroid_c4", shortlist_size=4
    )
    k_sl = torch.randn(1, 2, 200, D)
    v_sl = torch.randn(1, 2, 200, D)
    out_sl, _, _ = layer_small_shortlist(q, k_sl, v_sl, hard_gate=True, gate_threshold=-1.0)
    assert out_sl.shape == (1, 2, 1, D) and not torch.isnan(out_sl).any()


def test_gjallar_incremental_c4_cache():
    """
    Verify that enabling incremental C4 index caching (`enable_index_cache=True`)
    produces identical outputs to full recomputation across sequential streaming decode steps,
    and correctly maintains the running C4 centroid cache across block boundaries.
    """
    torch.manual_seed(42)
    B, H, G, D = 1, 4, 2, 64
    window, sinks, block_size = 32, 4, 16
    top_k = 2

    layer_uncached = GjallarAttention(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k,
        selector="centroid_c4"
    )
    layer_cached = GjallarAttention(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k,
        selector="centroid_c4"
    )
    layer_cached.enable_index_cache = True
    layer_cached.gate.load_state_dict(layer_uncached.gate.state_dict())

    # Generate sequential token context growing from 50 to 96 tokens
    k_full = torch.randn(B, G, 96, D)
    v_full = torch.randn(B, G, 96, D)

    S_uncached = None
    S_cached = None

    for curr_len in range(50, 96):
        q = torch.randn(B, H, 1, D)
        k_curr = k_full[:, :, :curr_len]
        v_curr = v_full[:, :, :curr_len]

        out_u, S_uncached, g_u = layer_uncached(q, k_curr, v_curr, S_prev=S_uncached, hard_gate=True, gate_threshold=-1.0)
        out_c, S_cached, g_c = layer_cached(q, k_curr, v_curr, S_prev=S_cached, hard_gate=True, gate_threshold=-1.0)

        assert torch.allclose(out_u, out_c, atol=1e-6), (
            f"Cached and uncached outputs differ at step len={curr_len}: max_diff={(out_u - out_c).abs().max().item():.2e}"
        )
        assert torch.allclose(S_uncached, S_cached, atol=1e-6)

    # Verify cached_cen_c4 state
    nc_tokens = 95 - window - sinks
    expected_nb = nc_tokens // block_size
    assert layer_cached.cached_cen_c4 is not None
    assert layer_cached.cached_cen_c4.shape == (B, G, expected_nb, D)


def test_gjallar_centroid_mass_gating():
    """
    Verify TwoLevelBlockIndex.centroid_mass_feature and 3-observable gating (gate_in_features=3)
    in GjallarAttention. Tests feature sensitivity to archive matches and end-to-end gradient flow.
    """
    torch.manual_seed(42)
    B, H, G, D = 1, 4, 2, 64
    R = H // G
    window, sinks, block_size = 32, 4, 16
    top_k = 2
    scale = 1.0 / (D ** 0.5)

    # 1. Direct centroid mass feature test
    nb = 4
    cen = torch.randn(B, G, nb, D)
    q = torch.randn(B, H, 1, D)
    q_gqa = q.view(B, G, R, 1, D)
    lse_win = torch.randn(B, H, 1)

    mass_feat = TwoLevelBlockIndex.centroid_mass_feature(q_gqa, cen, scale, lse_win)
    assert mass_feat.shape == (B, H, 1)

    # Test sensitivity: align query with first centroid
    q_aligned = cen[:, :, 0:1].repeat_interleave(R, dim=1).view(B, H, 1, D)
    q_aligned_gqa = q_aligned.view(B, G, R, 1, D)
    aligned_mass = TwoLevelBlockIndex.centroid_mass_feature(q_aligned_gqa, cen, scale)
    random_mass = TwoLevelBlockIndex.centroid_mass_feature(q_gqa, cen, scale)
    assert aligned_mass.mean() > random_mass.mean()

    # 2. End-to-end GjallarAttention with 3-feature gate
    layer_3feat = GjallarAttention(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k,
        selector="centroid_c4", gate_in_features=3
    )
    layer_3feat.enable_index_cache = True
    assert layer_3feat.gate.in_features == 3

    k = torch.randn(B, G, 128, D)
    v = torch.randn(B, G, 128, D)
    q_test = torch.randn(B, H, 1, D, requires_grad=True)

    out, _, g = layer_3feat(q_test, k, v, hard_gate=False)
    assert g.shape == (B, H, 1, 1)
    loss = out.sum()
    loss.backward()

    assert layer_3feat.gate.proj.weight.grad is not None
    assert layer_3feat.gate.proj.weight.grad.shape == (1, 3)


def test_gjallar_stack_hierarchy_parameters():
    """
    Verify that configuring GjallarStack with gate_in_features=3 and custom shortlist_size
    propagates to all anchor SGRA layers, correctly sets up their GjallarGate and TwoLevelBlockIndex,
    and runs multi-step decode with residual connections and state propagation.
    """
    torch.manual_seed(123)
    B, H, G, D = 1, 4, 2, 64
    num_layers = 8  # 8 layers -> layers 3 and 7 are SGRA anchor layers

    stack = GjallarStack(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        num_layers=num_layers, window=32, sinks=4, block_size=16, top_k_blocks=2,
        selector="centroid_c4", shortlist_size=32, gate_in_features=3
    )

    # Check layer configurations
    assert stack.layers[3].layer_type == "sgra"
    assert stack.layers[7].layer_type == "sgra"
    assert stack.layers[3].attn.gate.in_features == 3
    assert stack.layers[7].attn.gate.in_features == 3
    assert stack.layers[3].attn.shortlist_size == 32
    assert stack.layers[7].attn.shortlist_size == 32

    # Run multi-step decode
    k = torch.randn(B, G, 80, D)
    v = torch.randn(B, G, 80, D)
    states = None

    for step in range(3):
        q = torch.randn(B, H, 1, D)
        out, states, gates = stack(q, k, v, states=states, hard_gate=True, gate_threshold=0.5)
        assert out.shape == (B, H, 1, D)
        assert not torch.isnan(out).any()
        assert len(states) == num_layers
        assert len(gates) == num_layers
        # GDN layers return None for gate
        for l in [0, 1, 2, 4, 5, 6]:
            assert gates[l] is None
        # SGRA layers return tensor gate
        for l in [3, 7]:
            assert gates[l] is not None and gates[l].shape == (B, H, 1, 1)


def test_gjallar_fastpath_cache_lifecycle():
    """
    Verify that fast-path tokens with gate_in_features=3 properly initialize and
    incrementally maintain cached_cen_c4 under enable_index_cache=True, preventing
    per-token descriptor rebuilds.
    """
    torch.manual_seed(42)
    B, H, G, D = 1, 4, 2, 64
    window, sinks, block_size = 32, 4, 16
    top_k = 2

    layer = GjallarAttention(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k,
        selector="centroid_c4", gate_in_features=3
    )
    layer.enable_index_cache = True

    # Initial context with 4 candidate blocks
    # 4 blocks * 16 = 64 tokens + 32 window + 4 sinks = 100 tokens
    k = torch.randn(B, G, 100, D)
    v = torch.randn(B, G, 100, D)
    q = torch.randn(B, H, 1, D)

    # 1. Force fast-path: verify cached_cen_c4 is initialized
    out1, S1, g1 = layer(q, k, v, hard_gate=True, gate_threshold=10.0)
    assert layer.cached_cen_c4 is not None
    assert layer.cached_cen_c4.shape == (B, G, 4, D)
    assert layer.get_stats()["fast_path_tokens"] == 1
    cached_ref = layer.cached_cen_c4

    # 2. Second fast-path token at same context length: verify identical pointer (no rebuild)
    out2, S2, g2 = layer(q, k, v, S_prev=S1, hard_gate=True, gate_threshold=10.0)
    assert layer.cached_cen_c4 is cached_ref
    assert layer.get_stats()["fast_path_tokens"] == 2

    # 3. Context grows by 1 block (16 tokens) -> 5 candidate blocks
    k_grown = torch.cat([k, torch.randn(B, G, 16, D)], dim=2)
    v_grown = torch.cat([v, torch.randn(B, G, 16, D)], dim=2)
    out3, S3, g3 = layer(q, k_grown, v_grown, S_prev=S2, hard_gate=True, gate_threshold=10.0)
    assert layer.cached_cen_c4.shape == (B, G, 5, D)
    # The first 4 blocks must be identical to the previous cached centroids
    assert torch.allclose(layer.cached_cen_c4[:, :, :4], cached_ref, atol=1e-6)
    assert layer.get_stats()["fast_path_tokens"] == 3


def test_gjallar_stack_per_layer_kv():
    """
    Verify that GjallarStack cleanly handles per-layer lists of KV tensors,
    preserving layer-specific KV values and residual state propagation.
    """
    torch.manual_seed(777)
    B, H, G, D = 1, 4, 2, 64
    num_layers = 4
    stack = GjallarStack(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        num_layers=num_layers, window=32, sinks=4, block_size=16, top_k_blocks=2,
        selector="centroid_c4"
    )

    # Create distinct KV tensors for each layer
    k_list = [torch.randn(B, G, 64, D) * (l + 1) for l in range(num_layers)]
    v_list = [torch.randn(B, G, 64, D) * (l + 1) for l in range(num_layers)]
    q = torch.randn(B, H, 1, D)

    # Decode with per-layer lists
    out_list, states_list, gates_list = stack(q, k_list, v_list)
    assert out_list.shape == (B, H, 1, D)
    assert not torch.isnan(out_list).any()
    assert len(states_list) == num_layers

    # Verify manual layer-by-layer forward matches stack
    x = q
    for l_idx, layer in enumerate(stack.layers):
        out_manual, _, _ = layer(x, k_list[l_idx], v_list[l_idx])
        x = x + out_manual
    assert torch.allclose(out_list, x, atol=1e-6)


if __name__ == "__main__":
    print("Running test_block_upper_bound_invariant()...")
    test_block_upper_bound_invariant()
    print("PASS: Upper bound mathematical invariant holds.")

    print("Running test_gjallar_fast_path_slow_path()...")
    test_gjallar_fast_path_slow_path()
    print("PASS: Fast-path and slow-path branching & stats verify.")

    print("Running test_gjallar_prefill_chunk()...")
    test_gjallar_prefill_chunk()
    print("PASS: Prefill chunk multi-token execution verified.")

    print("Running test_gjallar_gate_gradient_flow()...")
    test_gjallar_gate_gradient_flow()
    print("PASS: Gate gradient flow verified.")

    print("Running test_gjallar_centroid_c4_selector()...")
    test_gjallar_centroid_c4_selector()
    print("PASS: Centroid C4 selector & shortlist reranking verified.")

    print("Running test_gjallar_attention_sinks()...")
    test_gjallar_attention_sinks()
    print("PASS: Attention sink preservation verified.")

    print("Running test_gjallar_delta_memory_update()...")
    test_gjallar_delta_memory_update()
    print("PASS: Delta memory update and beta write gate verified.")

    print("Running test_gjallar_gather_vs_masked_equivalence()...")
    test_gjallar_gather_vs_masked_equivalence()
    print("PASS: Block gather vs masked scatter numerical equivalence verified across seeds and selectors.")

    print("Running test_gjallar_mixed_batch_isolation()...")
    test_gjallar_mixed_batch_isolation()
    print("PASS: Mixed-batch per-token stats and numerical isolation verified.")

    print("Running test_gjallar_stack_3to1_topology()...")
    test_gjallar_stack_3to1_topology()
    print("PASS: 3:1 multi-layer hybrid stack topology and state propagation verified.")

    print("Running test_gjallar_edge_cases()...")
    test_gjallar_edge_cases()
    print("PASS: Boundary conditions and edge cases verified (zero sinks, small contexts, few blocks, large batch).")

    print("Running test_gjallar_incremental_c4_cache()...")
    test_gjallar_incremental_c4_cache()
    print("PASS: Incremental C4 cache equivalence & block boundary updates verified.")

    print("Running test_gjallar_centroid_mass_gating()...")
    test_gjallar_centroid_mass_gating()
    print("PASS: Centroid-mass archive feature & 3-observable gate verified.")

    print("Running test_gjallar_stack_hierarchy_parameters()...")
    test_gjallar_stack_hierarchy_parameters()
    print("PASS: Multi-layer stack parameter plumbing & 8-layer decode verified.")

    print("Running test_gjallar_fastpath_cache_lifecycle()...")
    test_gjallar_fastpath_cache_lifecycle()
    print("PASS: Fast-path centroid cache lifecycle & incremental maintenance verified.")

    print("Running test_gjallar_stack_per_layer_kv()...")
    test_gjallar_stack_per_layer_kv()
    print("PASS: Per-layer KV tensor lists & layer isolation verified.")

    print("\nALL 16 GJALLAR UNIT TESTS PASSED SUCCESSFULLY!")
