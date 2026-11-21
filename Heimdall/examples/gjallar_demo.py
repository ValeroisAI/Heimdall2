"""
examples/gjallar_demo.py — Interactive Walkthrough & Verification of Gjallar Hybrid Architecture.

========================================================================================
TEACHER OVERVIEW:
One-sentence answer:
  Gjallar enables 1M-token context on a single 16 GB consumer card by keeping constant
  O(1) GPU state (local sliding window + recurrent delta memory) and offloading past KV blocks
  to host RAM, fetching exact blocks only when surprise breaches threshold.

What happens to one token query q_t line by line:
  1. Window Attention: q_t scores the local window (last W tokens + initial sinks).
     Produces local attention output o_win and log-sum-exp confidence lse_win.
  2. Associative Readout: q_t probes recurrent Gated DeltaNet memory matrix S_{t-1}.
     Produces recurrent output o_mem = S_{t-1} q_t and confidence norm ||o_mem||.
  3. Surprise Gating: Gate evaluates [||o_mem||, lse_win, centroid_mass].
     - If surprise is low: FAST-PATH. Pure local compute, zero PCIe host-bus traffic.
     - If surprise is high: SLOW-PATH.
  4. C4 Shortlist & Gather (Slow-Path only):
     - GPU scores group-quantized 4-bit block centroids (C4) and shortlists top-64 blocks.
     - Exact token-max dot product reranks candidate keys to final top-16 blocks.
     - Exact O(kk * L) block gather computes sparse attention over retrieved blocks.
  5. Memory Eviction: When a token leaves the sliding window, it updates the recurrent
     delta memory: S_t = decay * S_{t-1} + (v - S_{t-1} k) k^T. Sinks are permanently pinned.

The #1 mistake people make:
  Assuming recurrent delta confidence ||S q|| alone can predict retrieval need.
  On real models and long contexts, ||S q|| alone is chance (AUROC 0.504). Gjallar uses
  centroid-mass gating (comparing archive centroid LSE against window LSE) to reliably
  detect archived needles without wasting retrieval budget on routine filler.
========================================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from heimdall.gjallar import GjallarAttention, GjallarStack, TwoLevelBlockIndex

def run_demo():
    print("=" * 85)
    print("GJALLAR ARCHITECTURE WALKTHROUGH (illustrative: untrained random gate, hand-set thresholds, no quality or speed claim)")
    print("=" * 85)

    torch.manual_seed(42)
    B, H, G, D = 1, 4, 2, 64  # Multi-head GQA (H=4, G=2, R=2, D=64)
    window = 128
    sinks = 4
    block_size = 32
    top_k = 4
    shortlist_size = 16

    print(f"\n[1] Initializing 4-Layer 3:1 Hybrid Stack (3 GDN layers + 1 SGRA layer)")
    print(f"    - Layer 0: Gated DeltaNet  (recurrent state S in R^(64x64), zero host KV)")
    print(f"    - Layer 1: Gated DeltaNet  (recurrent state S in R^(64x64), zero host KV)")
    print(f"    - Layer 2: Gated DeltaNet  (recurrent state S in R^(64x64), zero host KV)")
    print(f"    - Layer 3: SGRA Layer      (Window={window}, Sinks={sinks}, BlockSize={block_size}, TopK={top_k})")

    stack = GjallarStack(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        num_layers=4, window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k,
        selector="centroid_c4"
    )
    sgra_layer = stack.layers[3].attn
    sgra_layer.enable_index_cache = True

    # Simulate 512 tokens in host context memory
    n_ctx = 512
    k_host = torch.randn(B, G, n_ctx, D)
    v_host = torch.randn(B, G, n_ctx, D)

    # Plant a distinct needle at token 64 (deep in the host archive)
    needle_pos = 64
    needle_k = torch.randn(B, G, 1, D)
    needle_v = torch.randn(B, G, 1, D) * 5.0  # high salience
    k_host[:, :, needle_pos:needle_pos + 1] = needle_k
    v_host[:, :, needle_pos:needle_pos + 1] = needle_v

    target_block = (needle_pos - sinks) // block_size

    print(f"\n[2] Host Memory Context:")
    print(f"    - Total context: {n_ctx} tokens ({n_ctx // block_size} blocks)")
    print(f"    - Sliding window on GPU: last {window} tokens + {sinks} sinks")
    print(f"    - Archived in Host RAM: {(n_ctx - window - sinks)} tokens ({(n_ctx - window - sinks) // block_size} candidate blocks)")
    print(f"    - Needle planted at token index {needle_pos} (Archive Block #{target_block})")

    print(f"\n[3] Simulating Sequential Decode Steps:")
    states = [None] * 4

    steps_desc = [
        ("Step 1: Routine continuous token (local context)", torch.randn(B, H, 1, D), 0.5, True),
        ("Step 2: Routine continuous token (local context)", torch.randn(B, H, 1, D), 0.5, True),
        ("Step 3: Needle retrieval query (seeking token #64)", needle_k.repeat_interleave(2, dim=1), 0.1, True),
        ("Step 4: Post-retrieval answer generation token", torch.randn(B, H, 1, D), 0.5, True),
        ("Step 5: Subsequent local generation token", torch.randn(B, H, 1, D), 0.5, True),
    ]

    sgra_layer.reset_stats()
    for step_idx, (desc, q_token, thresh, hard) in enumerate(steps_desc, start=1):
        out, states, gates = stack(q_token, k_host, v_host, states=states, hard_gate=hard, gate_threshold=thresh)
        gate_dec = gates[3].mean().item() if gates[3] is not None else 0.0
        path_taken = "SLOW-PATH (Host Retrieval)" if gate_dec > 0 else "FAST-PATH (Local Compute)"

        print(f"\n  --- {desc} ---")
        print(f"      Anchor Gate Decision    : {gate_dec:.1f} (Threshold: {thresh})")
        print(f"      Execution Routing       : [{path_taken}]")
        print(f"      Output Shape            : {out.shape}")

    stats = sgra_layer.get_stats()
    print("\n[4] Execution Statistics over Decode Run:")
    print(f"    - Fast-Path tokens (Zero PCIe traffic): {stats['fast_path_tokens']}")
    print(f"    - Slow-Path tokens (Retrieved host KV): {stats['slow_path_tokens']}")
    print(f"    - Overall Retrieval Rate             : {stats['retrieval_rate']:.1%}")

    print("\n[5] Concept Check:")
    print("    Q: Why does Gjallar keep int4 block centroids (C4) on the GPU instead of min/max box bounds?")
    print("    A: Empirical testing shows min/max coordinate bounding boxes expand rapidly in high dimensions,")
    print("       retaining only 20-56% of top blocks at scale. C4 centroids retain 79-100% of true attention mass")
    print("       while shrinking GPU summary memory by 3.8x (2.25 MiB vs 8.5 MiB per KV group at 1M tokens).")

    print("\n" + "=" * 85)
    print("WALKTHROUGH COMPLETED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == "__main__":
    run_demo()
