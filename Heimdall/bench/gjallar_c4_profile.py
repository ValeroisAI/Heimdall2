"""
bench/gjallar_c4_profile.py — Performance Doctor Profiling of Gjallar C4 Index & Reranking.

Profiles breakdown of decode latency in TwoLevelBlockIndex.centroid_c4_select across
growing context sizes (8K to 131K tokens), and measures the speedup of incremental
centroid maintenance & 5D gather optimization.

Metric: CPU decode latency (ms) per token step.
Evidence: Detailed phase breakdown comparing recomputed vs cached centroid index.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from heimdall.gjallar import TwoLevelBlockIndex

def profile_c4_selection():
    torch.manual_seed(42)
    B, G, R, t, D = 1, 2, 4, 1, 128  # Qwen-like: H=8, G=2, R=4, D=128
    L = 64  # block size 64 tokens
    top_k = 16
    shortlist_size = 64
    scale = 1.0 / (D ** 0.5)

    contexts = [8192, 32768, 65536, 131072]

    print("=" * 85)
    print("PERFORMANCE DOCTOR: Gjallar C4 Selection & Reranking Latency Breakdown")
    print(f"Geometry: B={B}, G={G}, R={R}, D={D}, BlockSize={L}, Shortlist={shortlist_size}, TopK={top_k}")
    print("=" * 85)
    print("PART 1: Baseline (Recomputing Centroids & Quantization Every Decode Step)")
    print(f"{'Context':>8} | {'Blocks':>6} | {'Cen Mean':>9} | {'C4 Quant':>9} | {'Score+TopK':>10} | {'Short Gather':>12} | {'Exact Rerank':>12} | {'Total':>8}")
    print("-" * 85)

    baseline_totals = {}

    for n_ctx in contexts:
        nb = n_ctx // L
        kb = torch.randn(B, G, nb, L, D)
        q_gqa = torch.randn(B, G, R, t, D)

        # Warmup
        for _ in range(5):
            _ = TwoLevelBlockIndex.centroid_c4_select(q_gqa, kb, scale, top_k=top_k, shortlist_size=shortlist_size)

        iters = 30
        t_mean, t_quant, t_score, t_gather, t_rerank = 0.0, 0.0, 0.0, 0.0, 0.0

        for _ in range(iters):
            t0 = time.perf_counter()
            cen = kb.mean(dim=3)
            t1 = time.perf_counter()
            t_mean += (t1 - t0)

            cen_c4 = TwoLevelBlockIndex.quantize_c4(cen, group_size=32)
            t2 = time.perf_counter()
            t_quant += (t2 - t1)

            c4_scores = torch.einsum("bgrtd,bgnd->bgrtn", q_gqa, cen_c4) * scale
            sc = c4_scores.amax(dim=2)
            short_idx = sc.topk(shortlist_size, dim=-1).indices
            t3 = time.perf_counter()
            t_score += (t3 - t2)

            idx_exp = short_idx.unsqueeze(-1).unsqueeze(-1).expand(B, G, t, shortlist_size, L, D)
            kb_exp = kb.unsqueeze(2).expand(B, G, t, nb, L, D)
            kb_short = torch.gather(kb_exp, 3, idx_exp)
            t4 = time.perf_counter()
            t_gather += (t4 - t3)

            dot_short = torch.einsum("bgrtd,bgtmld->bgrtml", q_gqa, kb_short) * scale
            tok_max = dot_short.amax(dim=-1).amax(dim=2)
            pick = tok_max.topk(top_k, dim=-1).indices
            top_blocks = torch.gather(short_idx, dim=-1, index=pick)
            t5 = time.perf_counter()
            t_rerank += (t5 - t4)

        t_mean = (t_mean / iters) * 1000.0
        t_quant = (t_quant / iters) * 1000.0
        t_score = (t_score / iters) * 1000.0
        t_gather = (t_gather / iters) * 1000.0
        t_rerank = (t_rerank / iters) * 1000.0
        total = t_mean + t_quant + t_score + t_gather + t_rerank
        baseline_totals[n_ctx] = total

        print(f"{n_ctx:>8} | {nb:>6} | {t_mean:>8.3f}ms | {t_quant:>8.3f}ms | {t_score:>9.3f}ms | {t_gather:>11.3f}ms | {t_rerank:>11.3f}ms | {total:>7.3f}ms")

    print("\n" + "=" * 85)
    print("PART 2: Optimized (Incremental/Cached C4 Centroids + Direct 5D Shortlist Gather)")
    print(f"{'Context':>8} | {'Blocks':>6} | {'Score+TopK':>10} | {'5D Gather':>10} | {'Exact Rerank':>12} | {'Opt Total':>9} | {'Speedup':>8}")
    print("-" * 85)

    for n_ctx in contexts:
        nb = n_ctx // L
        kb = torch.randn(B, G, nb, L, D)
        q_gqa = torch.randn(B, G, R, t, D)

        # Precompute cached C4 centroids as in incremental maintenance
        cen = kb.mean(dim=3)
        cen_c4 = TwoLevelBlockIndex.quantize_c4(cen, group_size=32)

        iters = 30
        t_score, t_gather, t_rerank = 0.0, 0.0, 0.0

        for _ in range(iters):
            # 1. Score + topk directly from cached cen_c4
            t0 = time.perf_counter()
            c4_scores = torch.einsum("bgrtd,bgnd->bgrtn", q_gqa, cen_c4) * scale
            sc = c4_scores.amax(dim=2)
            short_idx = sc.topk(shortlist_size, dim=-1).indices
            t1 = time.perf_counter()
            t_score += (t1 - t0)

            # 2. Optimized 5D gather for t=1 (avoids 6D tensor expansion)
            if t == 1:
                gather_idx = short_idx[:, :, 0].unsqueeze(-1).unsqueeze(-1).expand(B, G, shortlist_size, L, D)
                kb_short = torch.gather(kb, 2, gather_idx).unsqueeze(2)
            else:
                idx_exp = short_idx.unsqueeze(-1).unsqueeze(-1).expand(B, G, t, shortlist_size, L, D)
                kb_exp = kb.unsqueeze(2).expand(B, G, t, nb, L, D)
                kb_short = torch.gather(kb_exp, 3, idx_exp)
            t2 = time.perf_counter()
            t_gather += (t2 - t1)

            # 3. Exact rerank
            dot_short = torch.einsum("bgrtd,bgtmld->bgrtml", q_gqa, kb_short) * scale
            tok_max = dot_short.amax(dim=-1).amax(dim=2)
            pick = tok_max.topk(top_k, dim=-1).indices
            top_blocks = torch.gather(short_idx, dim=-1, index=pick)
            t3 = time.perf_counter()
            t_rerank += (t3 - t2)

        t_score = (t_score / iters) * 1000.0
        t_gather = (t_gather / iters) * 1000.0
        t_rerank = (t_rerank / iters) * 1000.0
        opt_total = t_score + t_gather + t_rerank
        speedup = baseline_totals[n_ctx] / opt_total

        print(f"{n_ctx:>8} | {nb:>6} | {t_score:>9.3f}ms | {t_gather:>9.3f}ms | {t_rerank:>11.3f}ms | {opt_total:>8.3f}ms | {speedup:>7.2f}x")

    print("=" * 85)

if __name__ == "__main__":
    profile_c4_selection()
