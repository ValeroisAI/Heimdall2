"""
bench/gjallar_multineedle.py — Synthetic multi-needle selector recall benchmark.

NOTE: This benchmark uses synthetic random keys (selector recall only, not model quality).
Measures top-16 block recall of C4-shortlist+rerank vs min/max box-bound vs exact token-max
at a fixed 16-block budget across synthetic contexts (4K, 16K, 32K tokens) across 3 random seeds.
Also evaluates shortlist size scaling (32, 64, 128, 256) at 32K context.

Exact reproduction command: python bench/gjallar_multineedle.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn.functional as F
from heimdall.gjallar import TwoLevelBlockIndex

def run_multineedle_benchmark(seeds=(42, 43, 44)):
    B, H, G, D = 1, 4, 2, 64  # Multi-head GQA (H=4, G=2, R=2, D=64)
    R = H // G
    window = 512
    sinks = 4
    block_size = 64
    top_k = 16
    shortlist_size = 64
    scale = 1.0 / math.sqrt(D)

    contexts = [4096, 16384, 32768]
    num_needles = 8
    needle_depths = [0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95]

    print("=" * 95)
    print("SYNTHETIC MULTI-NEEDLE SELECTOR RECALL BENCHMARK")
    print(f"Scope: Selector recall only, synthetic random vectors, not model quality.")
    print(f"Seeds: {seeds} | Needles: {num_needles} across depths {needle_depths}")
    print(f"Geometry: H={H}, G={G}, R={R}, D={D}, Window={window}, Sinks={sinks}, BlockSize={block_size}")
    print(f"Fixed Retrieval Budget: top-{top_k} blocks (shortlist={shortlist_size})")
    print("=" * 95)

    for n_ctx in contexts:
        nb = (n_ctx - window - sinks) // block_size
        print(f"\n--- Context: {n_ctx:,} tokens ({nb} archive blocks) ---")

        recalls = {"exact": [], "c4": [], "box": [], "rand": []}

        for seed in seeds:
            torch.manual_seed(seed)
            k = torch.randn(B, G, n_ctx, D)
            k = F.normalize(k, dim=-1)

            archive_start = sinks
            archive_end = n_ctx - window
            archive_len = archive_end - archive_start

            needle_keys = []
            needle_block_indices = []

            for d_ratio in needle_depths:
                pos = archive_start + int(d_ratio * archive_len)
                blk_id = (pos - archive_start) // block_size
                exact_pos = archive_start + blk_id * block_size + (block_size // 2)

                n_k = torch.randn(1, G, 1, D)
                n_k = F.normalize(n_k, dim=-1)

                k[:, :, exact_pos:exact_pos + 1] = n_k
                needle_keys.append(n_k)
                needle_block_indices.append(blk_id)

            kb = k[:, :, sinks:sinks + nb * block_size].view(B, G, nb, block_size, D)

            hits_exact, hits_c4, hits_box, hits_rand = 0, 0, 0, 0

            for n_idx in range(num_needles):
                target_blk = needle_block_indices[n_idx]
                q = needle_keys[n_idx].repeat_interleave(R, dim=1)
                q_gqa = q.view(B, G, R, 1, D)

                # Exact token-max
                exact_dots = torch.einsum("bgrtd,bgnld->bgrtnl", q_gqa, kb) * scale
                exact_block_scores = exact_dots.amax(dim=-1).amax(dim=2)
                sel_exact = exact_block_scores.topk(min(top_k, nb), dim=-1).indices

                # C4 Shortlist-64 + Exact Rerank
                sel_c4 = TwoLevelBlockIndex.centroid_c4_select(
                    q_gqa, kb, scale=scale, top_k=top_k, shortlist_size=shortlist_size
                )

                # Min/Max Box Bound
                kmin = kb.amin(dim=3).view(B, G, 1, 1, nb, D)
                kmax = kb.amax(dim=3).view(B, G, 1, 1, nb, D)
                ub = TwoLevelBlockIndex.block_upper_bounds(q_gqa, kmin, kmax)
                sel_box = ub.topk(min(top_k, nb), dim=-1).indices

                # Random 16 blocks control
                rand_blks = torch.randperm(nb)[:top_k]

                if (sel_exact == target_blk).any().item(): hits_exact += 1
                if (sel_c4 == target_blk).any().item(): hits_c4 += 1
                if (sel_box == target_blk).any().item(): hits_box += 1
                if (rand_blks == target_blk).any().item(): hits_rand += 1

            recalls["exact"].append(hits_exact / num_needles)
            recalls["c4"].append(hits_c4 / num_needles)
            recalls["box"].append(hits_box / num_needles)
            recalls["rand"].append(hits_rand / num_needles)

        for name, label in [
            ("exact", "Exact Token-Max (Oracle)"),
            ("c4",    "C4 Shortlist+Rerank     "),
            ("box",   "Box-Bound (Quest-style) "),
            ("rand",  "Random 16 Blocks Control"),
        ]:
            vals = recalls[name]
            mean_rec = sum(vals) / len(vals)
            formatted_vals = ", ".join([f"{v:5.1%}" for v in vals])
            print(f"  {label}: Mean {mean_rec:5.1%}  (Seeds: {formatted_vals})")

    # Shortlist scaling sweep at 32K context
    print("\n" + "=" * 95)
    print("SHORTLIST SCALING SWEEP AT 32K CONTEXT (503 Archive Blocks, Top-16 Selected)")
    print(f"{'Shortlist Size':>15} | {'Fraction of Archive':>20} | {'Mean Recall':>15} | {'Per-Seed Recalls':>25}")
    print("-" * 95)

    n_ctx = 32768
    nb = (n_ctx - window - sinks) // block_size

    for sl_size in [32, 64, 128, 256]:
        sl_recalls = []
        for seed in seeds:
            torch.manual_seed(seed)
            k = torch.randn(B, G, n_ctx, D)
            k = F.normalize(k, dim=-1)

            archive_start = sinks
            archive_end = n_ctx - window
            archive_len = archive_end - archive_start

            needle_keys = []
            needle_block_indices = []

            for d_ratio in needle_depths:
                pos = archive_start + int(d_ratio * archive_len)
                blk_id = (pos - archive_start) // block_size
                exact_pos = archive_start + blk_id * block_size + (block_size // 2)

                n_k = torch.randn(1, G, 1, D)
                n_k = F.normalize(n_k, dim=-1)

                k[:, :, exact_pos:exact_pos + 1] = n_k
                needle_keys.append(n_k)
                needle_block_indices.append(blk_id)

            kb = k[:, :, sinks:sinks + nb * block_size].view(B, G, nb, block_size, D)

            hits = 0
            for n_idx in range(num_needles):
                target_blk = needle_block_indices[n_idx]
                q = needle_keys[n_idx].repeat_interleave(R, dim=1)
                q_gqa = q.view(B, G, R, 1, D)

                sel_c4 = TwoLevelBlockIndex.centroid_c4_select(
                    q_gqa, kb, scale=scale, top_k=top_k, shortlist_size=sl_size
                )
                if (sel_c4 == target_blk).any().item():
                    hits += 1
            sl_recalls.append(hits / num_needles)

        mean_sl = sum(sl_recalls) / len(sl_recalls)
        pct_archive = (sl_size / nb) * 100.0
        seed_str = ", ".join([f"{v:5.1%}" for v in sl_recalls])
        print(f"{sl_size:>15} | {pct_archive:>19.1f}% | {mean_sl:>14.1%}| {seed_str:>25}")

    print("=" * 95)
    print("Reproduction command: python bench/gjallar_multineedle.py")
    print("=" * 95)

if __name__ == "__main__":
    run_multineedle_benchmark(seeds=(42, 43, 44))
