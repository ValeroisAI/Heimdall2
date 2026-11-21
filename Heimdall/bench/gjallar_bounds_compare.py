"""
gjallar_bounds_compare.py — Empirical comparison of block summary descriptors for KV retrieval.

Compares:
  1. Exact: True max inner product per block: max_i (q · k_i)
  2. Box Bound (Quest/Bifrost): Σ_d [ max(0, q_d)*kmax_d + min(0, q_d)*kmin_d ] (Size: 2D)
  3. Centroid Only: q · μ (Size: D)
  4. Spherical Bound (Centroid + Max Radius): q · μ + ||q||*r (Size: D + 1)
  5. Mean-Sphere Score (Centroid + Mean Radius): q · μ + ||q||*r_mean (Size: D + 1)

Evaluates:
  - Top-16 block recall (relative to ground-truth exact top-16 blocks)
  - Bound tightness / gap: Mean(Bound - Exact)
  - Memory footprint per block (descriptors in bytes)
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)


def run_benchmark(n_blocks=1024, block_size=64, dim=128, n_queries=128, seed=42):
    torch.manual_seed(seed)

    # Simulate realistic key distribution: clustered semantics with local drift
    # Cluster centers + token variations
    n_clusters = max(1, n_blocks // 16)
    centers = F.normalize(torch.randn(n_clusters, dim), dim=-1)
    cluster_assign = torch.randint(0, n_clusters, (n_blocks,))

    keys = torch.empty(n_blocks, block_size, dim)
    for b in range(n_blocks):
        c = centers[cluster_assign[b]]
        # Mixture of cluster center + token-specific noise
        toks = c.unsqueeze(0) + 0.3 * torch.randn(block_size, dim)
        keys[b] = F.normalize(toks, dim=-1)

    queries = F.normalize(torch.randn(n_queries, dim), dim=-1)

    # 1. Precompute Block Summaries
    # (a) Min/Max Box (Quest / Bifrost)
    kmin = keys.amin(dim=1)  # [n_blocks, dim]
    kmax = keys.amax(dim=1)  # [n_blocks, dim]

    # (b) Centroid (Mean)
    mu = keys.mean(dim=1)    # [n_blocks, dim]

    # (c) Spherical Radii
    diff = keys - mu.unsqueeze(1) # [n_blocks, block_size, dim]
    dists = diff.norm(dim=-1)     # [n_blocks, block_size]
    r_max = dists.amax(dim=1)     # [n_blocks]
    r_mean = dists.mean(dim=1)    # [n_blocks]

    # Memory descriptors (float32 counts per block)
    mem_box = 2 * dim
    mem_centroid = dim
    mem_sphere = dim + 1

    # 2. Evaluate Queries
    recalls_box = []
    recalls_centroid = []
    recalls_sphere = []
    recalls_mean_sphere = []

    gaps_box = []
    gaps_sphere = []

    K = 16

    for q_idx in range(n_queries):
        q = queries[q_idx] # [dim]

        # Ground truth exact max per block
        # keys: [n_blocks, block_size, dim] @ q [dim] -> [n_blocks, block_size]
        exact_dots = torch.einsum("bld,d->bl", keys, q) # [n_blocks, block_size]
        exact_block_max = exact_dots.amax(dim=1)        # [n_blocks]
        top_exact = exact_block_max.topk(K).indices.tolist()
        set_exact = set(top_exact)

        # (a) Box Bound
        # pos*kmax + neg*kmin
        q_pos = F.relu(q)
        q_neg = -F.relu(-q)
        scores_box = (q_pos * kmax + q_neg * kmin).sum(dim=-1) # [n_blocks]
        top_box = scores_box.topk(K).indices.tolist()
        recalls_box.append(len(set_exact.intersection(top_box)) / K)
        gaps_box.append((scores_box - exact_block_max).mean().item())

        # (b) Centroid
        scores_centroid = (mu @ q) # [n_blocks]
        top_centroid = scores_centroid.topk(K).indices.tolist()
        recalls_centroid.append(len(set_exact.intersection(top_centroid)) / K)

        # (c) Spherical Bound (Centroid + Max Radius)
        scores_sphere = scores_centroid + r_max # [n_blocks]
        top_sphere = scores_sphere.topk(K).indices.tolist()
        recalls_sphere.append(len(set_exact.intersection(top_sphere)) / K)
        gaps_sphere.append((scores_sphere - exact_block_max).mean().item())

        # (d) Mean-Sphere Score
        scores_mean_sphere = scores_centroid + r_mean
        top_mean_sphere = scores_mean_sphere.topk(K).indices.tolist()
        recalls_mean_sphere.append(len(set_exact.intersection(top_mean_sphere)) / K)

    return {
        "dim": dim,
        "n_blocks": n_blocks,
        "recall_box": float(np.mean(recalls_box)),
        "recall_centroid": float(np.mean(recalls_centroid)),
        "recall_sphere": float(np.mean(recalls_sphere)),
        "recall_mean_sphere": float(np.mean(recalls_mean_sphere)),
        "gap_box": float(np.mean(gaps_box)),
        "gap_sphere": float(np.mean(gaps_sphere)),
        "mem_box_floats": mem_box,
        "mem_sphere_floats": mem_sphere,
        "mem_savings_pct": (1.0 - mem_sphere / mem_box) * 100.0,
    }


if __name__ == "__main__":
    print(f"{'Dim':<5} {'Desc Type':<16} {'Top-16 Recall':<15} {'Mean Bound Gap':<16} {'Size (B/blk)':<14} {'Savings':<8}")
    print("-" * 80)
    for dim in (64, 128, 256):
        res = run_benchmark(n_blocks=1024, block_size=64, dim=dim, n_queries=100)
        print(f"{dim:<5} {'Box (Quest)':<16} {res['recall_box']:<15.3f} {res['gap_box']:<16.3f} {res['mem_box_floats']*4:<14} {'0.0%':<8}")
        print(f"{dim:<5} {'Centroid':<16} {res['recall_centroid']:<15.3f} {'N/A (no bound)':<16} {dim*4:<14} {'50.0%':<8}")
        print(f"{dim:<5} {'Sphere (μ+r)':<16} {res['recall_sphere']:<15.3f} {res['gap_sphere']:<16.3f} {res['mem_sphere_floats']*4:<14} {res['mem_savings_pct']:<7.1f}%")
        print(f"{dim:<5} {'Mean-Sphere':<16} {res['recall_mean_sphere']:<15.3f} {'N/A (soft)':<16} {res['mem_sphere_floats']*4:<14} {res['mem_savings_pct']:<7.1f}%")
        print("-" * 80)
