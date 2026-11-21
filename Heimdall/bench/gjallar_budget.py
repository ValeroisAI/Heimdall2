"""
Gjallar memory/bandwidth budget for the kept full-attention layers (by calculation, no GPU needed).

Assumptions taken from README (Qwen3.8-27B Bifrost 12/4): 4 kept layers, 4 KV heads, head_dim 256, window 2048 + 4 sinks.
Block = 64 tokens, summary = per-dim min and max of keys. Two-level index: superblock = 64 blocks (4096 tokens).

    python bench/gjallar_budget.py
"""
LAYERS, HKV, HD = 4, 4, 256
BLOCK, SUPER = 64, 64
WINDOW, SINKS = 2048, 4
KV_B = 2          # f16 bytes per element
SUM_B = 2


def mb(x):
    return x / 2**20


def row(ctx, kblocks=16, ksuper=8, gate_read_frac=1.0):
    tok_kv = 2 * HKV * HD * KV_B                      # bytes per token per layer (K+V)
    host_kv = ctx * tok_kv * LAYERS
    n_blocks = ctx // BLOCK
    n_super = max(1, n_blocks // SUPER)
    blk_sum = 2 * HKV * HD * SUM_B                    # min+max per block per layer
    flat_gpu = n_blocks * blk_sum * LAYERS            # one-level: all block summaries on GPU
    two_gpu = n_super * blk_sum * LAYERS              # two-level: only superblock summaries on GPU
    win_gpu = (WINDOW + SINKS) * tok_kv * LAYERS
    # per decode step traffic when a read is triggered (per layer): level-1 summaries of ksuper superblocks + kblocks of KV per head
    lvl1 = ksuper * SUPER * blk_sum
    kv_fetch = kblocks * BLOCK * tok_kv               # k blocks per KV head; tok_kv already spans all heads
    step = (lvl1 + kv_fetch) * LAYERS * gate_read_frac
    return host_kv, flat_gpu, two_gpu, win_gpu, step


if __name__ == "__main__":
    print(f"{'ctx':>9} {'host KV f16':>12} {'GPU flat idx':>13} {'GPU 2-lvl idx':>14} {'GPU window':>11} {'MB/step full':>13} {'MB/step gate30%':>16}")
    for ctx in (32 * 1024, 128 * 1024, 512 * 1024, 1024 * 1024, 4 * 1024 * 1024):
        h, f, t, w, s = row(ctx)
        _, _, _, _, s3 = row(ctx, gate_read_frac=0.3)
        print(f"{ctx:>9} {mb(h):>9.0f} MB {mb(f):>10.1f} MB {mb(t):>11.2f} MB {mb(w):>8.0f} MB {mb(s):>10.1f} MB {mb(s3):>13.1f} MB")
    print("\nPCIe 4.0 x16 ~ 25 GB/s usable -> 1 MB/step ~ 0.04 ms; compare: decode step at 33.7 t/s = 29.7 ms.")
