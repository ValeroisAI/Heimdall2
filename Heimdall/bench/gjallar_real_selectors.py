"""
Block-selector comparison on REAL Qwen3.5-0.8B keys (captured by the patched llama.cpp, Vulkan): which per-block descriptor
picks the 64-token blocks that actually hold the attention mass?

Input layout (same as the calibration captures): for layer L and kind in {active, archive}
    cal-<kind>-q-<L>  float32 [256 steps, 2 KV groups, 4 heads, 256]
    cal-<kind>-k-<L>  float16 [>=32768+256 tokens, 2 groups, 256]
Context 32768, window 512, sinks 4, softmax scale 1/16. Only the OLD region (outside window and sinks) is selectable.

Metric (answer-relevant, not agreement with another heuristic): fraction of the old-region attention mass that lies inside the 16
selected blocks, averaged over (step, head) weighted by that old-region mass.  Oracle = the 16 blocks with the most true mass.

    python bench/gjallar_real_selectors.py <capture dir> <layer> [<layer> ...]
"""
import sys
from pathlib import Path

import numpy as np

import os

DEPTH, STEPS, WIN, SINKS = 32768, 256, 512, 4
BLOCK = int(os.environ.get("GJ_BLOCK", 64))      # tokens per block; the read budget stays 1024 old tokens
TOPK = 1024 // BLOCK
SHORT_TOKENS = 4096                              # shortlist = 4096 tokens' worth of blocks (64 blocks at block size 64)
SCALE = 1 / 16


def load(root, kind, layer):
    n = DEPTH + STEPS
    q = np.fromfile(root / f"cal-{kind}-q-{layer}", "<f4").reshape(STEPS, 2, 4, 256)
    k = np.fromfile(root / f"cal-{kind}-k-{layer}", "<f2").reshape(-1, 2, 256)[:n].astype("f4")
    return q, k


SHORT = SHORT_TOKENS // BLOCK


def quant(x, levels):
    """Symmetric quantization in groups of 32 values with a per-group scale (levels=127 -> int8, 7 -> int4); returns dequantized x."""
    g = x.reshape(x.shape[0], -1, 32)
    scale = np.maximum(np.abs(g).max(-1, keepdims=True), 1e-12) / levels
    return (np.round(g / scale).clip(-levels, levels) * scale).reshape(x.shape)


def evaluate(q, k, step_stride=4, seed=0):
    rng = np.random.default_rng(seed)
    old = np.arange(SINKS, DEPTH - WIN + 1)
    nb = len(old) // BLOCK
    old = old[: nb * BLOCK]
    names = ["oracle", "token-max (exact flat)", "box bound (Quest)", "centroid", "centroid+radius", "random",
             "C8 (int8 centroid)", "C4 (int4 centroid)", "C8 shortlist-64 + exact rerank", "C4 shortlist-64 + exact rerank"]
    num = {n: 0.0 for n in names}
    den = 0.0
    for g in range(2):
        kg = k[old, g]                                             # [nb*BLOCK, 256]
        kb = kg.reshape(nb, BLOCK, 256)
        kmin, kmax, cen = kb.min(1), kb.max(1), kb.mean(1)
        rad = np.linalg.norm(kb - cen[:, None], axis=-1).max(1)
        for s in range(0, STEPS, step_stride):
            pos = DEPTH + s
            qs = q[s, g]                                           # [4, 256]
            kall = k[: pos + 1, g]
            a = qs @ kall.T * SCALE                                # [4, pos+1]
            p = np.exp(a - a.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            mass = p[:, old].reshape(4, nb, BLOCK).sum(-1)         # true mass per old block
            tok = (qs @ kg.T * SCALE).reshape(4, nb, BLOCK).max(-1)
            box = (np.maximum(qs, 0)[:, None] * kmax[None] + np.minimum(qs, 0)[:, None] * kmin[None]).sum(-1)
            ctr = qs @ cen.T
            sph = ctr + np.linalg.norm(qs, axis=-1, keepdims=True) * rad[None]
            c8, c4 = quant(cen, 127), quant(cen, 7)
            scores = {"oracle": mass, "token-max (exact flat)": tok, "box bound (Quest)": box, "centroid": ctr,
                      "centroid+radius": sph, "random": rng.random((4, nb)),
                      "C8 (int8 centroid)": qs @ c8.T, "C4 (int4 centroid)": qs @ c4.T}
            tot = mass.sum(-1)                                      # old-region mass per head
            for name, sc in scores.items():
                sel = np.argpartition(-sc, TOPK, axis=-1)[:, :TOPK]
                num[name] += np.take_along_axis(mass, sel, -1).sum()
            for name, sc in (("C8 shortlist-64 + exact rerank", scores["C8 (int8 centroid)"]),
                             ("C4 shortlist-64 + exact rerank", scores["C4 (int4 centroid)"])):
                short = np.argpartition(-sc, SHORT, axis=-1)[:, :SHORT]
                rr = np.take_along_axis(tok, short, -1)             # exact token-max score inside the shortlist only
                pick = np.take_along_axis(short, np.argpartition(-rr, TOPK, axis=-1)[:, :TOPK], -1)
                num[name] = num.get(name, 0.0) + np.take_along_axis(mass, pick, -1).sum()
            den += tot.sum()
    return {n: num[n] / den for n in names}


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for layer in sys.argv[2:]:
        for kind in ("active", "archive"):
            q, k = load(root, kind, layer)
            r = evaluate(q, k)
            print(f"layer {layer:>2} {kind:>7}: " + "  ".join(f"{n.split(' (')[0]} {v:.3f}" for n, v in r.items()), flush=True)
