"""
bench/gjallar_gdn_gate.py — Offline gate feature evaluation on REAL GDN captures.

Evaluates predictive AUROC of recurrent observable ||o_gdn|| from Gated DeltaNet layers (layers 2, 6, 10...),
paired with downstream SGRA attention layers (layers 3, 7, 11...).

Per GPT's hardware finding: GDN and Attention use different head counts and query projections.
The observable that carries the recurrent state confidence is the pre-normalization GDN output
cal-<kind>-gdn-out-<gdn_layer> [STEPS, native_heads, native_dim], reduced over native_dim and heads.

Usage:
    python bench/gjallar_gdn_gate.py <capture_dir> <attn_layer> [<attn_layer> ...]
"""

import sys
from pathlib import Path
import numpy as np

DEPTH, STEPS, WIN, SINKS = 32768, 256, 512, 4
SCALE = 1.0 / 16.0


def load_gdn_capture(root: Path, kind: str, attn_layer: int):
    root = Path(root)
    gdn_layer = attn_layer - 1  # e.g., layer 2 pairs with layer 3
    q_path = root / f"cal-{kind}-q-{attn_layer}"
    k_path = root / f"cal-{kind}-k-{attn_layer}"
    gdn_path = root / f"cal-{kind}-gdn-out-{gdn_layer}"
    s_path = root / f"cal-{kind}-s-{gdn_layer}"

    missing = []
    if not q_path.exists():
        missing.append(str(q_path))
    if not k_path.exists():
        missing.append(str(k_path))
    if not gdn_path.exists() and not s_path.exists():
        missing.append(f"{gdn_path} or {s_path}")

    if missing:
        raise FileNotFoundError(
            f"Missing required real GDN capture file(s): {', '.join(missing)}.\n"
            f"Run the Vulkan/llama.cpp GDN capture harness to generate cal-{kind}-gdn-out-{gdn_layer} "
            f"for attention layer {attn_layer} before running this gate evaluation."
        )

    n = DEPTH + STEPS
    q = np.fromfile(q_path, "<f4").reshape(STEPS, 2, 4, 256)
    k = np.fromfile(k_path, "<f2").reshape(-1, 2, 256)[:n].astype("f4")

    if gdn_path.exists():
        # Pre-normalization GDN output: [STEPS, native_heads, native_dim]
        raw = np.fromfile(gdn_path, "<f4")
        # Infer dimensions: STEPS is fixed at 256
        rem = len(raw) // STEPS
        gdn_out = raw.reshape(STEPS, rem)
        # Reduced norm per step across native dimensions
        gdn_norm = np.linalg.norm(gdn_out, axis=-1)
    else:
        # Fallback to state matrix if present
        S = np.fromfile(s_path, "<f4").reshape(STEPS, 2, 256, 256)
        gdn_norm = np.zeros(STEPS)
        for s in range(STEPS):
            gdn_norm[s] = np.linalg.norm(S[s])

    return q, k, gdn_norm


def evaluate_gdn_gate(q, k, gdn_norm, stride=2):
    """
    Computes AUROC of ||o_gdn||, -LSE_win, and combined features predicting
    top-quartile full-attention vs window-attention output error.
    """
    steps = list(range(0, STEPS, stride))
    gdn_norms = []
    lse_win = []
    err_rel = []

    for s in steps:
        pos = DEPTH + s
        norm_s = gdn_norm[s]
        for g in range(2):
            qs = q[s, g]  # [4, 256]
            gdn_norms.extend([norm_s] * 4)

            # Window LSE
            a = qs @ k[: pos + 1, g].T * SCALE  # [4, pos+1]
            tok = np.arange(pos + 1)
            win = (tok < SINKS) | (tok >= pos - WIN + 1)
            aw = np.where(win, a, -np.inf)
            wmax = aw.max(-1, keepdims=True)
            ew = np.exp(aw - wmax)
            sw = ew.sum(-1, keepdims=True)
            lse = (wmax + np.log(sw))[:, 0]
            lse_win.extend(lse.tolist())

            # Full vs window output difference
            pf = np.exp(a - a.max(-1, keepdims=True))
            pf /= pf.sum(-1, keepdims=True)
            y_diff = np.linalg.norm(pf[:, win] - (ew / sw)[:, win], axis=-1)
            err_rel.extend(y_diff.tolist())

    err_arr = np.array(err_rel)
    thresh = np.percentile(err_arr, 75)
    labels = err_arr > thresh

    def auroc(score, target):
        pos = score[target]
        neg = score[~target]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))

    auroc_gdn = auroc(-np.array(gdn_norms), labels)
    auroc_lse = auroc(-np.array(lse_win), labels)
    return {"auroc_gdn": auroc_gdn, "auroc_lse": auroc_lse}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])
    layers = [int(x) for x in sys.argv[2:]]
    for layer in layers:
        for kind in ("active", "archive"):
            try:
                q, k, gdn_norm = load_gdn_capture(root, kind, layer)
                res = evaluate_gdn_gate(q, k, gdn_norm)
                print(f"Layer {layer} ({kind}): AUROC(-||o_gdn||) = {res['auroc_gdn']:.3f}, AUROC(-LSE_win) = {res['auroc_lse']:.3f}")
            except FileNotFoundError as e:
                print(f"Layer {layer} ({kind}): {e}")
