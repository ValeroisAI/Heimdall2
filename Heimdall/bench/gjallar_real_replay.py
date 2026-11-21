"""
Gjallar offline replay on REAL Qwen3.5-0.8B captures (attention-output level, one layer at a time, no model rerun):

  closed step : window (512) + 4 sinks only
  open step   : window + sinks + the 16 blocks chosen by "C4 shortlist-64 -> exact token-max rerank" (exact softmax over those tokens)
  error       : ||y - y_full|| / ||y_full||, y_full = full attention output, averaged over heads and steps

A schedule opens the layer on 25 % of the steps (one decision per layer and step, all heads together). Schedules compared at equal budget:
  all-closed, all-open (100 %), random 25 %, window-LSE gate (-LSE), reservoir gate (log(N/R)+LSE(R sampled old scores)-LSE, R=32),
  oracle (the 25 % of steps where opening reduces the error most).

This is a proxy for answer quality (attention-output error on one layer), not a task result.
    python bench/gjallar_real_replay.py <capture dir> <layer> [<layer> ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gjallar_real_selectors import DEPTH, STEPS, WIN, SINKS, BLOCK, TOPK, SHORT, SCALE, load, quant  # noqa: E402

BUDGET, R = 0.25, 32
import os
SHARE = int(os.environ.get("GJ_SHARE", 0))     # 1: one block list per KV group (max over heads) instead of per head


def run(q, k, v, stride=2, seed=0):
    rng = np.random.default_rng(seed)
    old = np.arange(SINKS, DEPTH - WIN + 1)
    nb = len(old) // BLOCK
    old = old[: nb * BLOCK]
    res_idx = rng.choice(old, size=R, replace=False)
    steps = list(range(0, STEPS, stride))
    e_closed = np.zeros(len(steps)); e_open = np.zeros(len(steps)); f_lse = np.zeros(len(steps)); f_res = np.zeros(len(steps))
    f_cen = np.zeros(len(steps))
    for g in range(2):
        cen = k[old, g].reshape(nb, BLOCK, 256).mean(1)
        c4 = quant(cen, 7)
        for si, s in enumerate(steps):
            pos = DEPTH + s
            qs = q[s, g]
            a = qs @ k[: pos + 1, g].T * SCALE                      # [4, pos+1]
            tok = np.arange(pos + 1)
            win = (tok < SINKS) | (tok >= pos - WIN + 1)
            pf = np.exp(a - a.max(-1, keepdims=True)); pf /= pf.sum(-1, keepdims=True)
            y_full = pf @ v[: pos + 1, g]
            aw = np.where(win, a, -np.inf)
            wmax = aw.max(-1, keepdims=True)
            ew = np.exp(aw - wmax); sw = ew.sum(-1, keepdims=True)
            y_win = (ew / sw) @ v[: pos + 1, g]
            lse = (wmax + np.log(sw))[:, 0]
            # open: choose 16 blocks per head (C4 shortlist-64 -> exact token-max rerank)
            tmax = a[:, old].reshape(4, nb, BLOCK).max(-1)
            sc = qs @ c4.T
            cm = (sc * SCALE).max(-1, keepdims=True)               # "centroid-mass" gate feature: archive mass estimated from ALL block centroids
            f_cen_step = (cm[:, 0] + np.log(np.exp(sc * SCALE - cm).sum(-1)) + np.log(BLOCK) - lse).mean() / 2
            f_cen[si] += f_cen_step
            if SHARE:                                              # one shortlist per KV group (max over its 4 heads), as in heimdall/gjallar.py
                sc = np.broadcast_to(sc.max(0, keepdims=True), sc.shape)
                tmax = np.broadcast_to(tmax.max(0, keepdims=True), tmax.shape)
            short = np.argpartition(-sc, SHORT, axis=-1)[:, :SHORT]
            rr = np.take_along_axis(tmax, short, -1)
            pick = np.take_along_axis(short, np.argpartition(-rr, TOPK, axis=-1)[:, :TOPK], -1)
            y_open = np.empty_like(y_win)
            for h in range(4):
                keep = win.copy()
                for b in pick[h]:
                    keep[old[b * BLOCK:(b + 1) * BLOCK]] = True
                ao = np.where(keep, a[h], -np.inf)
                eo = np.exp(ao - ao.max()); y_open[h] = (eo / eo.sum()) @ v[: pos + 1, g]
            nf = np.linalg.norm(y_full, axis=-1)
            e_closed[si] += (np.linalg.norm(y_win - y_full, axis=-1) / nf).mean() / 2
            e_open[si] += (np.linalg.norm(y_open - y_full, axis=-1) / nf).mean() / 2
            f_lse[si] += (-lse).mean() / 2
            samp = a[:, res_idx]
            peak = samp.max(-1, keepdims=True)
            est = peak[:, 0] + np.log(np.exp(samp - peak).sum(-1)) + np.log(len(old) / R)
            f_res[si] += (est - lse).mean() / 2
    return e_closed, e_open, f_lse, f_res, f_cen


def ridge_fit(x, y, lam=0.25):
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + lam * len(z) * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return lambda t: ((t - mu) / sd) @ w


def report(e_closed, e_open, f_lse, f_res, seed=0, learned=None, f_cen=None):
    """learned: optional per-step score from a ridge gate trained on the OTHER text (cross-text, no leakage)."""
    n = len(e_closed); m = max(1, int(round(BUDGET * n)))
    rng = np.random.default_rng(seed)

    def sched(score):
        opened = np.zeros(n, bool); opened[np.argsort(-score)[:m]] = True
        return np.where(opened, e_open, e_closed).mean()

    rand = np.mean([sched(rng.random(n)) for _ in range(200)])
    out = {"all-closed": e_closed.mean(), "all-open": e_open.mean(), "random 25%": rand, "window-LSE gate": sched(f_lse),
           "reservoir gate": sched(f_res)}
    if f_cen is not None:
        out["centroid-mass gate"] = sched(f_cen)
    if learned is not None:
        out["ridge(LSE,res) cross-text"] = sched(learned)
    out["oracle 25%"] = sched(e_closed - e_open)
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for layer in sys.argv[2:]:
        res = {}
        for kind in ("active", "archive"):
            q, k = load(root, kind, layer)
            v = np.fromfile(root / f"cal-{kind}-v-{layer}", "<f2").reshape(-1, 2, 256)[: DEPTH + STEPS].astype("f4")
            res[kind] = run(q, k, v)
        for kind, other in (("active", "archive"), ("archive", "active")):
            ec, eo, fl, fr, fc = res[other]                        # train on the other text: features -> error reduction
            fit = ridge_fit(np.stack([fl, fr], 1), ec - eo)
            ec, eo, fl, fr, fc = res[kind]
            r = report(ec, eo, fl, fr, learned=fit(np.stack([fl, fr], 1)), f_cen=fc)
            print(f"layer {layer:>2} {kind:>7}: " + "  ".join(f"{n} {x:.3f}" for n, x in r.items()), flush=True)
