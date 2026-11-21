"""
Gjallar experiment 1: MQAR with delta-rule memory + window + block retrieval, and a learned read/no-read gate.

Single layer, CPU. Modes:
  none   : window + delta memory only
  always : retrieval at every query token (top-k blocks, exact softmax)
  gated  : retrieval only if gate(features) > 0.5 at eval; soft gate + budget penalty in training
Features for the gate (observable before answering): ||S q||, window LSE.
Train with gap <= 64, test at 64/256/512.   python bench/gjallar_mqar.py [steps] [lam]
"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
NK, NV, FILL = 64, 64, 0
K0, V0 = 1, 1 + NK
VOCAB = 1 + NK + NV
import os
# Memory must be capacity-limited or retrieval is pointless: PAIRS > D. (D=64, PAIRS=8: delta memory alone scores 100 %.)
D, WIN, BLK, TOPK = int(os.environ.get("D", 32)), 16, 8, 4
PAIRS = int(os.environ.get("PAIRS", 40))


MIXED = int(os.environ.get("MIXED", 0))      # 1: RECENT extra pairs are written right before the queries (easy: inside the window)
RECENT = 4
SEL = os.environ.get("SEL", "exact")         # block selector: exact | bound (min/max box upper bound)
RES = int(os.environ.get("RES", 0))          # reservoir size for the third gate feature (0 = off)
import math


def batch(rng, B, gap):
    """Returns ids, targets, hard. hard[b,t] = query whose pair is further back than the window (needs retrieval/memory)."""
    ids, tg, hd = [], [], []
    for _ in range(B):
        if MIXED == 2:                                # balanced: each query is preceded by a fresh pair (easy) or a distractor (hard)
            keys = rng.choice(NK, NK, replace=False)
            old, spare = keys[:PAIRS], keys[PAIRS:]
            ov = rng.integers(NV, size=PAIRS)
            seq = [t for k, v in zip(old, ov) for t in (K0 + k, V0 + v)] + [FILL] * gap
            t, h = [-100] * len(seq), [False] * len(seq)
            kinds = rng.permutation([True] * 16 + [False] * 16)     # True = easy
            hard_idx = iter(rng.choice(PAIRS, 16, replace=False))
            for easy in kinds:
                sk, sv = spare[rng.integers(len(spare))], rng.integers(NV)
                seq += [K0 + sk, V0 + sv]; t += [-100, -100]; h += [False, False]
                if easy:
                    seq.append(K0 + sk); t.append(V0 + sv); h.append(False)
                else:
                    i = next(hard_idx); seq.append(K0 + old[i]); t.append(V0 + ov[i]); h.append(True)
            ids.append(seq); tg.append(t); hd.append(h)
            continue
        n_old = PAIRS
        keys = rng.choice(NK, PAIRS + (RECENT if MIXED else 0), replace=False)
        vals = rng.integers(NV, size=len(keys))
        seq = [t for k, v in zip(keys[:n_old], vals[:n_old]) for t in (K0 + k, V0 + v)] + [FILL] * gap
        vpos = {i: 2 * i + 1 for i in range(n_old)}
        for i in range(n_old, len(keys)):            # recent pairs, written after the gap
            vpos[i] = len(seq) + 1
            seq += [K0 + keys[i], V0 + vals[i]]
        t = [-100] * len(seq)
        h = [False] * len(seq)
        for i in rng.permutation(len(keys)):
            h.append(len(seq) - vpos[i] >= WIN); seq.append(K0 + keys[i]); t.append(V0 + vals[i])
        ids.append(seq); tg.append(t); hd.append(h)
    return torch.tensor(ids), torch.tensor(tg), torch.tensor(hd)


class Net(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(VOCAB, D)
        self.q, self.k, self.v = (nn.Linear(D, D, bias=False) for _ in range(3))
        self.beta = nn.Linear(D, 1)
        self.o = nn.Linear(D, D)
        self.out = nn.Linear(D, VOCAB)
        self.gate = nn.Linear(3 if RES else 2, 1)
        nn.init.constant_(self.beta.bias, 0.0)
        with torch.no_grad():
            self.k.weight.copy_(self.q.weight)   # start with q/k aligned so key matching is learnable quickly

    def forward(self, ids, hard=False, rand_p=None, hardmask=None):
        B, L = ids.shape
        x = self.emb(ids)
        q, k, v = F.normalize(self.q(x), dim=-1), F.normalize(self.k(x), dim=-1), self.v(x)
        k = F.pad(k, (0, 0, 1, 0))[:, :-1]      # key of position t comes from token t-1 (value is stored with its key token)
        beta = torch.sigmoid(self.beta(x))                         # write gate [B,L,1]
        S = x.new_zeros(B, D, D)
        outs, gates = [], []
        for t in range(L):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            omem = torch.einsum("bij,bj->bi", S, qt)
            lo = max(0, t - WIN + 1)
            kw, vw = k[:, lo:t + 1], v[:, lo:t + 1]
            sw = torch.einsum("bd,btd->bt", qt, kw) * 8
            lse = torch.logsumexp(sw, -1)
            owin = torch.einsum("bt,btd->bd", sw.softmax(-1), vw)
            o = omem + owin
            g = torch.zeros(B, 1)
            if self.mode != "none" and lo > 0:
                kp, vp = k[:, :lo], v[:, :lo]
                sp = torch.einsum("bd,btd->bt", qt, kp) * 8
                n = (lo // BLK) * BLK
                if n >= BLK:
                    if SEL == "bound":                                    # admissible min/max box bound per block (Quest-style)
                        kb = kp[:, :n].reshape(B, -1, BLK, D)
                        kmax, kmin = kb.amax(2), kb.amin(2)
                        bs = (F.relu(qt)[:, None] * kmax + (-F.relu(-qt))[:, None] * kmin).sum(-1)
                    elif SEL == "mean":                                   # one centroid key per block (cheapest summary)
                        bs = torch.einsum("bd,bnd->bn", qt, kp[:, :n].reshape(B, -1, BLK, D).mean(2))
                    elif SEL == "meanbound":                              # centroid score plus half-width of the box along q
                        kb = kp[:, :n].reshape(B, -1, BLK, D)
                        c, hw = (kb.amax(2) + kb.amin(2)) / 2, (kb.amax(2) - kb.amin(2)) / 2
                        bs = torch.einsum("bd,bnd->bn", qt, c) + torch.einsum("bd,bnd->bn", qt.abs(), hw) * 0.5
                    else:
                        bs = sp[:, :n].reshape(B, -1, BLK).max(-1).values   # exact block scores
                    kk = min(TOPK, bs.shape[1])
                    top = bs.topk(kk, -1).indices
                    mask = torch.zeros(B, n, dtype=torch.bool)
                    for j in range(kk):
                        idx = (top[:, j:j + 1] * BLK + torch.arange(BLK)[None])
                        mask.scatter_(1, idx, True)
                    spm = sp[:, :n].masked_fill(~mask, -1e9)
                    oret = torch.einsum("bt,btd->bd", spm.softmax(-1), vp[:, :n])
                    if self.mode == "always":
                        g = torch.ones(B, 1)
                    elif self.mode == "oracle":                      # reads exactly at the known query positions
                        g = torch.full((B, 1), float(t >= L - PAIRS - (RECENT if MIXED else 0)))
                    elif self.mode == "oraclehard":                  # reads only at queries that are really outside the window
                        g = hardmask[:, t:t + 1].float()
                    else:
                        fs = [omem.norm(dim=-1), lse]
                        if RES:                                       # reservoir estimate of archive softmax mass vs window
                            idx = torch.randint(0, lo, (min(RES, lo),))
                            res = math.log(lo / len(idx)) + torch.logsumexp(sp[:, idx], -1) - lse
                            fs.append(res)
                        feat = torch.stack(fs, -1)
                        gl = self.gate(feat)
                        soft = torch.sigmoid(gl)
                        hardg = (gl > 0).float()
                        # eval: hard; train: straight-through (hard forward, soft gradient) so train == eval
                        g = hardg if hard else hardg + soft - soft.detach()
                        if rand_p is not None:                      # control: random gate, same overall read rate
                            g = (torch.rand(B, 1) < rand_p).float()
                    o = o + g * oret
            gates.append(g)
            outs.append(o)
            # delta-rule write (gated)
            b = beta[:, t]
            err = vt - torch.einsum("bij,bj->bi", S, kt)
            S = S + b[:, :, None] * torch.einsum("bi,bj->bij", err, kt)
        h = torch.stack(outs, 1)
        return self.out(torch.relu(self.o(h))), torch.stack(gates, 1)


def evaluate(net, gap, n=64, seed=1, rand_p=None):
    """Returns accuracy, read fraction at query tokens, read fraction over ALL tokens eligible for retrieval."""
    rng = np.random.default_rng(seed)
    ids, tg, hd = batch(rng, n, gap)
    with torch.no_grad():
        lg, g = net(ids, hard=True, rand_p=rand_p, hardmask=hd)
    m = tg != -100
    ok = lg.argmax(-1) == tg
    acc = ok[m].float().mean().item()
    elig = torch.zeros_like(m); elig[:, WIN + BLK:] = True       # positions where retrieval is possible
    gg = g[:, :, 0]
    reads_q = gg[m].float().mean().item()
    reads_all = gg[elig].float().mean().item()
    if MIXED:                                                    # split by whether the answer is inside the window
        easy, hard_ = m & ~hd, m & hd
        return acc, reads_q, reads_all, dict(acc_easy=ok[easy].float().mean().item(), acc_hard=ok[hard_].float().mean().item(),
                                             rd_easy=gg[easy].mean().item(), rd_hard=gg[hard_].mean().item())
    return acc, reads_q, reads_all, {}


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["none", "always", "oracle", "gated"]
    seed = int(os.environ.get("SEED", 0))
    for mode in modes:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = Net(mode)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
        t0 = time.time()
        for s in range(steps):
            gap = int(rng.integers(0, 65))
            ids, tg, hd = batch(rng, 32, gap)
            lg, g = net(ids, hardmask=hd)
            m = tg != -100
            loss = F.cross_entropy(lg[m], tg[m])
            if mode == "gated" and s > steps // 3:                   # budget penalty after the model has learned to retrieve
                loss = loss + lam * g[:, :, 0][:, WIN + BLK:].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[{mode}] trained {steps} steps in {time.time()-t0:.0f}s, final loss {loss.item():.3f}", flush=True)
        for gap in (16, 64, 256, 512):
            acc, rq, ra, ex = evaluate(net, gap)
            line = f"   gap {gap:>4}: acc {acc:.2f}  reads@query {rq:.2f}  reads@all-eligible {ra:.3f}"
            if ex:
                line += "  | easy acc {acc_easy:.2f} reads {rd_easy:.2f}  hard acc {acc_hard:.2f} reads {rd_hard:.2f}".format(**ex)
            if mode == "gated":
                acc_r, _, ra_r, _ = evaluate(net, gap, rand_p=max(ra, 1e-3))
                line += f"  | random gate, same overall rate: acc {acc_r:.2f} (reads {ra_r:.3f})"
            print(line, flush=True)
