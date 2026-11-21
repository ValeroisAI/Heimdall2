# Gjallar — working plan (draft v0, 2026-11-20)

## Idea
Hybrid of Gated DeltaNet (fixed-size memory) and **surprise-gated retrieval attention**.
Every delta-rule layer already computes a prediction error `v − S·k`. Instead of reading exact KV blocks
for every token in the global layers (current `BIFROST_RET`), read them only when the memory is "unsure"
(low read-out confidence / high delta error). Keys/values live in host RAM; GPU keeps only block summaries
(min/max per 64-token block), the window and the state → GPU memory O(1), context bounded by RAM/disk.
FFN: latent-free low-bit (BitFlip). Goal: one claim that is easy to share —
*"1M-token context on one 16 GB card, memory flat, and the model decides when to look back."*

## Known / inferred / unknown
- Known (repo README): 3:1 GDN:attention hybrid, Bifrost conversion (+2.9 % ppl at 12/4), block-bound
  retrieval attention (CPU, 0.8B, exact softmax over picked tokens), BitFlip 1-bit FFN loses ~0.02 nats at 100M.
- Known (search, unverified beyond abstracts): Kimi Linear (2510.26692, 3:1), Qwen3.8-Next GDN hybrid (2608.30320),
  KVBuffer v3 (2605.19049), HybridSpec (2611.04055), ternary: Sherry, RoleQ, LittleBit-2, ParetoQ.
- Unknown: whether surprise-gated retrieval is already published (Antigravity checking); whether the gate works in a trained model.

## Experiment 0 (done, `bench/gjallar_surprise.py`, toy, CPU, no training)
Delta-rule memory d=64, random unit keys. Findings:
1. With **unselective** writes (beta=1 on every filler token) memory is fully overwritten after 256 filler tokens
   (100 % failure at every load) → the learned write gate/decay is essential, not optional.
2. Where recall is partly working (no filler), read-out norm `‖S q‖` separates failures from successes:
   AUROC 0.65–0.94 (best at low load / decay 0.999, worst at 128 pairs in d=64).
Confidence: low (toy). It supports trying the gate; it does not prove it. A trained model on MQAR is the real test.

## Experiment 1 (done, `bench/gjallar_mqar.py`, CPU, 1 seed, 1500 steps, `D=16 PAIRS=48`, lam=0.1, train gap ≤64)
Window 16, block 8, top-4 blocks, exact block scores (toy), gate features [‖S·q‖, window LSE].

| mode | gap 16 | gap 64 | gap 256 | gap 512 |
|---|---|---|---|---|
| delta memory only | 0.46 | 0.46 | 0.46 | 0.46 |
| always retrieve | 1.00 | 1.00 | 1.00 | 1.00 |
| gated (reads at all eligible tokens) | 0.31 (10 %) | 0.64 (17 %) | 0.90 (13 %) | 0.92 (46 %) |
| random gate, same overall read rate | 0.17 | 0.23 | 0.21 | 0.50 |

Earlier invalid/negative runs, kept for honesty: (a) no key shift → 14–17 % everywhere (bug); (b) `D=64 PAIRS=8` and
`D=32 PAIRS=40`: memory alone already 100 % / 98 %, so retrieval is unneeded there.
Reading: the gate beats a random gate at equal read rate and beats memory-only at long gaps, **but** it does not
match always-retrieve, is worse than memory-only at gap 16 (soft-trained, hard-evaluated mismatch), and its read
rate grows with gap (46 % at 512). Unproven: that it ranks *which query* needs retrieval rather than just *which
token is a query* (GPT's point). Needed: oracle schedule (read only at known query tokens), mixed short/long
recall in one sequence, straight-through hard gate in training, several seeds.

### Experiment 1b — oracle + straight-through gate, 3 seeds (same setting; seeds 0,1,2)
| mode | accuracy (gap 16/64/256/512) | reads over all eligible tokens (gap 16/64/256/512) |
|---|---|---|
| oracle (read only at the known query tokens) | 1.00 on every seed and gap | 0.35 / 0.26 / 0.13 / 0.08 |
| gated, straight-through | 0.99–1.00 on every seed and gap | seed 0: 0.61/0.45/0.22/0.13 · seed 1: 0.68/0.50/0.25/0.15 · seed 2: 0.88/0.65/0.32/0.19 |
| random gate, same read rate | 0.21–0.89 | same as gated |

Reading: the straight-through gate fixes the train/eval mismatch and reaches oracle accuracy, but it spends ~1.7–2.5× the
oracle's reads, and reads@query = 1.00: it learned "this token is a query", not "this query needs the archive". The
experiment cannot yet show that the delta-error signal ranks queries by need, because every query here needs retrieval.
Next test: mixed sequences where some queries are answerable from window/memory and some are not, and compare read rate
on the two groups. (Seeds 3–4 by GPT, frozen snapshot, may be added.) Single toy layer, CPU; no claim about real models.

### Experiment 1c — mixed easy/hard queries (`MIXED=1`, seeds 0–2) — INCONCLUSIVE
oracle-hard (read only outside-window queries): 1.00 on every seed/gap. ST-gated: accuracy 0.99; reads on easy queries 0.00 on seeds 0
and 1 but 1.00 on seed 2; hard queries 0.99–1.00. Weak test: only ~2 % of queries are "easy" and they cluster at the start of the
query suffix, so a position shortcut explains seeds 0–1. GPT is running a balanced control (one fresh pair before every query,
16 local / 16 archived, shuffled). Seeds 3–4 of Exp1b (GPT): same as seeds 0–2 (acc 1.00, reads@query 0.99–1.00, all-eligible 19–90 %).

### Experiment 1d — balanced control (GPT's snapshot, seed 0, 1500 steps; 16 local / 16 archived queries per sequence, shuffled)
| | acc | reads on local (easy) queries | reads on archived (hard) queries | reads over all eligible tokens (gap 16→512) |
|---|---|---|---|---|
| oracle-hard (ideal) | 1.00 | 0.00 | 1.00 | 4.4 % → 1.9 % |
| shuffled oracle labels, same read count | 0.76 | 0.50 | 0.50 | same as oracle |
| ST-gated | 1.00 | **0.81** | 1.00 | 43.8 % → 18.4 % |
| random gate, same overall rate | 0.61–0.73 | – | – | same as gated |

Reading (1 seed, toy): the gate opens on all archived queries and still on 81 % of local ones, so it is mostly a query detector
with weak need ranking (1.00 vs 0.81) and spends ~4–10× the oracle's reads. The recurrent-confidence signal alone is **not** shown to
rank queries by need in this toy. Real-Qwen features (below) are more promising than the toy gate. Do not publish "surprise gating
solves it"; publishable today: oracle needs 2–4 % reads for 100 % here, the learned gate has not got there.

### Experiment 1e — balanced control (`MIXED=2`, own implementation in the repo script) with the reservoir feature, 3 seeds
`D=16 PAIRS=48`, 1500 steps, lam=0.1, ST gate. RES = reservoir size for the third gate feature `log(N/R)+LSE(R sampled old scores)−LSE(window)`.

| gate features | acc (all gaps) | reads on easy (local) queries | reads on hard (archived) queries | reads over all eligible tokens (gap 16→512) |
|---|---|---|---|---|
| ‖S·q‖ + window LSE (RES=0), seeds 0–2 | 0.98–0.99 | **1.00** | 1.00 | 1.00 (gate saturated open) |
| + reservoir 32 (RES=32), seed 0 | 0.99–1.00 | **0.02–0.04** | 0.99–1.00 | 59 % → 16 % |
| + reservoir 32, seed 1 | 1.00 | 0.02–0.07 | 0.99–1.00 | 60 % → 21 % |
| + reservoir 32, seed 2 | 0.99–1.00 | 0.04–0.10 | 0.99–1.00 | 60 % → 55 % |

Reading (toy, 3 seeds): the reservoir feature is what makes the gate rank queries by need. Without it the gate opens on everything;
with 32 sampled keys it reads ~97 % fewer easy queries at the same accuracy, matching the oracle's easy/hard split (0 / 1.00).
It still opens on many non-query tokens (all-eligible 16–60 % vs oracle 2–4 %), so the budget claim is NOT yet at oracle level; seed 2 is
the worst (55 % at gap 512). Consistent with the real-Qwen AUROC gain (0.74 → 0.83). Still one toy layer, exact block scores,
no real-model task quality. Caveat to check: the penalty on non-query tokens (lam, schedule) was not tuned.

#### Exp1e-2 — budget-penalty sweep (RES=32, seed 0)
| lam | acc @ gap 16 / 512 | easy reads | hard reads | all-eligible reads (gap 16 → 512) |
|---|---|---|---|---|
| 0.1 | 1.00 / 1.00 | 0.02–0.04 | 0.99–1.00 | 59 % → 16 % |
| 0.3 | 0.99 / 0.99 | 0.00–0.01 | 0.98–1.00 | 41 % → 12.6 % |
| 1.0 | 0.96 / 0.99 | 0.00 | 0.93–0.99 | 27.5 % → **8.1 %** (oracle 4.4 % → 1.9 %) |
| 3.0 | 0.86 / 0.98 | 0.00 | 0.73–0.99 | 17.6 % → 6.2 % (accuracy drops at short gaps) |
→ lam≈1 is the knee: ~4× the oracle's reads at 0.99 accuracy for long gaps.
**Replicated at lam=1.0, seeds 1–3** (same setting): acc 0.95–0.96 at gap 16, 0.98–0.99 at gap 512; easy reads 0.00–0.01; hard reads 0.90–1.00;
all-eligible reads 25–27 % (gap 16) → 7.6–8.6 % (gap 512). Four seeds agree. (Command to reproduce:
`D=16 PAIRS=48 MIXED=2 RES=32 SEED=<s> python bench/gjallar_mqar.py 1500 1.0 gated`.)
Honest limits: toy single layer; exact block scores instead of the S8 index; gate sees a reservoir sampled from the same sequence;
short gaps (16) lose 4–5 points of accuracy at this lam; no real-model task result yet. Not yet claimable: "Gjallar works on LLMs".

### Experiment 1f — selector realism (`SEL=bound`: min/max box upper bound instead of exact block scores; `MIXED=2 RES=32`, lam=1.0)
| | acc | hard-query acc | easy reads | hard reads | all-eligible reads (gap 16 → 512) |
|---|---|---|---|---|---|
| always retrieve, exact selector (Exp1) | 1.00 | 1.00 | – | 1.00 | 100 % |
| always retrieve, min/max bound selector | **0.92** | 0.87 | 1.00 | 1.00 | 100 % |
| gated, min/max bound, seed 0 | 0.84–0.88 | 0.70–0.78 | 0.00 | 0.85–0.98 | 24 % → 7.5 % |
| gated, min/max bound, seed 1 | 0.82–0.86 | 0.66–0.74 | 0.00 | 0.85–0.99 | 24 % → 7.5 % |

Reading (toy, 2 seeds for gated, 1 for always): a loose block bound costs accuracy even with unconditional reads (hard queries 0.87),
and the gate inherits that ceiling (0.66–0.78). The gate's easy/hard split is unaffected (easy reads 0.00). So selector quality, not
the gate, is the limiting factor here — consistent with GPT's real-Qwen result (min/max pages top-8 retain only 52 % of flat top-16)
and the reason the S8 shortlist is the planned index. Not yet tested: the S8 selector inside the toy.

#### Exp1f-2 — other block summaries, always-retrieve, MIXED=2, seed 0 (`SEL=mean|meanbound`)
| selector | acc (gap 16 / 512) | hard-query acc |
|---|---|---|
| exact block scores | 1.00 (Exp1) | 1.00 |
| min/max box bound | 0.92 | 0.87 |
| centroid + 0.5·half-width·|q| (`meanbound`) | 0.91 | 0.84 |
| centroid only (`mean`) | 0.86 / 0.77 | 0.56–0.75 |

Centroid-only is the WORST here, which contradicts the synthetic-random-vector comparison in `bench/gjallar_bounds_compare.py`
(Antigravity: centroid/sphere 2–2.4× the box bound's top-16 recall, but absolute recall 0.07–0.09 on random vectors, i.e. near chance).
Lesson: summary ranking depends on the key distribution; neither result transfers. Decide on real Qwen keys with answer-bearing queries.
Also: GPT's "S8 768/768" measures agreement with the box-bound ranking (a quantized copy of the same min/max descriptor), not with
true attention, and Exp1f shows that ranking is lossy. Single seed, toy, block size 8.

### Experiment 2 — block selectors on REAL Qwen3.5-0.8B keys (`bench/gjallar_real_selectors.py`)
Data: GPT's captured q/k at 32K context (6 attention layers × {active, archive} source text, 64 query steps × 2 KV groups × 4 heads
per capture). 512 old blocks of 64 tokens, 16 selected. Metric = share of the old-region attention mass inside the selected blocks
(mass-weighted over steps/heads), **a direct answer-relevant proxy, not agreement with another heuristic**. Layer 3 / 11 / 23 examples:

| selector | L3 active | L11 active | L23 active | mean over 12 captures, as fraction of oracle (min–max) |
|---|---|---|---|---|
| oracle (true top-16 mass) | 0.436 | 0.855 | 0.551 | 1.00 |
| exact token-max (needs all keys) | 0.416 | 0.847 | 0.538 | 0.977 (0.954–0.991) |
| **centroid** (q·mean key, 1 vector/block) | 0.338 | 0.792 | 0.431 | **0.818 (0.719–0.928)** |
| min/max box bound (Quest; S8 is its quantized copy) | 0.236 | 0.609 | 0.239 | 0.608 (0.434–0.743) |
| centroid + radius (Cauchy–Schwarz sphere) | 0.185 | 0.341 | 0.125 | 0.442 (0.227–0.610) |
| random | 0.029 | 0.034 | 0.034 | 0.048 |

Reading: on real keys the centroid beats the box in all 12 captures and uses half the descriptor memory (D vs 2D); the sphere bound is
the worst (loose upper bound). My toy ranking (centroid worst) was wrong for real keys — toy keys are not Qwen keys. Antigravity's
"sphere 2× box" claim (random vectors) is also wrong on real keys; "centroid > box" is right. Gap to remember: centroid reaches only
~82 % of oracle mass (72–93 %), so a rerank over a shortlist of exact keys (token-max on the shortlist) is still needed.
One 0.8B checkpoint, one source text per capture, 32K context, mass proxy not task accuracy. Quantized centroid ("C8") not yet tested.

#### Experiment 2b — quantized centroids and shortlist + exact rerank (same data; layers 3 and 11, active/archive; mass captured)
| | L3 active | L3 archive | L11 active | L11 archive |
|---|---|---|---|---|
| oracle | 0.436 | 0.466 | 0.855 | 0.806 |
| centroid (fp32) | 0.338 | 0.335 | 0.792 | 0.697 |
| C8 (int8, groups of 32, fp scale) | 0.338 | 0.335 | 0.791 | 0.697 |
| C4 (int4, groups of 32) | 0.327 | 0.331 | 0.789 | 0.694 |
| C4 shortlist-64 → exact token-max rerank → 16 | 0.389 | 0.418 | 0.840 | 0.784 |
| C8 shortlist-64 → exact rerank → 16 | 0.391 | 0.421 | 0.840 | 0.783 |
| box bound (S8 is its quantized copy) | 0.236 | 0.231 | 0.609 | 0.586 |

Reading: quantizing the centroid to 4 bits costs ~1–3 % of the captured mass, and shortlist-64 + exact rerank recovers 90–98 % of the
oracle (L3 0.389/0.436 = 0.89 … L11 0.840/0.855 = 0.98) versus 0.54–0.71 for the box. Descriptor size by calculation: C4 ≈ 144 B per
64-token block per KV group (256×4 bit + 8 fp16 scales) vs S8 ≈ 576 B → **≈ 2.25 MiB vs 8.5 MiB per KV group per layer at 1M tokens (≈ 3.8× smaller)**.
Cost to be honest about: the rerank needs the exact keys of the 64 shortlisted blocks (64 × 64 tokens × 256 × 2 B ≈ 2 MiB per KV group per
layer per read) from host RAM, i.e. 4× the key traffic of reading only the final 16 blocks — a trade of PCIe bytes for index memory and
quality. Not measured in latency. Same 0.8B captures; 2 layers here (all 6 layers were run for the unquantized selectors above).

#### Experiment 2c — block-size sweep (same captures; read budget fixed at 1,024 old tokens, shortlist = 4,096 tokens' worth of blocks; `GJ_BLOCK=32|128`)
Layers 3 and 11, both texts. Ratio to the oracle mass (L3 active / L3 archive / L11 active / L11 archive):
| block | box bound | centroid | C4 centroid | C4 shortlist + exact rerank |
|---|---|---|---|---|
| 32 tokens | 0.54 / 0.57 / 0.80 / 0.78 | 0.78 / 0.77 / 0.96 / 0.89 | 0.77 / 0.77 / 0.95 / 0.89 | 0.91 / 0.93 / 0.99 / 0.98 |
| 128 tokens | 0.54 / 0.45 / 0.59 / 0.59 | 0.76 / 0.67 / 0.90 / 0.77 | 0.74 / 0.66 / 0.90 / 0.76 | 0.88 / 0.88 / 0.97 / 0.95 |
The ordering (centroid > box; C4 ≈ C8; shortlist + rerank closest to the oracle) holds at 32, 64 and 128 tokens per block; smaller
blocks select better but need a larger index (descriptor count doubles per halving). 4 captures per row, one model.

#### Reference-layer note (CPU, `bench/gjallar_gather_bench.py`)
`heimdall/gjallar.py` now gathers the selected blocks instead of masking the whole history. I re-ran the bench: attention over the already-selected
16 blocks takes 0.25–0.33 ms at 1K–64K tokens versus 0.45–9.0 ms for the masked variant (1.4× → 36× at 64K), max diff ≤ 7e-8, equivalence test
passes. Scope: CPU PyTorch reference, selected blocks given; block selection (still scores every descriptor) is not included. Not a Vulkan/GPU result.

### Experiment 3 — offline replay on real Qwen captures (`bench/gjallar_real_replay.py`, attention-output proxy)
Closed step = window 512 + 4 sinks. Open step = + 16 blocks picked by C4 shortlist-64 → exact rerank. One decision per layer and step
(all 8 heads together), 25 % of steps opened, 128 steps per capture (stride 2), 12 captures (6 layers × 2 texts). Error = ‖y−y_full‖/‖y_full‖.
Share of the (all-closed → all-open) error reduction recovered while opening only 25 % of the steps:

| schedule | mean | min–max over 12 captures |
|---|---|---|
| random | 0.253 | 0.242–0.261 |
| window-LSE gate (−LSE) | 0.307 | 0.193–0.442 |
| **reservoir gate (R=32)** | **0.379** | 0.309–0.478 |
| oracle (best 25 % of steps) | 0.469 | 0.419–0.551 |

Example errors (layer 15 archive): all-closed 0.616, all-open 0.094, random 25 % 0.482, LSE gate 0.515 (worse than random), reservoir gate 0.426, oracle 0.357.
Reading: on natural text the reservoir gate recovers ~1.5× the benefit of a random schedule at the same budget (and is better than
the window-LSE gate in 8 of 12 captures; 2 ties/near-ties at ±0.001 and 2 where LSE-only wins by 0.007–0.017), reaching ~80 % of what an oracle schedule could do — but the achievable benefit is modest:
retrieval need is spread over many steps, so 25 % of the reads buys ~38 % (oracle 47 %) of the error reduction. The big read savings seen
in MQAR (reads 8 % at 0.99 acc) come from needle-like tasks where need is concentrated; do not generalize them to ordinary text.
A ridge gate on [−LSE, reservoir] trained on the *other* text (cross-text, 128 steps) recovers 0.365 — no better than the raw
reservoir feature (0.379), so a linear combination of these two features adds nothing; the next gain has to come from new features
(e.g. recurrent-state confidence in a GDN layer, which these captures do not contain) or from the selector, not from re-weighting.
Also note all-open itself leaves error (16 blocks of 64 tokens ≈ 1,024 old tokens). Proxy only: no downstream NLL/QA; per-layer, one model.

### Experiment 4 — model-level replay and follow-ups (GPT's Vulkan/llama.cpp harness; this session)
**Shared vs per-head block lists** (`GJ_SHARE=1` in `bench/gjallar_real_replay.py`, one C4 shortlist per KV group via max over its 4 heads, as in
`heimdall/gjallar.py`): share of the closed→open error reduction at 25 % of steps — random 0.249, window-LSE 0.307, reservoir 0.381, oracle 0.479
(per-head: 0.253 / 0.307 / 0.379 / 0.469); all-open error 0.092 vs 0.082. Sharing costs almost nothing in quality.

**Model-level NLL** (GPT, Qwen3.5-0.8B, 32K prefill, archived source, 256 teacher-forced tokens; routes and block picks frozen from full-attention captures;
full route matches stock llama.cpp exactly). nats/token, lower is better:
| route | NLL | share of the window→full gap recovered |
|---|---|---|
| full attention | 0.5639 | 1.00 |
| window only | 1.1870 | 0.00 |
| always open, C4 shortlist + rerank | 0.6301 | 0.89 |
| random 25 % of steps | 1.0582 | 0.21 |
| window-LSE gate, 25 % | 1.0876 | 0.16 (worse than random) |
| **reservoir gate, 25 %** | **0.9529** | **0.38** |
So the layer-level proxy (0.38) is confirmed on actual next-token loss: the reservoir gate is the best 25 %-budget schedule, but it recovers well under half of
what always-open achieves; the C4 selector itself is good (0.89). **Active-source replication** (GPT): full 0.5406, window 1.4726, always-C4 0.5731, random 25 % 1.2321, window-LSE 25 % 1.0238, reservoir 25 % 0.9430 →
share of the gap recovered: always-C4 0.97, reservoir 0.57, window-LSE 0.48, random 0.26. Reservoir beats window-LSE on both texts (0.57 vs 0.48; 0.38 vs 0.16) and random on both,
but the margin over LSE is text-dependent. GPT's seed check: the reservoir schedule beat all five random schedules per corpus (random NLL 1.012–1.061 archived, 1.122–1.232 active, vs 0.953 / 0.943).
Two texts, one 0.8B checkpoint, 256 tokens each; schedules frozen from full-attention captures (offline), not an online gate.

**Causal (online-feasible) threshold** (GPT): the "top 25 %" schedules above rank steps over the whole continuation, which an online gate cannot do. Two causal variants:
a fixed reservoir threshold calibrated on the other corpus gave read rates of 37.0 % (archive) and 16.0 % (active) — it does not hold a budget; a threshold from the previous 64 scores
held 27.7 % / 25.5 % reads with NLL 0.9432 / 1.0028 (≈ 39 % / 50 % of the window→full gap), better than all five per-layer read-matched random controls on both texts (archive 0.9432 vs 0.9825–1.0949; active 1.0028 vs 1.1878–1.2602; reproducible via `--causal` in GPT's harness, see `bench/gjallar_vulkan.md`). Full repo pytest: 65 passed, 4 skipped (GPT). This is the first gate variant
that uses only past scores; it still replays frozen full-attention captures rather than running an online model.

**NEGATIVE: pass-key at 32K (GPT, depths 10 / 50 / 90 %, exact answer-prefix match, 3 trials of 6 answer tokens)**: full attention, always-open C4 and the fixed-budget
window-LSE gate got 3/3; window-only, random and the fixed-budget reservoir gate got 0/3 — the reservoir opened 0 of 36 layer-steps during each six-token answer. The reservoir
estimates archive mass from 32 sampled keys, so a single needle is almost never in the sample; it helps on diffuse long-range need (NLL) and misses sparse needles, which are exactly
the case window-LSE catches. A union gate (reservoir OR window-LSE) is being tested on all five inputs; until it passes pass-key and NLL together, no gate here is a general solution.

**Union gate (rolling reservoir OR rolling window-LSE; GPT)**: with both cutoffs at the 75th percentile the union costs 38–47 % reads; at the 87.5th percentile it uses 23.1 % (active) / 30.0 % (archive)
reads and 26.4–28.4 % on pass-key. Results: pass-key 3/3 at all depths (all 36 answer layer-steps opened); NLL 1.0625 (active) / 0.9744 (archive) → ≈ 44 % / 34 % of the window→full gap.
The rolling reservoir alone has lower NLL (1.0028 / 0.9432) but 0/3 on pass-key. Checked the cause: no reservoir sample fell inside the needle span at 10 % / 90 % depth; at 50 % one layer-3 sample did, yet the
fixed-budget reservoir still opened no answer step. So the union is the current candidate: it keeps the needle case (from window-LSE) at a ~23–30 % read rate and gives up ~0.06 nats of the reservoir's text gain.
A per-layer burst budget on the union (8 reads available for a burst, 56 metered over 256 steps) was tried: pass-key 3/3 at 24.0–24.3 % reads, source NLL 1.1361 (active) / 1.0304 (archive) — a hard 25 % cap with recall kept, at the price of higher text NLL than the uncapped union. Still offline replay of frozen captures, 0.8B, two texts, one pass-key prompt per depth.

**NEGATIVE: delayed pass-key (64 more answer-side steps; GPT)**: the capped union (8-read burst + 56 metered) got 0/3 exact answers (4/6, 5/6, 3/6 answer tokens) at 23.9–24.4 % reads,
opening only 18 of 36 answer layer-steps. Always-open C4, rolling window-LSE and the tighter uncapped union (87.5th percentile) got 3/3. So a hard per-layer budget starves the answer when
the needle query comes late, and its earlier 3/3 on the short pass-key was luck of timing. Capping window-LSE alone reached 2/3 exact at 20.1–20.4 % reads (it missed the 90 %-depth key). All three capped-union failures followed balances that were already depleted before the answer began; the uncapped union asked for 31–32 answer reads and the cap admitted 15–18. Current candidate:
the uncapped rolling union (≈ 23–30 % observed reads, but no hard guarantee), or window-LSE alone; not the capped union.

**Budget variants tried after the delayed-pass-key failure (GPT)**: (a) separate buckets — 16 reads reserved for window-LSE triggers, 48 for reservoir: 0/3 delayed keys recovered; the LSE bucket was already
empty when the answer came and denied 24/25, 22/26 and 21/23 answer-side LSE requests (window-LSE fires often during the filler delay, so a causal bucket drains before the needle query);
(b) borrowing between the two buckets: 1/3 exact (5/6, 6/6, 5/6 answer tokens) at 24.1–24.6 % reads, source NLL 1.0040 (active) / 0.9797 (archive) — better text than the plain cap (1.1361 / 1.0304).
(c) higher rolling quantiles for the reserved LSE bucket (Antigravity's hypothesis that q=0.75 wastes the burst on filler): priority LSE still requested 106–116 layer reads at q=0.90, 86–90 at q=0.95 and 68–69 at q=0.98 in the 64 delay steps,
while at q=0.98 only 3–6 of 36 answer layer-steps fired; all three cutoffs got 0/3 delayed pass-keys (q=0.95: 21.0–21.1 % reads, 4/6, 2/6, 4/6 answer tokens, 5/10, 4/8, 4/7 priority answer requests denied). The hypothesis is refuted:
the needle's window-LSE is not an outlier against the preceding 64 scores, so a stricter rolling threshold loses the needle before it saves budget.
No causal budget tried so far both holds ≈25 % reads and answers late needles; the unrestricted union does (3/3) but with no hard cap. The open problem is trigger precision during long filler, not the bucket size.

**Calibration fix (GPT)**: the q=0.98 cutoff had been calibrated on the source text, but the filler's first-64 scores sit 1.0–6.9 LSE units higher at every anchor, producing 32–33 of 68–69 pre-answer requests in the first 16 steps.
Calibrating the rolling threshold on same-context prefill history cut q=0.95 pre-answer LSE requests from 86–90 to 22–23 and raised the reserved-bucket route from 0/3 to **2/3** delayed pass-keys (6/6, 6/6, 5/6 tokens) at 19.7–20.2 % reads;
borrowing stayed 1/3 at ≈ 21 % reads. The needle's block was in the always-C4 selection in 69–72 of 72 answer group-steps, so the selector is fine — the failures are scheduling (when to open). Best causal budgeted gate so far: 2/3, not yet 3/3.

**Centroid-mass gate feature** (new, `bench/gjallar_real_replay.py`, 12 real captures, share of the closed→open error reduction at 25 % of steps): archive mass estimated from all C4 block centroids minus window-LSE —
random 0.253, window-LSE 0.307, reservoir (32 sampled keys) 0.379, **centroid-mass 0.413 (0.373–0.470)**, oracle 0.469. It beats the reservoir on average and in its worst capture (0.373 vs 0.309), but it needs a scan of
every block descriptor with per-head scores at every step (the GPU probe so far returns one max-over-heads score per row; per-head scores plus a log-sum-exp reduction have not been measured). **Model-level run (GPT, rolling 87.5th percentile, same-context prefill history)**: delayed pass-key 0/3 (5/6, 4/6, 5/6 answer tokens) at 14.7–15.6 % reads; source NLL 1.1006 (active, 14.8 % reads) and 0.9628 (archive, 17.7 %).
Read rates are well below 25 %, so this is not a matched-budget comparison yet; the cutoff is probably too high for the mass feature. Device scan with per-head scores plus a log-sum-exp reduction at 1M geometry: 205–208 µs versus
125–129 µs for a score-only scan (max LSE error 1e-6), i.e. ≈ 80 µs extra per layer per step on top of the scan itself.

**Budget-matched centroid-mass result (GPT)**: with the cutoff chosen so the completed trace reads exactly 25 %, centroid-mass recovered **all 3 delayed pass-keys** (answer NLL 0.0250 / 0.0396 / 0.0280) and gave source NLL
**0.9982 (active) / 0.8888 (archive)** — better than the matched reservoir (1.0019 / 0.9416) — i.e. ≈ 51 % / 48 % of the window→full gap. **Caveat (GPT, and it matters):** the cutoffs were fitted to the finished trace's read
count, which uses hindsight about how often the gate fires; a live gate must hold ≈ 25 % reads with a causal controller, which is untested. So this shows the centroid-mass *ranking* is the best feature so far, not that a
deployable 25 % gate exists. Costs: ≈ 205 µs per layer-step for the device scan with per-head scores and LSE at 1M geometry (vs 125 µs score-only), plus any host copy when the gate opens.

**Causal centroid-mass gate, no hindsight (GPT; thresholds from same-context prefill history only)**:
- fixed q = 0.75: delayed pass-key **3/3** at 23.9–25.4 % reads; on the source texts it read 27.7–28.8 % (slightly over a 25 % budget);
- adaptive per-layer controller (cutoff adjusted from past read counts, cap 64 reads per 256 steps): source reads 25.0 % / 24.7 % with NLL 1.0736 (active) / 0.8983 (archive), but only **2/3** delayed pass-keys at 23.2–24.0 % reads (the deepest key missed one token).
This is the first gate that is causal, near budget and finds late needles most of the time; remaining gaps: the controller costs one needle, the fixed rule overshoots the budget by ~3 points, and everything is still offline replay of
frozen captures on one 0.8B model with two short texts and three pass-key depths.

**Shared causal controller (GPT) — best result so far**: one 384-read cap shared across all layers, cutoff adjusted slowly from the combined past read rate and the prefill history, centroid-mass feature. Delayed pass-key **3/3** at 23.8 / 24.1 / 24.7 %
reads; both source texts held at **25.0 %** reads with NLL **1.0591 (active) / 0.9015 (archive)** (≈ 44 % / 46 % of the window→full gap; always-open is 0.57 / 0.63). Decisions use only past scores and counts, no cutoff fitted on the finished trace.
Still: scores and block picks come from frozen full-attention captures (not a live model with the gate in the loop), one 0.8B checkpoint, two short texts, three pass-key depths, gate cost ≈ 205 µs per layer-step plus host copies not included. Weak point found by GPT: the shared cap ran out at step 217 (active) / 221 (archive) of 256, so the last ~35–40 steps ran window-only — a late burst
of need (like a query arriving after a long stretch of ordinary text) is not covered by the 64-step pass-keys, and the cap is a hard budget, not a refill rate.

**Late-needle probe (GPT, answer at step 224 of 256, 90 % depth)**: the shared hard cap still answered 6/6 at exactly 25 % reads (30 of 36 answer layer-steps open) because the cap had not yet run out at that point;
a refill bucket (4 reads per layer to start, +1 per layer every 4 steps; at most ≈ 26.6 % over the run) used 19.5 % reads overall but admitted only 9 of 36 answer reads and got 3/6 — the refill rate throttled the burst. So a slow
refill is worse than the hard shared cap for a late needle at this position; a needle after the cap is already spent (> step ≈ 220) remains uncovered.

**Live gate inside the model scheduler (GPT; first non-frozen result)**: a custom GGML op runs after the current query and window logits are ready and makes the centroid-mass decision and the rerank from the *current* queries
(1,536 anchor decisions per active-text run; archive is an immutable prefill snapshot; closed reads are masked, so the physical KV fetch is not skipped). Active text NLL: gate forced closed 1.4740, forced open 0.5722,
adaptive (384-read shared cap) **1.0719**; archived text 0.9197 at 25 % reads. Pass-key: **4/4** — three at step 64 and the deepest key at step 224 — at 24.3–24.4 % reads (late key 25.0 %).
Remaining caveats: this masks reads rather than skipping the host copy and scan (so no speed result), numeric differences from the frozen controls were still being checked when reported, one 0.8B checkpoint, two short texts, four keys.

**Speed of the live probe (GPT, active source after prefill, two runs)**: frozen adaptive replay 203–205 tok/s; live adaptive **44.1 tok/s**; live with the gate forced closed 70–79 tok/s. The probe does CPU scoring, device↔host transfers,
synchronization and a fixed-shape gather every step, so this is a correctness prototype that is several times *slower* than the frozen path — there is no speed gain to claim from Gjallar yet; moving score + top-k + gather onto the device is the open engineering step.

**First optimization of the live probe (GPT)**: profile showed 2.34–2.36 s of CPU exact reranking per replay; a four-head SIMD rerank cut it to 1.15–1.16 s and raised live adaptive decode from 43.4–43.9 to 57.5–57.7 tok/s (≈ 1.3×),
reproducing all 1,536 decisions and 256 NLL rows byte-for-byte. Second step: reusing the packed heads for the centroid scan cut that stage from 0.94–0.95 s to 0.30 s per 256 tokens, giving 67.5–67.9 tok/s (1.55× the scalar 43.4–43.9), again byte-identical route traces and NLL rows.
Third step: reusing the decode graph (255 of 256 reused) gave 75.6 tok/s in both runs versus 54.8 and 66.6 tok/s with reuse disabled (that baseline is noisy, so the gain is ≈ 1.1–1.4×, not a clean figure); decisions and NLL rows identical.
Still ~2.7× below the frozen replay (203–205 tok/s); remaining cost is CPU scoring, transfers and synchronization. Complexity cost: a SIMD CPU kernel in the probe, not a device kernel.

**Synthetic multi-needle selector test (Antigravity, re-run by me; `bench/gjallar_multineedle.py`, 3 seeds, random keys, 16 blocks, selector recall only — not model quality)**: recall of the planted needle blocks
at 4K / 16K / 32K tokens — exact token-max 100 / 100 / 100 %, C4 shortlist + rerank 100 / 91.7 / 79.2 %, box bound 58.3 / 37.5 / 20.8 %, random 33.3 / 4.2 / 4.2 %. Direction agrees with the real-Qwen results (C4 + rerank ≫ box), but the C4 selector
degrades with context here (shortlist fixed at 64 blocks, so recall falls as the candidate count grows); that is a limit to watch at 128K–1M, where the real-key tests have not been run.

**GDN readout as a gate feature** (Antigravity, `bench/gjallar_gdn_gate.py`, 12 real captures of the *original* Qwen GDN layer outputs, not a Bifrost memory):
mean AUROC of −‖GDN output‖ for predicting where retrieval matters is 0.504 (0.462–0.558) — chance. Caveat: this is the unconverted model's GDN output norm, not the
‖S·q‖ read-out of a trained Bifrost memory, so it does not refute the recurrent-confidence idea for converted layers; it shows the raw norm carries no signal here.

**C4 Vulkan probe** (GPT): packed int4 centroid scoring matches CPU scores to 9.9e-5; at 1M-token geometry 4.5 MiB for two groups, 391 µs score + top-64
versus 818 µs for the S8 fp16 box variant; 64-block key uploads 193 µs (one group) / 289 µs (two groups) — repeat for variance pending.

### Real-model gate-feature check (GPT, Qwen3.5-0.8B layer 3, 32K, 64 tokens × 8 heads; attention-output proxy, not task accuracy)
- −window LSE predicts top-quartile full-vs-window output error: mean per-head AUROC 0.739 (heads 0.546–0.865).
- Actual outside-window softmax mass: 0.927 (but needs the archive scan).
- Reservoir estimate of archive mass (32 old keys, 32 KiB/layer): AUROC 0.830, recall at a 25 % read budget 58.3 % vs 48.4 %;
  128 keys: 0.865 / 62.3 %. → Candidate gate feature set: window LSE + reservoir mass estimate (+ ‖S·q‖).

## Prior art and caveats (from GPT)
- UT-ACA (2603.18446): token-uncertainty-driven context adjustment via rollback; HLA (2610.05842): query-dependent gates over GDN
  transitions. The Gjallar claim can only be the specific recurrent-feature, pre-emptive host-KV read decision at matched budget.
- AttentionIndex (2611.02508): contiguous two-stage selection loses recall as context grows at a fixed budget.
  HiSparse (2608.07009): bounded GPU cache + exact host fetch already exists. Gate must be tested atop these selectors.
- Measured by GPT (`bench/gjallar_index.py`, Qwen3.5-0.8B layer 3 at 131K): top-8 of 32 pages recovers only 56.5 % of the
  flat top-16 blocks (min/max bounds never prune). Our "4 MB at 1M" index is a size calculation, not a quality result.
- Better selector (GPT, all six Qwen3.5-0.8B attention layers, 131K, 768 samples, rank fidelity only — not latency or task
  accuracy): flat quantized leaf summaries + shortlist-64 + exact rerank. 4-bit: 99.93 % of flat top-16 retained; 8-bit: exact
  top-16 certified with 64 FP32 rerank candidates in 768/768 samples (4-bit needed a median 1,342 of 2,039 to certify).
  Adjacent layers overlap only 19–27 %, so don't share top-k lists across layers. Packed Vulkan shortlist kernel, 1M-token
  geometry (16,384 rows × 4 heads): Q8_0 8.5 MiB, 0.291 ms median. → Use S8 shortlist as the first index; min/max pages
  top-8 retained only 52 % of flat top-16. Note this index is O(N) summary memory (8.5 MiB per KV group per layer at 1M),
  so strict O(1) VRAM needs paging it too.
- Measured by GPT (Vulkan, RX 9070 XT): 24 MiB host→VRAM staging ≈ 1.0 ms; fixed BIFROST_RET decode 338→253 tok/s at 32K
  (fresh prefill), so the retrieval branch costs ~25 % when always on.
- Quest (min/max top-k blocks, ICML 2024), InfiniGen (host-RAM KV prefetch, OSDI 24), ECHO (OSDI 26) already cover
  block bounds and offload. **Novelty must be the read/no-read gate**, shown at matched fetch budget.
- Block summaries on GPU still grow with context: cap them (two-level summary) or keep the index on host.

## Next steps (ordered, each checkable)
1. Train small hybrid on `bench/mqar.py` with a learned gate on `‖S q‖`; metric: recall vs. fraction of blocks read.
2. Add gate to `retrieval attention` in PyTorch reference; compare to fixed top-k at equal read budget (0.8B pass-key).
3. Host-RAM KV: GPU holds summaries only; measure decode t/s at 32K–256K on the RX 9070 XT (Vulkan).
4. Port gate to llama.cpp patch only if 1–3 hold.
5. Publish figure + numbers; failures go in the README too.

## Environment
Windows 11, RX 9070 XT 16 GB, 32 GB RAM, Python 3.14, torch 2.13 **CPU-only** here (GPU training = ROCm path in README, Vulkan for inference).
