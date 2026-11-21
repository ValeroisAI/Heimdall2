# Gjallar: Architecture Specification (v1.3)

**Long-Context Language Modeling via Surprise-Gated Sparse KV Retrieval and Host-RAM Offloading**

---

## 1. Executive Summary & Core Contribution

Modern long-context language models face a fundamental trade-off:
- **Full-Attention Transformers** provide exact in-context associative recall but suffer from $O(N)$ inference KV-cache memory explosion ($8\text{ GB}$ at $128\text{K}$ tokens for $27\text{B}$ models), exhausting consumer GPU VRAM.
- **Linear Attention and State Space Models (GDN, Mamba, RWKV)** achieve $O(1)$ memory and linear complexity, but suffer from "memory fog" and decaying multi-needle recall across long context horizons.
- **Sparse Attention & Offloading (Quest, InfiniGen, ECHO, DeepSeek NSA)** attempt top-$k$ block filtering or PCIe host streaming, but evaluate retrieval **unconditionally for every single token**, bottlenecking memory bus bandwidth and decode throughput.

**Gjallar** introduces a fundamentally new paradigm: **Surprise-Gated Retrieval Attention (SGRA)**. 
In Gjallar:
1. **Cheap Surprise Signal**: The recurrent Gated DeltaNet (GDN) layers already compute a prediction error $\delta_t = v_t - S_{t-1} k_t$ and readout confidence $\|S_t q_t\|$ during their standard forward pass, so the gate's inputs come at near-zero extra cost. **Status: the toy-scale gate does not yet rank queries by need** (see Experiment 1d in `GJALLAR_PLAN.md`).
2. **Event-Driven Retrieval Gate**: Instead of querying past KV blocks on every step, Gjallar dynamically branches execution. When the recurrent memory and local sliding window are confident (low surprise / high $\|S_t q_t\|$), the model executes in **Fast-Path (zero PCIe bus traffic, pure local compute)**. Only when surprise breaches a learned threshold does the layer invoke **Slow-Path**, fetching a small top-$k$ set of exact KV blocks from Host RAM.
3. **S8 Quantized Leaf Summaries & Shortlist Index**: Rather than relying on loose min/max bounding boxes (which empirically retain only $56.5\%$ of top blocks at $131\text{K}$), Gjallar uses flat **8-bit quantized leaf summaries with a shortlist-64 exact reranker**. Measured on Qwen3.5-0.8B (2 KV groups) at 1M-token geometry: **$\approx 8.5\text{ MiB}$ per KV group per layer** ($\approx 17\text{ MiB}$ per layer), shortlist kernel $\approx 0.29$–$0.36\text{ ms}$ per layer on the RX 9070 XT (Vulkan). The index is **O(N)** in context, not O(1). **By calculation** for the 27B target (4 KV heads, 4 kept layers) it is on the order of $\approx 136\text{ MiB}$ at 1M tokens plus the 2048-token window ($\approx 32\text{ MB}$); that figure is not measured on the 27B, and the 16 GB of exact KV stays in host RAM.
4. **Hardware Synergy**: Validated on single consumer workstations (AMD Radeon RX 9070 XT 16 GB + Ryzen 7 7800X3D) via Vulkan compute and low-bit (1-bit / 1.58-bit BitFlip) feed-forward representations.

---

## 2. Prior Art Boundary & Scope of Novelty

To ensure scientific rigor, we delineate existing prior art from Gjallar's specific contributions:
- **Block-bound pruning**: **Quest (ICML 2024)** introduced min/max coordinate bounding of query-key inner products for top-$k$ block selection.
- **Host-RAM KV Offloading**: **InfiniGen (OSDI 2024)** and **ECHO (OSDI 2026)** established speculative prefetching and host-to-device streaming of KV pages over PCIe.
- **Adaptive Context & Roller**: **UT-ACA (2026, 2603.18446)** performs token-uncertainty-driven context rollback; **HLA (2026, 2610.05842)** explores query-dependent gates over GDN state transitions.
- **Two-Stage Retrieval Bounds**: **AttentionIndex (2026, 2611.02508)** showed contiguous two-stage selection loses recall at scale; **HiSparse (2026, 2608.07009)** pairs bounded GPU caches with exact host fetch.
- **Neural Memory Updates**: **Titans (Google, 2024/2025)** explored surprise ($\nabla \ell$) as an internal weight update rule for neural MLPs at test time, without external KV caches or host offloading.

**The Novelty of Gjallar is specifically**:
1. Utilizing the recurrent delta-rule prediction error $\delta_t = v_t - S_{t-1} k_t$, associative readout norm $\|S_t q_t\|$, and reservoir archive mass estimates as a **pre-emptive read/no-read gating trigger** for sparse KV retrieval.
2. Operating the PCIe host-bus on an **event-driven basis**: reserving Host-RAM access strictly for queries that exceed local associative memory capacity.

---

## 3. System Architecture & Macro-Layout

Gjallar adopts a **3:1 interleaved macro-layout**: for every 3 Gated DeltaNet (recurrent) layers, 1 Surprise-Gated Retrieval Attention (SGRA) layer is placed.

```
Layer Pattern (Repeating Block of 4 Layers):
├── Layer 4i + 0 : Gated DeltaNet Layer       [Constant State: S ∈ ℝ^(d_k × d_v)]
├── Layer 4i + 1 : Gated DeltaNet Layer       [Constant State: S ∈ ℝ^(d_k × d_v)]
├── Layer 4i + 2 : Gated DeltaNet Layer       [Constant State: S ∈ ℝ^(d_k × d_v)]
└── Layer 4i + 3 : Surprise-Gated SGRA Layer  [Window + Delta Memory + Host-RAM Exact KV]
```

For a 64-layer model (such as Qwen3.8-27B or Bifrost 12/4):
- **48 Layers**: Pure Gated DeltaNet (fixed memory footprint).
- **12 Converted Layers**: Bifrost dual-branch layers (local window $W=2048$ + recurrent delta state).
- **4 Anchor SGRA Layers** (Layers 15, 31, 47, 63): Surprise-Gated Retrieval Attention with S8 shortlist host offloading.

---

## 4. Empirical Evaluation: The Surprise Gating Mechanism

### 4.1 Experiment 1d: Balanced Control on MQAR
To determine whether a learned gate distinguishes *which queries genuinely need retrieval* (archived needles) versus queries answerable locally (easy needles), we conducted a balanced control benchmark (16 local queries / 16 archived queries per sequence, shuffled, 1500 steps, straight-through hard gate):

| Configuration | Test Accuracy | Reads on Local (Easy) Queries | Reads on Archived (Hard) Queries | Reads Over All Eligible Tokens (Gap 16 $\to$ 512) |
|---|---|---|---|---|
| **Oracle-Hard (Ideal Target)** | **1.00** | **0.00** | **1.00** | **4.4% $\to$ 1.9%** |
| **Shuffled Oracle Labels** | 0.76 | 0.50 | 0.50 | 4.4% $\to$ 1.9% |
| **Learned ST-Gated** | **1.00** | **0.81** | **1.00** | **43.8% $\to$ 18.4%** |
| **Random Gate (Matched Budget)** | 0.61 – 0.73 | – | – | Same as Gated |

#### Analysis & Honest Findings:
- The learned straight-through gate successfully achieves **100% recall** across all test gaps (16 to 512) and outperforms random gating by $+27\text{ to }+39$ accuracy points at equal read rates.
- However, in this toy setting, the gate opens on **$100\%$ of archived queries and $81\%$ of local queries**, indicating it functions primarily as a *query-token detector* with weak difficulty ranking ($1.00$ vs $0.81$), spending $4\times \text{ to } 10\times$ more reads than the theoretical oracle budget ($1.9\%$).
- **Conclusion**: The recurrent-confidence signal $\|S_t q_t\|$ alone is insufficient to completely suppress reads on easy queries in toy models; richer contextual features are required for production scaling.
- **Follow-up (Experiments 1e and 3 in `GJALLAR_PLAN.md`)**: adding a reservoir estimate of archive attention mass (32 sampled old keys) as a third gate feature dropped the toy gate's reads on easy queries from $1.00$ to $0.00$–$0.10$ while hard-query reads stayed at $0.90$–$1.00$ (4 seeds at the best penalty; all-token reads $8\%$ at gap 512 versus $\approx 2\%$ for the oracle). On real Qwen3.5-0.8B captures (attention-output error, 25% of steps opened) the reservoir gate recovered $0.38$ of the achievable error reduction versus $0.25$ for random and $0.47$ for an oracle schedule, so the benefit on natural text is modest; the large savings are specific to needle-like retrieval tasks. **Model-level confirmation** (Qwen3.5-0.8B, 32K prefill, 256 teacher-forced tokens per text, offline schedules from full-attention captures): window-only NLL $1.19$ / $1.47$ nats, always-open with the C4 selector $0.63$ / $0.57$ (full attention $0.56$ / $0.54$), and with 25% of steps open the reservoir gate reached $0.95$ / $0.94$ versus $1.06$ / $1.23$ for a random schedule, recovering $38\%$ / $57\%$ of the window-to-full gap (archived / active text). Two short texts, one checkpoint, not an online gate. **Pass-key at 32K exposed a failure of the reservoir gate alone**: it opened no layer during the answer and scored 0/3 (window-only 0/3, window-LSE gate, always-open and full attention 3/3), because a 32-key sample almost never contains a single needle. A union of rolling reservoir and rolling window-LSE gates scored 3/3 at 23–30% reads, with NLL $1.06$ (active) / $0.97$ (archive) versus $1.00$ / $0.94$ for the reservoir alone, so the union is the current candidate gate. A variant with a hard per-layer read cap (8-read burst, rest metered) failed a delayed pass-key (0/3 exact; 3/6 to 5/6 answer tokens) because the cap starved the late answer, so only the uncapped rolling union (about 23–30% observed reads, no hard guarantee) or window-LSE alone remained candidates. **Current best gate: centroid mass** (archive mass estimated from all int4 block centroids minus window-LSE, thresholds from the prompt's own prefill history). Causal and fixed at the 75th percentile it recovered 3/3 delayed pass-keys at about 24–25% reads but read 27.7–28.8% on the text continuations; with an adaptive per-layer read controller it held 25.0%/24.7% reads, NLL $1.07$ (active) / $0.90$ (archive), and 2/3 delayed pass-keys. A live version of the gate inside the model's graph (custom op using the current queries, reads masked rather than skipped) kept 4/4 pass-keys, including one at step 224, at 24.3–25.0% reads, with NLL $1.07$ (active) / $0.92$ (archive). That prototype decoded at 43–44 tok/s and, after SIMD reranking, packed-head centroid scoring and decode-graph reuse (all byte-identical in routing and NLL), at about 76 tok/s, versus 203–205 tok/s for the frozen replay (CPU scoring, host transfers, synchronization remain), so it demonstrates quality only; no speed advantage is claimed. A controller that shares one 384-read cap across layers (cutoff adjusted from past read counts only) did best: 3/3 delayed pass-keys at 23.8–24.7% reads and 25.0% reads on both texts, NLL $1.06$ (active) / $0.90$ (archive). Cost: about 205 µs per layer-step for the device scan with per-head scores at 1M-token geometry. Offline replay of frozen captures, one 0.8B model, two short texts and three pass-key depths; not a deployed gate.

### 4.2 Real-Model Gate Feature Audit (Qwen3.5-0.8B Layer 3 at 32K Context)
Direct empirical evaluation on real Transformer representations reveals much stronger predictive signals:
1. **Negative Window LSE ($-\text{LSE}_{\text{win}}$)**: Predicts top-quartile full-attention vs window-attention output error with a mean per-head **AUROC of 0.739** (heads ranging from 0.546 to 0.865).
2. **Reservoir Estimate of Archive Mass**: Maintaining a tiny reservoir of past keys ($32\text{ to }128$ keys, occupying only $32\text{ KiB per layer}$) raised per-head AUROC for predicting large window-output changes from 0.739 (window LSE alone) to 0.830 (32 keys) / 0.865 (128 keys), one layer of one 0.8B model at 32K. At a $25\%$ read budget, recall of high-error positions rose from $48.4\%$ (window-LSE gate) to $58.3\%$ / $62.3\%$. Surrogate attention-output labels, not answer quality.
3. **Production Feature Vector**:
   $$\mathbf{f}_t = \Big[ \, \text{LSE}_{\text{win}}(q_t), \; \text{ReservoirMass}(q_t), \; \|S_t q_t\|_2 \, \Big]^T$$

---

## 5. Index Architecture: S8 Quantized Shortlist vs Min/Max Bounds

### 5.1 Failure Mode of Loose Min/Max Superblock Bounds
Early designs evaluated Quest-style min/max coordinate bounding boxes over $64$-token blocks and $4096$-token superblocks. Empirical testing on Qwen3.5-0.8B (Layer 3 at $131\text{K}$ tokens, `bench/gjallar_index.py`) revealed that:
- Top-8 of 32 superblock pages recovered **only 56.5% of the flat top-16 blocks**, because coordinate-wise bounding boxes over 256 dimensions expand rapidly, causing loose upper bounds that fail to prune irrelevant history.
- Consequently, pure min/max superblock bounding is insufficient for production retrieval fidelity.

### 5.2 The S8 Quantized Shortlist + FP32 Rerank Solution
To achieve exact retrieval fidelity with minimal VRAM:
1. **Flat S8 Quantized Leaf Summaries**: Each 64-token block key summary is quantized to 8-bit integers (Q8_0 format).
2. **Shortlist-64 Candidate Filtering**: The GPU executes a packed dot-product kernel to select the top 64 candidate blocks.
3. **Exact FP32 Reranking**: The top 64 candidates are evaluated in FP32 to certify the final top-16 blocks.
4. **Empirical Verification**: Evaluated across all six attention layers of Qwen3.5-0.8B at $131\text{K}$ context ($768$ samples), the S8 shortlist-64 index retained the exact flat top-16 inside its shortlist in 768/768 samples (one 0.8B checkpoint; rank agreement with flat block bounds, not answer quality). **Caveat (Experiment 2 in `GJALLAR_PLAN.md`):** the S8 descriptor is a quantized min/max box, and on real keys the box bound captures only ≈0.61× the attention mass of the true top-16 blocks, versus ≈0.82× for a plain per-block centroid (12 captures, 6 layers). The selector is therefore likely to move to a centroid shortlist plus exact rerank; the S8 figures above describe the current prototype, not the final design.

### 5.3 GPU Memory Footprint & Vulkan Kernel Performance
- **Packed Vulkan Shortlist Kernel**: On an AMD Radeon RX 9070 XT, processing a $1\text{M}$-token geometry ($16,384\text{ rows} \times 4\text{ heads}$) takes a median latency of **$0.291\text{ ms}$**.
- **Memory Footprint**:
  - S8 Leaf Summaries (GPU resident): **$\approx 8.5\text{ MiB per KV group per layer}$**, i.e. $\approx 17\text{ MiB}$ per layer for 2 KV groups at $1\text{M}$ context (0.8B measurement; multiply by kept layers and KV heads for other models).
  - Local Sliding Window (2048 tokens + 4 sinks): **$32.0\text{ MB}$**.
  - **GPU memory at 1M tokens, by calculation for the 27B target**: window $\approx 32\text{ MB}$ + S8 index $\approx 136\text{ MiB}$ (4 kept layers × 4 KV heads × 8.5 MiB) $\approx 170\text{ MiB}$, versus $\approx 16\text{ GB}$ of f16 KV for the 4 kept layers at 1M (not measured on the 27B).
  - Exact FP16 KV Cache (Host RAM resident): **$16,384\text{ MB}$ ($16\text{ GB}$)**.

### 5.4 C4 Quantized Centroid Index Scaling (The 3.8x Footprint Reduction)
To overcome the loose bounds of min/max descriptors (which capture only 61% of attention mass versus 82% for centroids on real keys), Gjallar implements **C4 quantized centroid indexing**:
1. **Int4 Group-Quantized Centroids**: Centroids $\bar{k} \in \mathbb{R}^D$ are quantized to symmetric 7-level int4 in groups of 32 values with FP16 scale factors ($\approx 144\text{ B}$ per 64-token block per KV group, versus $\approx 576\text{ B}$ for S8).
2. **GPU Memory Footprint**: At 1M context, C4 index memory shrinks to **$\approx 2.25\text{ MiB}$ per KV group per layer** ($\approx 3.8\times$ smaller than S8).
3. **27B Target Scale (calc)**: Across all 4 kept attention layers (4 KV heads each), the total GPU index memory at 1M tokens drops from $\approx 136\text{ MiB}$ (S8) to **$\approx 36\text{ MiB}$ (C4)**.
4. **Attention Mass Recovery**: Centroid C4 shortlist-64 followed by exact token-max rerank recovers **$\approx 89\%–99\%$ of oracle attention mass** (layers 3 and 11, block sizes 32/64/128) across real Qwen3.5-0.8B captures.

### 5.5 Block Gather vs O(N) Masked Scatter (reference layer, CPU)
In production and reference implementations, attention in the slow path is computed strictly over gathered candidate blocks:
- **Masked Scatter (Legacy Anti-Pattern)**: Allocating an $O(N)$ boolean mask across all $N$ tokens forces $O(N)$ memory allocations and full matrix multiplication against inactive tokens ($10.95\text{ ms}$ at 64K tokens on CPU PyTorch reference).
- **Block gather**: Only the $kk$ retrieved blocks ($16 \times 64 = 1,024$ tokens) are gathered into a compact tensor $[B, G, kk \cdot L, D]$, and softmax and value aggregation run over those tokens only. On the CPU PyTorch reference (`bench/gjallar_gather_bench.py`, selected blocks given) this step took $0.25$–$0.33\text{ ms}$ at 1K–64K tokens versus $0.45$–$9.0\text{ ms}$ for the masked variant ($1.4\times$ to about $36\times$ at 64K; max difference $\le 7\times10^{-8}$). The step is flat in $N$; block selection, which scores every descriptor, is $O(N/\text{block})$ and is not part of this timing. Not a GPU/Vulkan result.

---

## 6. GPU Kernel Profiling & Break-Even Read Rate Analysis

### 6.1 GPU Kernel Profiling Breakdown (Measured via Vulkan Timestamps on RX 9070 XT)
Timestamp profiling on the RX 9070 XT at 64K context for a single decode step reveals:
- Gathered f32 V multiplication: **$581\text{ µs}$** (median over 80 steps; one earlier single-token sample read 659 µs)
- Block summary pooling: **$372\text{ µs}$** (median; 411 µs in the single-token sample)
- Bound-scoring matmuls: **$272\text{ µs}$**
- Top-$k$ selection: **$18\text{ µs}$**
(Source: `bench/vulkan_baseline.md`; Qwen3.5-0.8B fixed-retrieval fixture, summed GPU time per step.)

**System Implication**: Kernel engineering priority focuses on **fused typed gather + attention** (which experimentally improves 64K decode from $250.8 \to 306.5\text{ tok/s}$, $+23\%$) and **incremental pooling**.

### 6.2 Break-Even Read Rate Analysis
Unconditional fixed sparse retrieval (`BIFROST_RET`) incurs a GPU dispatch overhead:
- At 32K context: Full attention runs at **$339.5\text{ tok/s}$** ($2.95\text{ ms/tok}$), while fixed retrieval runs at **$254.6\text{ tok/s}$** ($3.93\text{ ms/tok}$), adding $\approx 0.98\text{ ms}$ overhead per token.
- If retrieval is gated with an activation probability $p$:
  $$\text{Latency}(p) = 2.95\text{ ms} + p \times 0.98\text{ ms}$$
- **This is a model, not a measurement.** It assumes a gated-open token costs the full fixed-retrieval overhead and a closed token costs nothing extra, and that the gate itself is free. No runtime gate exists yet, so no gated decode speed has been measured. For $p = 0.15$ the model gives $\approx 3.10\text{ ms/tok}$ ($\approx 323\text{ tok/s}$ in the model) versus the measured $339.5$ (full attention) and $254.6$ (fixed retrieval) tok/s; the real number depends on gate cost, host copies (≈1 ms per 24 MiB, measured copy only) and branch overhead.

---

## 7. Feed-Forward Representation: BitFlip FFN & LoRA-8 Corrector

Empirical ablation on BitFlip (`bench/gjallar_bitffn.py`) demonstrates:
- Dense FFN: loss **0.0050**
- Binary BitFlip FFN: loss **0.0082**
- Ternary BitFlip FFN: loss **0.0099**
- **Ternary BitFlip FFN + LoRA-8 Rank Corrector: loss 0.0053** (recovering dense accuracy while saving ~62% training state memory at 6 bytes/param).

---

## 8. Summary of Gjallar Design Matrix

| Metric | Full Attention Baseline | Fixed Sparse Retrieval (`BIFROST_RET`) | **Gjallar (Surprise-Gated SGRA)** |
|---|---|---|---|
| **GPU cache at 1M (27B target, calc)** | $\approx 16\text{ GB}$ f16 KV for the 4 kept layers | Same KV on GPU | **$\approx 68\text{ MiB}$ ($32\text{ MB}$ win + $\approx 36\text{ MiB}$ C4 index) [$\approx 170\text{ MiB}$ with S8]** |
| **Decode speed at 32K (0.8B fixture, measured)** | $338$–$341\text{ tok/s}$ | $253$–$256\text{ tok/s}$ | **not measured** (no runtime gate exists yet) |
| **Host copy cost** | N/A | N/A | **$24\text{ MiB}$ pinned staging $\approx 1.0\text{ ms}$ (measured, copy only)** |
| **Long-context recall** | Exact | $6/6$ pass-key smoke (0.8B) | **not measured** (selector agreement only: 768/768 top-16 on 0.8B at 131K) |
