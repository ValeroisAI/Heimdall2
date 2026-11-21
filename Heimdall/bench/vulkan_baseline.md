# Windows Vulkan baseline

Measured November 20, 2026 on Windows 11 Pro, Ryzen 7 7800X3D, and RX 9070 XT with AMD driver 32.0.32029.2010. `llama-bench --list-devices` reported 16,304 MiB of device memory for `Vulkan0`, the discrete GPU. The Vulkan backend reported cooperative matrix support and `fp4: 0` for this driver and build.

- Runtime: [llama.cpp b12590](https://github.com/ggml-org/llama.cpp/releases/tag/b12590), commit `34ff2e047`. `llama-bench.exe` SHA-256 `23a6378fdf67337eef83bba2a301f6f1a97aa3db349bd6bd8b2e52e99c868137`.
- Model: [Qwen3.5-0.8B Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/blob/f36b1ea49a332ede8fe5f389bbf5b3575ef71f48/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf). SHA-256 `fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526`.
- Flash attention on, f16 K and V, one sequence, 8 CPU threads, 5 repetitions with warmup, batch 2048, microbatch 512. The log confirms 26/26 layers offloaded. It also reports two graph splits, so layer placement alone cannot establish that every operation ran on the device.
- The raw reports and profiles from this machine are saved in `Heimdall/runs/vulkan-20261120/`.

| Preloaded tokens | Prefill 512, tok/s | Generate 64, tok/s | Decode KV, MiB |
| ---: | ---: | ---: | ---: |
| 0 | 16,350 ± 1,145 | 448.2 ± 21.8 | 3 |
| 4,096 | 18,402 ± 885 | 429.6 ± 21.1 | 51 |
| 32,768 | 6,528 ± 64 | 282.1 ± 13.0 | 387 |
| 65,536 | 4,293 ± 29 | 225.2 ± 2.0 | 771 |

Rates are the mean and sample standard deviation reported by `llama-bench` for the measured tokens after preloading. They exclude preload time and model loading. Decode KV is the buffer size in the Vulkan log. Recurrent state occupied 19.27 MiB at every depth. In an earlier run with the same settings, decode measured 390.1 tok/s at depth 0 and 226.0 tok/s at 65,536. Repeat short-context comparisons across alternating runs because the empty-context result varied between runs.

From the repository root, run with Windows PowerShell 5.1 or PowerShell 7:

```powershell
.\Heimdall\bench\vulkan_bench.ps1 `
    -BenchPath C:\path\to\llama-bench.exe `
    -ModelPath C:\path\to\model.gguf `
    -OutputDirectory .\Heimdall\runs\vulkan-f16
```

Each run saves raw timing samples, Vulkan device enumeration, model, executable and runtime DLL hashes, flags, and backend logs. Give each run a fresh output directory. Pairing two GGUFs in `-ModelPath` uses the same binary and CLI settings. Use the same source checkpoint and weight quantization for a patched-runtime comparison. For tests with `BIFROST_RET` enabled, prefill the whole prompt in a fresh context. `llama-bench` restores sequence state between depth repetitions, and the current patch does not serialize its retrieval summaries.

This throughput baseline uses synthetic tokens. Retrieval accuracy and task quality need a separate evaluation on real prompts. Peak process VRAM includes allocations beyond the buffers listed here.

## Same-runtime Bifrost comparison

Built `llama-bench` from llama.cpp `5047748f114564fcf271e6a648f60e2d2eae6264` plus [`bifrost.patch`](../llamacpp/bifrost.patch), using MSVC 19.51, Ninja, and Vulkan SDK 1.4.363.0. Benchmark executable SHA-256 `abb1aaf5e2546bdd00c006f705be6cedc771b208fa3c5c781a59d166853e0f4b`.

To build the benchmark from the repository root in an x64 Developer PowerShell with CMake and Ninja on `PATH`, choose fresh paths for `$llama` and `$build`:

```powershell
$llama = 'C:\path\to\new\llama-source'
$build = 'C:\path\to\new\llama-build'
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.363.0'
$commit = (Get-Content .\Heimdall\llamacpp\LLAMA_CPP_COMMIT -Raw).Trim()
$patch = (Resolve-Path .\Heimdall\llamacpp\bifrost.patch).ProviderPath
git init $llama
git -C $llama fetch --depth 1 https://github.com/ggml-org/llama.cpp.git $commit
git -C $llama checkout --detach FETCH_HEAD
git -C $llama apply $patch
cmake -S $llama -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON `
    -DGGML_CCACHE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build $build --target llama-bench -j 8
```

Both runs used this executable, with the settings above and `BIFROST_RET` unset. The Bifrost GGUF preserves all 335 original tensor types and byte payloads, which were compared with `gguf.GGUFReader`. It adds 60 float32 tensors initialized by `HybridAttention.convert` (PyTorch seed 0, six converted attention layers, window 512, four sink tokens, one archival head per layer). The initialized tensors exercise the Bifrost execution path for throughput and memory measurements. They need calibration before evaluating model quality. Bifrost GGUF SHA-256 `306aa9219231cab2406fa94a8ac9c6fe275fb3649cd5fc85e84a7f152d497bbf`.

| Preloaded tokens | Stock prefill 512 | Bifrost prefill 512 | Stock generate 64 | Bifrost generate 64 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 11,617 ± 77 | 10,432 ± 1,072 | 388.7 ± 35.2 | 325.6 ± 42.1 |
| 4,096 | 11,238 ± 926 | 13,561 ± 397 | 396.2 ± 42.3 | 349.6 ± 18.8 |
| 32,768 | 8,930 ± 151 | 13,645 ± 663 | 301.9 ± 27.5 | 350.4 ± 24.3 |
| 65,536 | 4,232 ± 41 | 13,147 ± 624 | 226.0 ± 5.4 | 334.8 ± 36.2 |

All rates are tokens per second. At depth 65,536, Bifrost measured 1.48× the stock decode rate and 3.11× the stock 512-token prefill rate in this run. At depth zero its mean decode rate was 16% lower. Both variants offloaded 26/26 layers. At depth 65,536 the stock variant allocated 771 MiB KV plus 19.27 MiB recurrent state for decode. Bifrost allocated no full-attention KV buffer and 60.61 MiB recurrent state at every measured depth, including 34.92 MiB for its window, sinks, and delta state. The Vulkan model buffer increased from 521.57 to 536.85 MiB.

A deeper run of the same Bifrost GGUF, with three repetitions, measured 302.3 ± 11.0 tok/s at depth 131,072 and 295.5 ± 2.3 tok/s at depth 261,632. Prefill of 512 tokens at those depths measured 13,331 ± 201 and 11,233 ± 2,319 tok/s. The recurrent allocation stayed at 60.61 MiB and the Vulkan compute buffer at 496.04 MiB in all four deep tests.

The Vulkan multi-sequence check used a window of 64 and 160 processed tokens per sequence. Copying a Bifrost sequence and reusing a cleared cell both produced bit-identical continuation logits for Q4_K_M and BF16. The strict `bifrost_multiseq` comparison of batched versus serial execution recorded maximum absolute differences of 0.665 for Bifrost and 0.703 for stock with Q4_K_M, and 0.212 for Bifrost and 0.217 for stock with BF16. A separate check held batch shape at two sequences while changing the other sequence's input. All 39,731,200 logits for the observed sequence were bit-identical on Bifrost Q4_K_M, including after it crossed the window.

## Stock KV cache precision

Repeated with the pinned patched runtime on the stock GGUF and varied only the K and V cache precision. The model weights stayed in Q4_K_M throughout.

| K and V type | 32K decode, tok/s | 32K KV, MiB | 64K decode, tok/s | 64K KV, MiB |
| --- | ---: | ---: | ---: | ---: |
| f16 | 294.8 ± 23.4 | 387.00 | 228.2 ± 7.0 | 771.00 |
| q8_0 | 307.1 ± 41.7 | 205.59 | 233.6 ± 22.0 | 409.59 |
| q4_0 | 343.7 ± 36.3 | 108.84 | 255.3 ± 30.2 | 216.84 |

At 64K, q4_0 used 72% less KV memory. Decode had a higher mean and wider variation. At the same depth, 512-token prefill measured 4,315 ± 77 tok/s for f16, 3,827 ± 253 for q8_0, and 3,739 ± 259 for q4_0. Report both prefill and decode when comparing cache formats.

[`bifrost_passkey.cpp`](../llamacpp/bifrost_passkey.cpp) tested the same stock Q4_K_M model with `NGL=999`, `FA=1`, `MAX_NEW=32`, and `GGML_VK_VISIBLE_DEVICES=0`. With fresh prefill at 8K and 32K, one trial at each of depths 0.1, 0.5, and 0.9, the first number generated matched the pass-key on all six prompts for f16, q8_0, q4_0, and f16 with fixed `BIFROST_RET=16,512,64,4`. These six prompts use repeated filler and serve as a smoke test for the cache and retrieval paths.

## Fixed sparse retrieval with fresh prefill

[`vulkan_fresh_decode.cpp`](vulkan_fresh_decode.cpp) creates a fresh context per depth and fills it with 512-token batches. It warms up 16 single-token steps, then measures the next 64 steps, including retrieval of each step's logits. The model and f16 KV stay on the GPU in both variants. This compares the existing fixed retrieval path with full attention using the same Windows build and stock Q4_K_M GGUF as above.

| Prefill depth | Full attention decode, tok/s | Fixed retrieval decode, tok/s |
| ---: | ---: | ---: |
| 32,768 | 338.1, 340.9 | 252.9, 256.3 |
| 65,536 | 264.8, 265.4 | 248.6, 249.7 |

The runs followed full, fixed, fixed, full order. Fixed `BIFROST_RET=16,512,64,4` was about 25% slower at 32K and 6% slower at 64K in this comparison. Use these measurements when estimating the break-even read rate for an on-demand gate. Host transfers and learned routing still need their own measurement.

The separate Vulkan timestamp run used `GGML_VK_PERF_LOGGER=1` and `GGML_VK_PERF_LOGGER_FREQUENCY=1`. At 64K, across 80 decode steps including warmup, median summed GPU time per step was 581 µs for multiplication by the selected f32 V, 372 µs across pooling dispatches, 272 µs for scoring block bounds, and 18 µs for top-k. Six top-k dispatches per step confirm that the fixed retrieval branch ran. These kernel timings prioritize attention over the gathered values and block summary maintenance for optimization. The timestamp run adds overhead, so use the uninstrumented table for end-to-end throughput.

A local experiment replaced the explicit QK, softmax, and V multiplication with grouped Flash Attention over the same gathered blocks. At 64K, decode rose from 249.0 and 250.8 tok/s to 306.5 and 306.3 tok/s. The experimental path passed the six pass-key prompts, but a separate 64-step logit probe at 4K recorded a maximum absolute difference of 0.217 and top-token agreement on 63/64 steps on Vulkan. The CPU f16 probe differed by up to 0.817 and agreed on 59/64 steps. The existing fixed retrieval path remains the reference for subsequent measurements.

To reproduce from an x64 Developer PowerShell with the patched llama.cpp build available:

```powershell
$llama = 'C:\path\to\patched\llama.cpp'
$build = 'C:\path\to\its\Ninja\build'
$model = 'C:\path\to\Qwen3.5-0.8B-Q4_K_M.gguf'
$importLib = (Get-ChildItem $build -Recurse -Filter llama.lib | Select-Object -First 1).FullName
$exe = Join-Path $build 'bin/vulkan_fresh_decode.exe'
cl.exe /nologo /O2 /EHsc /std:c++17 "/I$llama/include" "/I$llama/ggml/include" `
    "/Fo$build/vulkan_fresh_decode.obj" "/Fe$exe" .\Heimdall\bench\vulkan_fresh_decode.cpp $importLib
$env:GGML_VK_VISIBLE_DEVICES = '0'  # Confirm the physical index locally with llama-bench --list-devices.
Remove-Item Env:BIFROST_RET -ErrorAction SilentlyContinue
& $exe $model
$env:BIFROST_RET = '16,512,64,4'
& $exe $model
```

## Host-to-VRAM transfer

A Vulkan copy probe selected the discrete RX 9070 XT and submitted `vkCmdCopyBuffer` to its compute-capable queue. The source was host-visible, host-coherent system memory. The destination was device-local VRAM. The wall-time measurement covered submit through fence completion, with 5 warmups and 50 timed copies for each row. Multi-region runs copied noncontiguous host ranges into adjacent GPU ranges in one submission.

| Payload | Regions | Median, ms | p95, ms |
| ---: | ---: | ---: | ---: |
| 1 MiB | 1 | 0.106 | 0.173 |
| 4 MiB | 1 | 0.231 | 0.433 |
| 24 MiB | 1 | 1.039 | 1.247 |
| 24 MiB | 24 | 1.020 | 1.213 |
| 24 MiB | 96 | 1.020 | 1.145 |

Use about 1 ms as the measured standalone copy cost for a 24 MiB read. The inference cost will also depend on host index traversal, staging, the number of submissions, and overlap with GPU work. The probe source and samples are saved with the local run artifacts.

## Hierarchical block selection

[`gjallar_index.py`](gjallar_index.py) compares a flat top-16 block-bound scan with a two-level selector that takes the eight highest-scoring parent pages first. Each block contains 64 tokens and each page contains up to 64 blocks. Agreement is the overlap with the flat top-16 set, averaged across queries.

| Keys | Eligible blocks | Pages | Mean top-16 agreement | Worst agreement | Pages above exact cutoff, median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent synthetic keys, 1M tokens | 16,384 | 256 | 4.3% | 0.0% | 256 |
| Page-clustered synthetic keys, 1M tokens | 16,384 | 256 | 88.7% | 37.5% | 256 |
| Qwen3.5-0.8B layer 3, 131K prefill | 2,039 | 32 | 56.5% | 6.25% | 32 |

The synthetic runs use dimension 256, seed 0, and 32 queries. The Qwen run captured 64 query vectors per KV group and the two groups' key bounds from the patched runtime on Vulkan. The context came from tokenizing the repository's tracked Python, Markdown, and C++ source. Scores preserve the maximum over four query heads in each KV group. With 16 parent pages, mean Qwen agreement increased to 85.4%. At eight pages, geometric bounds excluded no parents even when intersected with a bound based on the centroid and enclosing sphere. Ranking by the page mean or mean plus two standard deviations improved mean agreement to 67.9% and 68.4%, respectively. An oracle with the best score of each page's actual leaf blocks reached 90.8% at the same eight-page budget.

These tests measure selection agreement with the flat block-bound heuristic. Answer quality and generalization to other models and workloads need separate evaluation. In this sample, the high-dimensional page boxes saved both index space and first-stage scan traffic at the cost of dropping high-ranking leaf blocks. An exact search using those bounds may scan every page. Raw captures and analysis scripts are in the local run artifacts.

Repeating the capture across all six full-attention layers at 131K produced 768 query and KV-group pairs. With eight pages, mean agreement was 51.9% and layer means ranged from 47.7% to 56.5%. With sixteen pages, mean agreement was 81.0%. The median number of parents above the exact cutoff was 32 of 32 in every layer.

### Quantized flat shortlist

On the same capture, each leaf descriptor concatenates 256 key maxima and 256 key minima. The experiment quantized groups of 32 values symmetrically with fp16 scales, formed a shortlist, then considered the top 16 under their original full-precision scores.

| Descriptor bits | Shortlist | Mean top-16 agreement | Complete agreement |
| ---: | ---: | ---: | ---: |
| 2 | 64 | 51.78% | 0.00% |
| 4 | 32 | 97.77% | 71.22% |
| 4 | 64 | 99.93% | 98.96% |
| 8 | 16 | 98.62% | 78.39% |
| 8 | 32 | 100.00% | 100.00% |

An exactness check bounded score error using the stored group's maximum and L2 quantization residuals. It first scored 64 shortlisted candidates in full precision, then verified any remaining block whose upper score bound could exceed the 16th exact score. At 8 bits, the initial 64 candidates sufficed in all 768 samples. At 4 bits the median candidate count required for verification rose to 1,342 of 2,039. Adjacent layers shared 18.8–27.0% of their original top-16 candidates in this test.

The native Vulkan Q8_0 shortlist also contained every original top-16 block when requesting 32 candidates on all 768 captured query and group pairs. Q4_0 contained all 16 in 767 of 768 pairs with a 64-candidate shortlist. The native probe requested each shortlist size directly. An implementation should rerank before truncating a returned set. A numerical certificate for this backend will also need to bound GPU arithmetic error.

A separate Vulkan probe timed only the packed shortlist stage with the index already resident. It used both KV groups of layer 3, four query heads per group, the max across heads, GPU top-64, and a readback of the resulting indices. The 16,384-row shape corresponds to 1M tokens with 64-token blocks. It repeats captured descriptor rows to fill the larger shape. Capture agreement above uses all real rows at 131K. Each hardware case used five warmups and 50 timed repeats.

| Blocks | Format | Index, MiB | Median, ms | p95, ms |
| ---: | --- | ---: | ---: | ---: |
| 2,039 | f32 | 7.96 | 0.295 | 0.600 |
| 2,039 | Q8_0 | 2.12 | 0.285 | 0.658 |
| 2,039 | Q4_0 | 1.12 | 0.269 | 0.554 |
| 16,384 | f32 | 64.00 | 0.420 | 0.825 |
| 16,384 | Q8_0 | 17.00 | 0.359 | 0.766 |
| 16,384 | Q4_0 | 9.00 | 0.335 | 0.751 |

Index bytes are for one attention layer across both groups. The Vulkan probe uses llama.cpp's Q8_0 and Q4_0 layouts. It times the shortlist stage only, so exact verification, index staging, and KV fetch still contribute to the full inference cost. A fully resident flat index grows with context length.

Uploading the entire layer's index at each call through `ggml_backend_tensor_set` raised 1M-geometry median latency to 3.876 ms for f32, 1.312 ms for Q8_0, and 0.815 ms for Q4_0, including the same shortlist and readback. The corresponding p95 values were 4.358, 1.652, and 1.261 ms. This measures a serialized upload from ordinary host memory. An implementation can evaluate a fixed-capacity device cache or overlap the transfer. KV fetch and final attention remain separate costs.

A custom Vulkan compute probe instead read the S8 index directly from mapped coherent system memory. Its layout uses 512 signed bytes and sixteen f32 scales per row. With 16,384 rows per group across two groups, the index occupied 18 MiB of system memory. Returning scores to an uncached mapping raised selection time to 7.79 ms median. Choosing cached, coherent host memory for output reduced median to 0.887 ms and p95 to 1.165 ms, including GPU scoring, fence completion, and CPU top-64 selection. The 1M shape repeats captured rows. At 2,039 rows and with changing queries across all six 131K layer captures, median selection ranged from 0.211 to 0.231 ms by layer and every top-64 set included the original flat top-16 for all 768 query and group pairs. Maximum observed difference from CPU scores for the same packed values was 0.000854. Model routing, certificate-based verification, and KV fetch remain outside this probe. Its source and measurements are in the local artifacts.

[`gjallar_s8_probe.cpp`](gjallar_s8_probe.cpp) and [`gjallar_s8_probe.comp`](gjallar_s8_probe.comp) reproduce mapped-host timing with seeded synthetic inputs or the captured layer-3 files. From an x64 Developer PowerShell at the repository root:

```powershell
$vk = 'C:\VulkanSDK\1.4.363.0'
$out = Join-Path $env:TEMP 'gjallar-s8-probe'
New-Item -ItemType Directory -Force -Path $out | Out-Null
& "$vk/Bin/glslc.exe" -O .\Heimdall\bench\gjallar_s8_probe.comp -o "$out/probe.spv"
cl.exe /nologo /O2 /EHsc /std:c++17 "/I$vk/Include" "/Fo$out/probe.obj" "/Fe$out/probe.exe" `
    .\Heimdall\bench\gjallar_s8_probe.cpp "$vk/Lib/vulkan-1.lib"
& "$out/probe.exe" "$out/probe.spv"
```

To replay captured data, append `layer-ksum-3` and `layer-q-3` from the local run directory. The input tensors use little-endian f32 and shapes `[4, 2052, 256]` and `[64, 8, 256]`, respectively. The probe chooses a discrete GPU and requires coherent host memory with a CPU-cached output type. It numerically checks each output score against a CPU reference.

A separate local experiment stored the mapped S8 scales in fp16, reducing the 1M-shape index from 18 to 17 MiB. In alternating runs the fp16-scale variant measured 0.853–0.880 ms median versus 0.895–0.904 ms for f32 scales. The fp16-scale scores agreed with the corresponding packed CPU reference within 0.000610. The repository probe retains f32 scales and builds without a CPU half-conversion instruction requirement.

A cache replay computed the exact flat top-16 set before looking up blocks in a fixed-size LRU cache. With 64 entries per KV group, the hit rate was 78.9% when reading at every captured position, 57.7% every fourth position, and 44.4% every eighth position. The f16 K/V cache would use 8 MiB per attention layer. These are simulated hits from one 64-query trace with empty caches at the start. Measuring hits from a trained gate will require a longer trace and the gate's actual read positions.

A Vulkan cache-transfer replay used the captured layer-3 block IDs and synthetic 64 KiB payloads per block. For every-fourth-position reads, a 64-entry-per-group cache reduced the average upload from 2 MiB to 0.73 MiB and median latency from 0.243 to 0.138 ms. At every-position reads, average upload dropped to 0.35 MiB and median time from 0.234 to 0.097 ms. These timings include the CPU cache lookup, packing misses into a pinned staging buffer, Vulkan copy submission and fence completion. The replay uses an empty cache at the start of each pass and keeps attention outside the timer.

## Gate features on captured Qwen attention

At 32K, the layer-3 trace captured 64 query positions and eight heads. Comparing full attention against a 512-token window plus four sinks, the median probability mass outside the window was 28.0%. The median relative L2 difference in each head's value-weighted output was 11.0%.

The target was each head's 16 positions with the largest relative output error. Using only negative window log-sum-exp, mean per-head AUROC was 0.739. The exact outside-window probability mass gave 0.927 and requires scoring the archive. A bounded candidate feature estimates archive log-sum-exp from a fixed-size uniform sample of older keys, adds a correction for archive length, then subtracts the local-window log-sum-exp. The same prefill reservoir was reused across the 64 decode positions. Results average over 32 random reservoirs.

| Keys per KV group | fp16 storage per two-group layer | Mean per-head AUROC | Recall of high-error positions at 25% reads |
| ---: | ---: | ---: | ---: |
| 0, window LSE only | 0 KiB | 0.739 | 48.4% |
| 8 | 8 KiB | 0.775 | 50.8% |
| 32 | 32 KiB | 0.830 | 58.3% |
| 128 | 128 KiB | 0.865 | 62.3% |
| 512 | 512 KiB | 0.893 | 67.2% |

These statistics use one layer on one source-text trace. The oracle target measures attention output, before the layer output projection. Task quality, per-token routing across heads, and runtime cost need separate measurements.

Repeating the 32K capture with full attention across all six layers gave mean per-head AUROC of 0.696 with window log-sum-exp alone, 0.758 with 32 sampled keys and 0.779 with 128. Per-head AUROC improved at every layer. For a single layer decision per token, averaging the eight features gave 0.781, 0.782 and 0.794 against the concatenated attention-output error. Three of six layers lost AUROC under this averaging rule. These results support testing a learned head aggregation before adopting the feature for a layer gate.

A held-out layer-level experiment captured 256 positions from each of two 32K prompts, one made from active repository source and one from archived source. A ridge model fit on one trace and scored the other. The split was reversed for each of six layers. Ridge features used each head's negative window log-sum-exp and, when present, its estimated archive log-partition relative to the window. Means for reservoir variants average eight placements per split.

| Layer decision | Mean held-out AUROC | High-error recall at 25% reads |
| --- | ---: | ---: |
| Mean window feature across heads | 0.729 | 46.1% |
| Ridge on eight window features | 0.731 | 49.5% |
| Ridge with 32 sampled keys per group | 0.794 | 55.1% |
| Ridge with 128 sampled keys per group | 0.827 | 59.1% |

Each reservoir size raised AUROC in 11 of 12 layer and split-direction pairs relative to the window-only ridge model. This evaluates transfer between two source-text traces using the same checkpoint. It measures which tokens had the largest attention-output differences. Long-context answer quality and the cost of making the decision in the runtime remain to be measured.
