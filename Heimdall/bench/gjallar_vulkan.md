# Gjallar Vulkan experiments

## Model replay

Use a Vulkan-enabled llama.cpp build of the commit in `llamacpp/LLAMA_CPP_COMMIT`. Apply `llamacpp/bifrost.patch`, then `bench/gjallar_model_replay.patch`, and rebuild the `llama` target. In an x64 Visual Studio developer PowerShell, set `$llama` to the source tree, `$build` to its build directory, and run these commands from the repository's inner `Heimdall` directory. Set `$model` to the Qwen3.5-0.8B Q4_K_M GGUF and `$corpus` to a UTF-8 text that tokenizes to at least 33,025 tokens.

```powershell
cl.exe /nologo /std:c++17 /EHsc /O2 /I"$llama/include" /I"$llama/ggml/include" `
    /Fo"$build/gjallar_model_replay.obj" /Fe"$build/bin/gjallar_model_replay.exe" `
    bench/gjallar_model_replay.cpp "$build/src/llama.lib" "$build/ggml/src/ggml-base.lib"

$out = (New-Item -ItemType Directory -Force "$build/gjallar-replay").FullName
$replay = Join-Path $build 'bin/gjallar_model_replay.exe'
$env:BIFROST_RET = '16,512,64,4'
$env:GJALLAR_DIAGNOSTIC_ROUTE = Join-Path $out 'full.u8'
[IO.File]::WriteAllBytes($env:GJALLAR_DIAGNOSTIC_ROUTE, [byte[]](@(1) * 1536))
& $replay $model $corpus --capture "$out/cal-archive" | Set-Content "$out/full.csv"
if ($LASTEXITCODE -ne 0) { throw 'Capture failed' }

python bench/gjallar_model_replay.py $out archive $out
if ($LASTEXITCODE -ne 0) { throw 'Route generation failed' }
$env:GJALLAR_DIAGNOSTIC_PICKS = Join-Path $out 'picks.i32'
foreach ($mode in 'window', 'always-c4', 'random', 'window-lse', 'reservoir') {
    $env:GJALLAR_DIAGNOSTIC_ROUTE = Join-Path $out "$mode.u8"
    & $replay $model $corpus | Set-Content "$out/$mode.csv"
    if ($LASTEXITCODE -ne 0) { throw "Replay failed for $mode" }
}
Remove-Item Env:GJALLAR_DIAGNOSTIC_ROUTE, Env:GJALLAR_DIAGNOSTIC_PICKS, Env:BIFROST_RET
```

Each CSV has 256 next-token NLLs and target and argmax token IDs. An all-full route serves as the numerical control. The three matched-budget routes open exactly 64 times per attention layer. Every open C4 step keeps the window, four sinks, and 16 blocks per KV group, using a shortlist of 64 and exact token-max reranking. The gate ranks steps offline using a complete trace, and the runtime also reads the block picks from that trace. The experiment measures the effect of fixed schedules on teacher-forced next-token loss. It uses full attention during prefill and retains the full KV cache for decode.

| Route | Archived source NLL | Active source NLL |
| --- | ---: | ---: |
| Full attention | 0.5639 | 0.5406 |
| Window + sinks | 1.1870 | 1.4726 |
| Always C4 | 0.6301 | 0.5731 |
| Random, 25%, mean of five seeds | 1.0415 | 1.1826 |
| Window LSE, 25% | 1.0876 | 1.0238 |
| Reservoir, 25% | 0.9529 | 0.9430 |

These are nats per token for two 256-token continuations of the local archived and active Heimdall source. The default random seed from the route generator yielded 1.0582 and 1.2321. Across all five seeds, random NLL ranged from 1.0116 to 1.0605 on the archive and 1.1223 to 1.2321 on active source. The archived full-route CSV matched stock full attention exactly. The recurrent captures contain the GDN output before gated normalization, with 16 heads of width 128, from the layer immediately before each attention anchor.

### Prefix-only thresholds

With captures for both corpora in the same directory, pass `--causal` to the route generator. Replay the two causal routes and their five random controls using the same loop. Each control matches its corresponding gate's read count per layer. The fixed threshold is the other text's 75th percentile. The rolling threshold is the 75th percentile of the preceding 64 scores, initialized from the other text.

| Text and gate | Reads | Gate NLL | Matched random NLL range |
| --- | ---: | ---: | ---: |
| Archive, fixed | 37.0% | 0.8326 | 0.9098–0.9903 |
| Archive, rolling | 27.7% | 0.9432 | 0.9825–1.0949 |
| Active, fixed | 16.0% | 1.1760 | 1.3094–1.3840 |
| Active, rolling | 25.5% | 1.0028 | 1.1878–1.2602 |

The rolling threshold kept the read rate closer to 25% on both texts. Its decisions used scores from the current step and prior steps only. Scores and C4 picks came from frozen full-attention traces.

### Passkey retrieval

Using the template from `llamacpp/bifrost_passkey.cpp`, insert passkeys 18758, 33056, and 17717 at 10%, 50%, and 90% of the repeated filler. Fill the prefix to 32,769 tokens so the first answer token is scored after position 32,768. Follow the prompt with the expected answer and enough filler to complete the replay. Count an exact answer when all six answer tokens, including the leading space, are the model's argmax under teacher forcing.

| Route | Exact answers | Reads during the six answer steps per case, across all layers |
| --- | ---: | ---: |
| Full attention | 3/3 | 36 |
| Window + sinks | 0/3 | 0 |
| Always C4 | 3/3 | 36 |
| Random, 25% | 0/3 | 7 |
| Window LSE, 25% | 3/3 | 32–33 |
| Reservoir, 25% | 0/3 | 0 |

The reservoir missed every answer in this small passkey check despite improving next-token loss on source text. Always C4 recovered the passkeys at all three depths. The route selected by window LSE found the answer steps within the same 25% budget.

### Rolling unions

A union opens if either rolling feature exceeds its threshold. The budgeted variant uses the 75th-percentile union with eight reads of initial credit per layer and spreads the remaining 56 credits over 256 steps. The balance is capped at eight, and every read consumes one credit. For passkey, the six answer steps start with a full balance.

| Gate | Active reads | Active NLL | Archive reads | Archive NLL | Passkey reads | Exact answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rolling reservoir | 25.5% | 1.0028 | 27.7% | 0.9432 | 26.6–28.6% | 0/3 |
| Rolling window LSE | 24.6% | 1.0547 | 28.8% | 1.0517 | 27.0–27.9% | 3/3 |
| Union at the 75th percentiles | 38.1% | 0.9195 | 46.2% | 0.8968 | 44.4–47.1% | 3/3 |
| Union at the 87.5th percentiles | 23.1% | 1.0625 | 30.0% | 0.9744 | 26.4–28.4% | 3/3 |
| Budgeted union | 22.3% | 1.1361 | 24.3% | 1.0304 | 24.0–24.3% | 3/3 |

To reproduce a passkey union, use `--causal --calibration-dir` with the active-source capture as calibration. The budgeted union preserved all three answers within the 25% cap and had higher source-text NLL than rolling reservoir. At matched per-layer read counts, the budgeted union beat all five random routes on active source (range 1.2067–1.3379) and was close to their mean of 1.0327 on archived source (range 0.9886–1.0779).

### Passkeys after 64 decode steps

The second passkey check places the question and answer 64 tokens later. Fill each prompt to 32,833 tokens and score the same six-token answer at replay steps 64–69. All routes use the same three keys, needle depths, and calibration as the first check.

| Route | Reads across all replay steps | Correct answer tokens at 10%, 50%, 90% needle depth | Exact answers |
| --- | ---: | ---: | ---: |
| Full attention | 100% | 6, 6, 6 | 3/3 |
| Always C4 | 100% | 6, 6, 6 | 3/3 |
| Window + sinks | 0% | 1, 1, 3 | 0/3 |
| Window LSE ranked over the continuation | 25.0% | 6, 6, 6 | 3/3 |
| Rolling reservoir | 28.1–31.1% | 6, 6, 5 | 2/3 |
| Rolling window LSE | 28.3–28.7% | 6, 6, 6 | 3/3 |
| Union at the 75th percentiles | 45.8–48.4% | 6, 6, 6 | 3/3 |
| Union at the 87.5th percentiles | 28.1–29.7% | 6, 6, 6 | 3/3 |
| Budgeted union | 23.9–24.4% | 4, 5, 3 | 0/3 |
| Budgeted union at the 87.5th percentiles | 20.4–21.7% | 6, 5, 5 | 1/3 |
| Budgeted window LSE | 20.1–20.4% | 6, 6, 3 | 2/3 |

Before each delayed answer the capped union had fewer than four reads of credit left in every layer. It admitted 15–18 of the 31–32 answer reads requested by the unrestricted union. Applying the same credit budget to window LSE improved two cases while leaving the deepest key unresolved. Ranking window LSE over the whole continuation selected 30–32 answer reads at exactly 25% overall. That control uses future scores and shows what the budget can support when reads land on the answer.

### Reserved credit and borrowing

Both experiments use the 87.5th percentile of the preceding 64 negative window-LSE scores for priority. `causal-borrow-budget` allows those requests to spend up to eight future reads while enforcing 64 total reads per layer. `causal-reserved-lse` allocates 16 reads to priority LSE requests and 48 to reservoir requests. The LSE account starts with ten reads and refills by six over the replay. The reservoir account starts with six, refills by 42, and carries at most eight. Both routes are generated by `--causal`.

| Route | Active reads and NLL | Archive reads and NLL | Delayed passkey reads | Correct answer tokens at 10%, 50%, 90% depth |
| --- | ---: | ---: | ---: | ---: |
| Budgeted union | 22.3%, 1.1361 | 24.3%, 1.0304 | 23.9–24.4% | 4, 5, 3 |
| Borrowing | 23.7%, 1.0040 | 24.6%, 0.9797 | 24.1–24.6% | 5, 6, 5 |
| Reserved LSE | 21.7%, 1.1014 | 21.2%, 1.0327 | 21.4–21.7% | 3, 2, 4 |

Borrowing improved NLL on both source continuations and recovered one of three delayed answers. It beat all five random controls with the same per-layer read counts on active source (1.0737–1.2727) and archived source (1.0082–1.0415). The reserved route denied 24/25, 22/26, and 21/23 priority LSE requests during the three answers. Earlier LSE requests had already used the reserved credit.

The selected C4 blocks overlapped the needle on 69–72 of 72 possible answer reads per fixture with always-C4, counting each KV group and layer separately. Whole-continuation LSE read the needle on 59–61 of those opportunities, the original capped union on 30–35, and borrowing on 46–51.

| Reserved LSE cutoff | Active reads and NLL | Archive reads and NLL | Delayed passkey reads | Correct answer tokens at 10%, 50%, 90% depth |
| --- | ---: | ---: | ---: | ---: |
| 90th percentile | 21.7%, 1.0976 | 21.4%, 1.0525 | 21.2–21.7% | 3, 2, 4 |
| 95th percentile | 21.9%, 1.0588 | 21.4%, 1.0305 | 21.0–21.1% | 4, 2, 4 |
| 98th percentile | 19.9%, 1.0847 | 20.7%, 1.0252 | 19.9–20.4% | 3, 2, 4 |

The cutoffs requested 106–116, 86–90, and 68–69 priority reads over the first 64 delayed steps, respectively, summed over six layers. At the 95th percentile, the reserved route denied 5/10, 4/8, and 4/7 priority answer requests. At the 98th percentile, only 3, 5, and 6 priority answer requests fired and 0, 0, and 3 were denied. All three cutoffs missed all three delayed answers.

### Calibration from the current prompt

Use `--capture-with-history` in place of `--capture` to save the final 64 prefill queries, then pass `--prefill-history --causal` to the Python route generator. The generator initializes only the window-LSE threshold with those queries and keeps reservoir calibration on the other text. Replay `causal-reserved-q95-prefill` and `causal-borrow-prefill` using the original C4 picks.

| Route initialized from prefill | Delayed passkey reads | Correct answer tokens at 10%, 50%, 90% depth | Exact answers |
| --- | ---: | ---: | ---: |
| Reserved LSE at the 95th percentile | 19.7–20.2% | 6, 6, 5 | 2/3 |
| Borrowing | 21.4–21.8% | 5, 6, 5 | 1/3 |

Initializing with the current prompt reduced the reserved route's early priority requests from 86–90 to 22–23 across six layers. It granted all 10, 8, and 7 priority answer requests, while the deepest answer still missed one token. Borrowing admitted 24–29 answer layer reads and lowered answer NLL at every depth compared with its cross-text calibration.

| Route initialized from prefill | Active reads and NLL | Archive reads and NLL |
| --- | ---: | ---: |
| Reserved LSE at the 95th percentile | 21.9%, 1.0460 | 20.8%, 1.0284 |
| Borrowing | 24.5%, 1.0281 | 24.2%, 0.9573 |

### Archive mass from C4 centroids

`--prefill-history --causal` also produces `causal-centroid-mass`. This route estimates each head's archive softmax mass by treating every block's 64 keys as its centroid. It subtracts the local window log-sum-exp and opens when the mean across heads exceeds the rolling 87.5th percentile. On active and archived source, it read 14.8% and 17.7% of layer steps and measured 1.1006 and 0.9628 NLL. On the delayed passkeys, it read 14.7–15.6% and predicted 5, 4, and 5 of six answer tokens correctly.

Add `--match-read-count` to fit a rolling cutoff for each route using total reads on the completed trace. This uses no answer labels, and the rolling decisions use previous scores. The cutoff fit uses future read-count information. All four routes issued exactly 384 reads over the six layers and 256 steps on each input.

| Route at 25% reads | Active NLL | Archive NLL | Exact delayed answers | Delayed answer NLL at 10%, 50%, 90% depth |
| --- | ---: | ---: | ---: | ---: |
| Centroid mass | 0.9982 | 0.8888 | 3/3 | 0.0250, 0.0396, 0.0280 |
| Reservoir | 1.0019 | 0.9416 | 2/3 | 0.6232, 0.5534, 0.7672 |
| Window LSE | 1.0042 | 1.0499 | 3/3 | 0.0327, 0.0617, 0.0420 |
| Reservoir or window LSE | 1.0279 | 0.9742 | 3/3 | 0.1415, 0.2626, 0.1459 |

The fitted centroid cutoffs were 0.7744 and 0.7896 for the active and archive continuations, and 0.7286, 0.7403, and 0.7557 for the delayed passkeys. Centroid mass and window LSE start from the current prompt's prefill history. Reservoir starts from the other text's history.

Without `--match-read-count`, the generator also produces three centroid routes with cutoffs based only on prior scores and decisions. The controllers start at the 75th percentile and raise or lower the quantile after each step based on the observed reads. The per-layer controller allows 64 reads per layer with a gain of 1/64. The shared controller allows 384 reads total, updates once per step from the combined read fraction, and uses a gain of 1/256.

| Causal centroid route | Active reads and NLL | Archive reads and NLL | Delayed passkey reads | Exact delayed answers |
| --- | ---: | ---: | ---: | ---: |
| Fixed 75th percentile | 27.7%, 0.9750 | 28.8%, 0.8957 | 23.9–25.4% | 3/3 |
| Adaptive, per-layer cap | 25.0%, 1.0736 | 24.7%, 0.8983 | 23.2–24.0% | 2/3 |
| Adaptive, shared cap | 25.0%, 1.0591 | 25.0%, 0.9015 | 23.8–24.7% | 3/3 |

The shared controller's delayed answer NLL was 0.0250, 0.0396, and 0.0280. Its 384-read limit spans a fixed 256-step replay. It spent the last read at step 217 on active source and step 221 on archived source, leaving subsequent steps to the window. The scores and block picks remain frozen from full-attention traces.

### Late passkey and refilling budget

The late fixture places the 90%-depth passkey answer at replay steps 224–229. The refilling route (`causal-centroid-refill`) uses the shared controller with a 24-read initial balance and capacity. It adds 1.5 reads of credit per step across six layers, giving a maximum of about 26.6% over this replay.

| Route | Overall reads | Answer layer reads | Correct answer tokens | Answer NLL |
| --- | ---: | ---: | ---: | ---: |
| Full attention | 100% | 36 | 6/6 | 0.0168 |
| Always C4 | 100% | 36 | 6/6 | 0.0184 |
| Window + sinks | 0% | 0 | 3/6 | 2.4125 |
| Fixed 75th percentile | 27.3% | 30 | 6/6 | 0.0188 |
| Adaptive, shared cap | 25.0% | 30 | 6/6 | 0.0186 |
| Adaptive, shared refill | 19.5% | 9 | 3/6 | 2.3700 |

The fixed cap had 117 reads available just before the answer. The refill route had half a read of credit left. Its limited carryover throttled the answer burst despite using fewer reads overall.

## Live-query diagnostic

Apply `bench/gjallar_live_replay.patch` after the model replay patch and rebuild `llama`. Capture the full run on the same model and corpus with `--capture-with-history`, then prepare the immutable archive and run the live probe in a fresh process:

```powershell
python bench/gjallar_prepare_live.py $out archive "$out/live-archive.bin"
if ($LASTEXITCODE -ne 0) { throw 'Archive packing failed' }
Remove-Item Env:GJALLAR_DIAGNOSTIC_ROUTE, Env:GJALLAR_DIAGNOSTIC_PICKS -ErrorAction SilentlyContinue
$env:BIFROST_RET = '16,512,64,4'
$env:GJALLAR_LIVE_ARCHIVE = Join-Path $out 'live-archive.bin'
$env:GJALLAR_LIVE_POLICY = 'adaptive'
& $replay $model $corpus | Set-Content "$out/live-adaptive.csv"
if ($LASTEXITCODE -ne 0) { throw 'Live replay failed' }
Remove-Item Env:GJALLAR_LIVE_ARCHIVE, Env:GJALLAR_LIVE_POLICY, Env:BIFROST_RET
```

Each single-token decode step sends its current query and window logits to a CPU custom operation. The operation recomputes C4 scores and the adaptive shared-budget decision, and reranks only on an open decision. Attention consumes those results in the same step. The prepared file contains the archive keys and centroid vectors from the full prefill, and the initial 64 per-layer scores. Archive blocks are gathered on the GPU in both open and closed steps using a fixed graph shape. A mask removes their contribution in closed steps. This setup measures live routing and quality under feedback. Savings in physical fetches and a device-native implementation remain separate work.

| Input | Archive reads | Adaptive NLL | Answer |
| --- | ---: | ---: | ---: |
| Active source | 384/1536 | 1.0719 | — |
| Archived source | 384/1536 | 0.9197 | — |
| Step 64 passkey, 10% needle depth | 375/1536 | 0.0572 | 6/6 |
| Step 64 passkey, 50% needle depth | 374/1536 | 0.0901 | 6/6 |
| Step 64 passkey, 90% needle depth | 375/1536 | 0.0963 | 6/6 |
| Step 224 passkey, 90% needle depth | 384/1536 | 0.0318 | 6/6 |

Passkey NLL covers the six teacher-forced answer tokens. The active-source control policies `never` and `always` measured 1.4740 and 0.5722 NLL at 0% and 100% reads. The first live score matched the frozen reference within 0.00001. The forced-closed graph changed one of 256 argmax tokens relative to frozen window attention, with a mean absolute token-NLL difference of 0.0056.

On active source, grouping four query heads into SIMD lanes preserved each head's accumulation order. Sharing the packed heads between the score and rerank stages added 8 KiB of transient query storage across both KV groups. Platforms without SSE2 use the scalar fallback. Two post-prefill runs per variant measured:

| CPU implementation | Centroid scoring and controller | Exact reranking | Live throughput |
| --- | ---: | ---: | ---: |
| Scalar | 0.95 s | 2.34–2.36 s | 43.4–43.9 tok/s |
| Four-head SIMD for reranking | 0.94–0.95 s | 1.15–1.16 s | 57.5–57.7 tok/s |
| Four-head SIMD for scoring and reranking | 0.30 s | 1.16–1.17 s | 67.5–67.9 tok/s |

The two SIMD variants reproduced the scalar NLL and routing traces on active source. The final SIMD version also reproduced the scalar NLL output on archived source and the four passkey fixtures. The timed interval covers 256 decode tokens, including logits and route logging.

Within the live decode range, the graph can reuse its buffers while new window indices and masks arrive at each step. On active source it reused 255 of 256 graphs and reached 75.6 tok/s in both runs. Setting `LLAMA_GRAPH_REUSE_DISABLE=1` gave 54.8 and 66.6 tok/s in those two paired runs. The two variants produced identical NLL rows and decision traces. The original graph checks still apply, and the live range's boundary forces a new graph.

## C4 shortlist and key upload

```powershell
& "$env:VULKAN_SDK/bin/glslc.exe" -O bench/gjallar_c4_probe.comp -o "$out/gjallar_c4.spv"
cl.exe /nologo /std:c++17 /EHsc /O2 /I"$env:VULKAN_SDK/Include" `
    /Fo"$out/gjallar_c4_probe.obj" /Fe"$out/gjallar_c4_probe.exe" `
    bench/gjallar_s8_probe.cpp /link /LIBPATH:"$env:VULKAN_SDK/Lib" vulkan-1.lib
& "$out/gjallar_c4_probe.exe" --c4 "$out/gjallar_c4.spv" "$out/cal-archive-k-3" "$out/cal-archive-q-3"
```

Omit the last two arguments to use synthetic keys and queries. The host benchmark uses x64 F16C instructions for scale conversion. On the RX 9070 XT, the C4 probe used 4.5 MiB across two KV groups at 16,384 rows per group and took 328 µs median from score submission through CPU top-64 selection (527 µs p95). Score submission and synchronization alone took 271 µs. The earlier S8 variant with fp16 scales used 17 MiB and measured 854 µs to the shortlist in the adjacent run. The two formats use different captured descriptors and share the same geometry. The 16,384-row C4 test repeats captured centroid rows to measure the index geometry.

Median PCIe upload was 150 µs for 2 MiB of fp16 keys from 64 selected regions in one KV group, and 225 µs for 4 MiB from 128 regions across two groups (263 and 433 µs p95). The measured interval runs from copy submission to fence completion, with an already pinned, group-major archive at 32K. Archive staging, token-max reranking, and value fetch fall outside that interval. The packed C4 path adds nibble decoding and fp16 scale conversion. The shorter descriptor and single centroid reduce bytes read at the cost of a separate rerank stage. C4 scores matched the packed CPU reference within 0.0001 in the captured run.

For device-local descriptor timing, build an additional per-head scorer and reduction shader.

```powershell
& "$env:VULKAN_SDK/bin/glslc.exe" -O -DPER_HEAD=1 bench/gjallar_c4_probe.comp -o "$out/gjallar_c4_head.spv"
& "$env:VULKAN_SDK/bin/glslc.exe" -O bench/gjallar_c4_mass_reduce.comp -o "$out/gjallar_c4_reduce.spv"
& "$out/gjallar_c4_probe.exe" --c4-device "$out/gjallar_c4.spv" "$out/cal-archive-k-3" "$out/cal-archive-q-3"
& "$out/gjallar_c4_probe.exe" --c4-mass "$out/gjallar_c4_head.spv" "$out/gjallar_c4_reduce.spv" `
    "$out/cal-archive-k-3" "$out/cal-archive-q-3"
```

On the RX 9070 XT at 16,384 rows per group, median submit-to-fence time was 125–129 µs for the scorer alone and 205–208 µs for the per-head scorer plus log-sum-exp. The latter adds a dispatch, a barrier, and 512 KiB of scratch at this geometry. The two modes use the same mapped query buffer and device-local descriptors and outputs. The result download for CPU verification runs outside the timed interval. The reduction matched CPU results within 0.000001. At 503 rows per group, medians ranged from 76 to 93 µs across the four alternating runs.
