"""Make frozen Qwen3.5 replay routes from full-attention q/k captures.

Usage: python bench/gjallar_model_replay.py CAPTURE_DIR active|archive OUTPUT_DIR [--causal] [--prefill-history]
       Add --match-read-count to fit rolling cutoffs to 25% overall using the completed score trace.

The runtime patch reads a [6,256] uint8 route (0: window, 1: full, 2: C4)
and optional [6,256,2,16] little-endian int32 block IDs. Layer order is
3,7,11,15,19,23. Each C4 read uses a 64-block shortlist shared by the
four heads in a KV group, then exact token-max reranking from captured keys.
Gate scores and picks are frozen from full-attention traces, so the replay
measures next-token NLL under those interventions with teacher forcing.
The optional causal experiment adds fixed and rolling thresholds, plus unions
of the two rolling features. A budgeted union allows short bursts while
capping reads. Random controls match each layer's observed read count.
"""

import argparse
from pathlib import Path

import numpy as np

DEPTH, STEPS, WINDOW, SINKS = 32768, 256, 512, 4
BLOCK, SHORTLIST, PICKS, RESERVOIR = 64, 64, 16, 32
GROUPS, HEADS, DIM = 2, 4, 256
LAYERS = range(3, 24, 4)
ATTENTION_SCALE = 1 / 16
OLD_COUNT = (DEPTH - WINDOW + 1 - SINKS) // BLOCK * BLOCK
HISTORY = 64


def logsumexp(scores):
    peak = scores.max(axis=-1, keepdims=True)
    return peak[..., 0] + np.log(np.exp(scores - peak).sum(axis=-1))


def gate_scores(q, k, rng):
    reservoir = rng.choice(OLD_COUNT, RESERVOIR, replace=False) + SINKS
    starts = DEPTH - WINDOW + 1 + np.arange(STEPS)
    window = starts[:, None] + np.arange(WINDOW)
    local = np.concatenate((np.broadcast_to(np.arange(SINKS), (STEPS, SINKS)), window), axis=1)
    lse = np.zeros(STEPS, dtype=np.float64)
    reservoir_score = np.zeros(STEPS, dtype=np.float64)
    for group in range(GROUPS):
        for step, query in enumerate(q[:, group]):
            local_lse = logsumexp(query @ k[local[step], group].T * ATTENTION_SCALE)
            sample_lse = logsumexp(query @ k[reservoir, group].T * ATTENTION_SCALE)
            lse[step] -= local_lse.mean() / GROUPS
            reservoir_score[step] += (sample_lse - local_lse).mean() / GROUPS
    return lse, reservoir_score


def archive_blocks(k):
    return k[SINKS:SINKS + OLD_COUNT].reshape(-1, BLOCK, GROUPS, DIM)


def quantized_centroids(old_keys):
    centroids = old_keys.mean(axis=1).reshape(-1, DIM // 32, 32)
    scale = np.maximum(np.abs(centroids).max(axis=-1, keepdims=True), 1e-12) / 7
    codes = np.rint(centroids / scale).clip(-7, 7)
    return (codes * scale.astype(np.float16).astype(np.float32)).reshape(-1, DIM)


def block_picks(q, k):
    old = archive_blocks(k)
    picks = np.empty((STEPS, GROUPS, PICKS), dtype="<i4")
    for group in range(GROUPS):
        old_keys = old[:, :, group].copy()
        c4 = quantized_centroids(old_keys)
        for step, query in enumerate(q[:, group]):
            sparsity = (query @ c4.T).max(axis=0)
            shortlist = np.argsort(-sparsity, kind="stable")[:SHORTLIST]
            tokens = old_keys[shortlist].reshape(-1, DIM)
            exact = (query @ tokens.T).reshape(HEADS, SHORTLIST, BLOCK).max(axis=(0, 2))
            picks[step, group] = shortlist[np.argsort(-exact, kind="stable")[:PICKS]]
    return picks


def layer_routes(lse, reservoir_score, rng):
    random_steps = rng.choice(STEPS, STEPS // 4, replace=False)
    ranked = {"random": random_steps, "window-lse": np.argsort(-lse, kind="stable")[:STEPS // 4],
              "reservoir": np.argsort(-reservoir_score, kind="stable")[:STEPS // 4]}
    routes = {"window": np.zeros(STEPS, np.uint8), "full": np.ones(STEPS, np.uint8),
              "always-c4": np.full(STEPS, 2, np.uint8)}
    for mode, open_steps in ranked.items():
        routes[mode] = np.zeros(STEPS, np.uint8)
        routes[mode][open_steps] = 2
    return routes


def rolling_reads(scores, calibration, quantile):
    history = np.concatenate((calibration[-HISTORY:], scores))
    thresholds = np.array([np.quantile(history[i:i + HISTORY], quantile) for i in range(STEPS)])
    return scores > thresholds


def apply_read_budget(candidates, priority=None):
    # Credits are integer multiples of 1 / STEPS reads.
    burst = 8
    capacity = burst * STEPS
    limit = STEPS // 4
    refill = limit - burst
    credit = capacity
    used = 0
    route = np.zeros(STEPS, dtype=np.uint8)
    if priority is None:
        priority = np.zeros_like(candidates, dtype=bool)
    for step, (wants_read, urgent) in enumerate(zip(candidates, priority)):
        credit = min(capacity, credit + refill)
        if (wants_read and credit >= STEPS) or (urgent and credit - STEPS >= -capacity):
            route[step] = 2
            credit -= STEPS
            used += 1
            if used == limit:
                break
    return route


def apply_reserved_budget(reservoir_reads, priority):
    # Allocate 16 reads to LSE (10 initial + 6 refilled) and 48 to the reservoir.
    lse_credit = 10 * STEPS
    reservoir_credit = 6 * STEPS
    route = np.zeros(STEPS, dtype=np.uint8)
    for step, (diffuse, urgent) in enumerate(zip(reservoir_reads, priority)):
        lse_credit = min(16 * STEPS, lse_credit + 6)
        reservoir_credit = min(8 * STEPS, reservoir_credit + 42)
        if urgent and lse_credit >= STEPS:
            lse_credit -= STEPS
            route[step] = 2
        elif diffuse and reservoir_credit >= STEPS:
            reservoir_credit -= STEPS
            route[step] = 2
    return route


def causal_routes(lse, scores, calibration_lse, calibration):
    reservoir_reads = rolling_reads(scores, calibration, 0.75)
    window_reads = rolling_reads(lse, calibration_lse, 0.75)
    union = reservoir_reads | window_reads
    priority = rolling_reads(lse, calibration_lse, 0.875)
    tight_union = rolling_reads(scores, calibration, 0.875) | priority
    routes = {
        "causal-fixed": np.where(scores > np.quantile(calibration, 0.75), 2, 0).astype(np.uint8),
        "causal-rolling": reservoir_reads.astype(np.uint8) * 2,
        "causal-window-lse": window_reads.astype(np.uint8) * 2,
        "causal-lse-budget": apply_read_budget(window_reads),
        "causal-union": union.astype(np.uint8) * 2,
        "causal-union-q875": tight_union.astype(np.uint8) * 2,
        "causal-union-q875-budget": apply_read_budget(tight_union),
        "causal-union-budget": apply_read_budget(union),
        "causal-borrow-budget": apply_read_budget(union, priority),
        "causal-reserved-lse": apply_reserved_budget(reservoir_reads, priority),
    }
    for percentile in (90, 95, 98):
        urgent = rolling_reads(lse, calibration_lse, percentile / 100)
        routes[f"causal-reserved-q{percentile}"] = apply_reserved_budget(reservoir_reads, urgent)
    return routes


def load_capture(directory, kind, layer):
    prefix = directory / f"cal-{kind}"
    q = np.fromfile(f"{prefix}-q-{layer}", "<f4").reshape(STEPS, GROUPS, HEADS, DIM)
    k = np.fromfile(f"{prefix}-k-{layer}", "<f2").reshape(DEPTH + STEPS, GROUPS, DIM).astype("f4")
    if not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError(f"Non-finite {kind} capture at layer {layer}")
    return q, k


def load_prefill_queries(directory, kind, layer):
    path = directory / f"cal-{kind}-history-q-{layer}"
    q = np.fromfile(path, "<f4").reshape(HISTORY, GROUPS, HEADS, DIM)
    if not np.isfinite(q).all():
        raise ValueError(f"Non-finite prefill queries at layer {layer}")
    return q


def local_attention_lse(query, keys, pos):
    local = np.r_[np.arange(SINKS), np.arange(pos - WINDOW + 1, pos + 1)]
    return logsumexp(query @ keys[local].T * ATTENTION_SCALE)


def prefill_lse_history(q, k):
    scores = np.zeros(HISTORY, dtype=np.float64)
    for group in range(GROUPS):
        for i, query in enumerate(q[:, group]):
            pos = DEPTH - HISTORY + i
            scores[i] -= local_attention_lse(query, k[:, group], pos).mean() / GROUPS
    return scores


def centroid_mass_scores(q, k, prefill_q):
    all_q = np.concatenate((prefill_q, q))
    old = archive_blocks(k)
    mass = np.zeros(len(all_q), dtype=np.float64)
    for group in range(GROUPS):
        centroids = quantized_centroids(old[:, :, group])
        archive_lse = logsumexp(all_q[:, group] @ centroids.T * ATTENTION_SCALE) + np.log(BLOCK)
        for i, query in enumerate(all_q[:, group]):
            pos = DEPTH - HISTORY + i
            local_lse = local_attention_lse(query, k[:, group], pos)
            mass[i] += (archive_lse[i] - local_lse).mean() / GROUPS
    return mass[HISTORY:], mass[:HISTORY]


def control_centroid_reads(scores, prior):
    target_rate = 0.25
    limit = STEPS // 4
    quantile = 0.75
    history = np.concatenate((prior, scores))
    route = np.zeros(STEPS, dtype=np.uint8)
    used = 0
    for step, score in enumerate(scores):
        if used == limit:
            break
        opened = score > np.quantile(history[step:step + HISTORY], quantile)
        if opened:
            route[step] = 2
            used += 1
        # Increase the cutoff when the observed read rate is above target.
        quantile = float(np.clip(quantile + (int(opened) - target_rate) / HISTORY, 0.50, 0.98))
    return route


def control_total_centroid_reads(layer_signals, refilling=False):
    layer_count = len(layer_signals)
    target_rate = 0.25
    # Keep the shared balance in quarters of one read.
    capacity = layer_count * (16 if refilling else STEPS)
    refill = layer_count if refilling else 0
    credit = capacity
    histories = [np.concatenate((prior, scores)) for scores, prior in layer_signals]
    quantile = 0.75
    route = np.zeros((layer_count, STEPS), dtype=np.uint8)
    for step in range(STEPS):
        credit = min(capacity, credit + refill)
        if credit < 4 and not refill:
            break
        opened = 0
        for layer, ((scores, _), history) in enumerate(zip(layer_signals, histories)):
            if credit >= 4 and scores[step] > np.quantile(history[step:step + HISTORY], quantile):
                route[layer, step] = 2
                credit -= 4
                opened += 1
        quantile = float(np.clip(quantile + (opened / layer_count - target_rate) / (4 * HISTORY), 0.50, 0.98))
    return route


def match_read_count(layer_signals):
    # Sort the history once so fitting a shared cutoff is cheap. Each decision
    # still compares the current score against only the prior HISTORY scores.
    traces = []
    for signals in layer_signals:
        layer = []
        for scores, prior in signals:
            history = np.concatenate((prior[-HISTORY:], scores))
            windows = np.lib.stride_tricks.sliding_window_view(history, HISTORY)[:-1]
            layer.append((scores, np.sort(windows, axis=-1)))
        traces.append(layer)

    def schedule(quantile):
        position = quantile * (HISTORY - 1)
        lower = int(position)
        return np.stack([
            np.logical_or.reduce([
                scores > ordered[:, lower] + (position - lower) * (ordered[:, lower + 1] - ordered[:, lower])
                for scores, ordered in layer
            ])
            for layer in traces
        ])

    target = len(layer_signals) * STEPS // 4
    lo, hi = 0.5, 0.99
    for _ in range(24):
        mid = (lo + hi) / 2
        if schedule(mid).sum() > target:
            lo = mid
        else:
            hi = mid
    quantile = min((lo, hi), key=lambda q: abs(schedule(q).sum() - target))
    return schedule(quantile).astype(np.uint8) * 2, quantile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("kind", choices=("active", "archive"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--causal", action="store_true", help="Generate prefix-only schedules and matched controls")
    parser.add_argument("--calibration-dir", type=Path, help="Location of the other full-attention capture for --causal")
    parser.add_argument("--prefill-history", action="store_true", help="Also calibrate LSE using the current prompt's final 64 prefill queries")
    parser.add_argument("--match-read-count", action="store_true", help="Fit cutoffs to 25%% overall on the replay trace")
    args = parser.parse_args()
    if (args.calibration_dir or args.prefill_history) and not args.causal:
        parser.error("--calibration-dir and --prefill-history require --causal")
    if args.match_read_count and not args.prefill_history:
        parser.error("--match-read-count requires --prefill-history")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_picks, all_routes = [], {}
    layer_signals = {mode: [] for mode in ("centroid", "reservoir", "lse", "union")}
    for layer in LAYERS:
        q, k = load_capture(args.capture_dir, args.kind, layer)
        rng = np.random.default_rng(2026 + layer)
        lse, scores = gate_scores(q, k, rng)
        all_picks.append(block_picks(q, k))
        routes = layer_routes(lse, scores, rng)
        if args.causal:
            other = "archive" if args.kind == "active" else "active"
            q_cal, k_cal = load_capture(args.calibration_dir or args.capture_dir, other, layer)
            calibration_lse, calibration = gate_scores(q_cal, k_cal, np.random.default_rng(2026 + layer))
            routes.update(causal_routes(lse, scores, calibration_lse, calibration))
            if args.prefill_history:
                prefill_q = load_prefill_queries(args.capture_dir, args.kind, layer)
                history = prefill_lse_history(prefill_q, k)
                observed = causal_routes(lse, scores, history, calibration)
                routes["causal-borrow-prefill"] = observed["causal-borrow-budget"]
                routes["causal-reserved-q95-prefill"] = observed["causal-reserved-q95"]
                centroid, centroid_prior = centroid_mass_scores(q, k, prefill_q)
                routes["causal-centroid-mass"] = rolling_reads(centroid, centroid_prior, 0.875).astype(np.uint8) * 2
                routes["causal-centroid-q75"] = rolling_reads(centroid, centroid_prior, 0.75).astype(np.uint8) * 2
                routes["causal-centroid-controller"] = control_centroid_reads(centroid, centroid_prior)
                layer_signals["centroid"].append([(centroid, centroid_prior)])
                if args.match_read_count:
                    layer_signals["reservoir"].append([(scores, calibration)])
                    layer_signals["lse"].append([(lse, history)])
                    layer_signals["union"].append([(scores, calibration), (lse, history)])
        for mode, route in routes.items():
            all_routes.setdefault(mode, []).append(route)
        print(f"Layer {layer}: {STEPS} C4 block lists per group", flush=True)
    np.stack(all_picks).tofile(args.output_dir / "picks.i32")
    for mode, route in all_routes.items():
        schedule = np.stack(route)
        schedule.tofile(args.output_dir / f"{mode}.u8")
        if mode.startswith("causal-"):
            counts = np.count_nonzero(schedule, axis=1)
            for seed in (71, 193, 804, 952, 2026):
                rng = np.random.default_rng(seed)
                control = np.zeros_like(schedule)
                for row, count in zip(control, counts):
                    row[rng.choice(STEPS, count, replace=False)] = 2
                control.tofile(args.output_dir / f"{mode}-random-{seed}.u8")
    if args.prefill_history:
        centroid_signals = [signals[0] for signals in layer_signals["centroid"]]
        for mode, refilling in (("causal-centroid-global", False), ("causal-centroid-refill", True)):
            control_total_centroid_reads(centroid_signals, refilling).tofile(args.output_dir / f"{mode}.u8")
    if args.match_read_count:
        for mode, traces in layer_signals.items():
            schedule, quantile = match_read_count(traces)
            schedule.tofile(args.output_dir / f"countmatched-{mode}.u8")
            print(f"{mode}: cutoff={quantile:.6f}, reads={(schedule != 0).mean():.3%}", flush=True)


if __name__ == "__main__":
    main()
