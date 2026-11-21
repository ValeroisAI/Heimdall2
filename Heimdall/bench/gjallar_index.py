"""Compare two-level block-bound selection with a flat scan on synthetic keys."""

import argparse
import json

import numpy as np


BLOCK_TOKENS = 64
BLOCKS_PER_PAGE = 64
SELECTED_BLOCKS = 16


def make_bounds(rng, pages: int, dim: int, center_scale: float):
    shape = (pages, BLOCKS_PER_PAGE, dim)
    minimum = np.empty(shape, dtype=np.float32)
    maximum = np.empty(shape, dtype=np.float32)
    for page in range(pages):
        keys = rng.standard_normal((BLOCKS_PER_PAGE, BLOCK_TOKENS, dim), dtype=np.float32)
        if center_scale:
            keys += center_scale * rng.standard_normal(dim, dtype=np.float32)
        minimum[page] = keys.min(axis=1)
        maximum[page] = keys.max(axis=1)
    return minimum, maximum


def score_bounds(queries, minimum, maximum):
    positive = np.maximum(queries, 0)
    negative = np.minimum(queries, 0)
    return positive @ maximum.T + negative @ minimum.T


def evaluate(queries, minimum, maximum, beam: int):
    pages, _, dim = minimum.shape
    page_upper = score_bounds(queries, minimum.min(axis=1), maximum.max(axis=1))
    block_upper = score_bounds(queries, minimum.reshape(-1, dim), maximum.reshape(-1, dim))
    assert np.all(block_upper.reshape(len(queries), pages, BLOCKS_PER_PAGE) <= page_upper[..., None] + 1e-10)

    overlaps, eligible_pages = [], []
    for page_scores, block_scores in zip(page_upper, block_upper):
        flat = np.argsort(-block_scores, kind="stable")[:SELECTED_BLOCKS]
        page_ids = np.argsort(-page_scores, kind="stable")[:beam]
        candidates = (page_ids[:, None] * BLOCKS_PER_PAGE + np.arange(BLOCKS_PER_PAGE)).ravel()
        sorted_candidates = candidates[np.lexsort((candidates, -block_scores[candidates]))]
        overlap = len(np.intersect1d(flat, sorted_candidates[:SELECTED_BLOCKS])) / SELECTED_BLOCKS
        overlaps.append(overlap)
        # Even with the final flat threshold known, an upper-bound traversal cannot rule these pages out.
        eligible_pages.append(int(np.count_nonzero(page_scores >= block_scores[flat[-1]])))

    return {
        "mean_top16_overlap": float(np.mean(overlaps)),
        "minimum_top16_overlap": float(min(overlaps)),
        "median_pages_above_exact_cutoff": float(np.median(eligible_pages)),
        "p95_pages_above_exact_cutoff": float(np.percentile(eligible_pages, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1_048_576)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pages, partial = divmod(args.tokens, BLOCK_TOKENS * BLOCKS_PER_PAGE)
    if partial or not 1 <= args.beam <= pages or min(args.head_dim, args.queries) < 1:
        parser.error("use complete pages, a beam in [1, pages], and positive dimensions and query count")

    rng = np.random.default_rng(args.seed)
    queries = rng.standard_normal((args.queries, args.head_dim))
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    for layout, center_scale in (("independent", 0.0), ("page_clustered", 1.0)):
        minimum, maximum = make_bounds(rng, pages, args.head_dim, center_scale)
        result = evaluate(queries, minimum, maximum, args.beam)
        print(json.dumps({"layout": layout, "pages": pages, **vars(args), **result}), flush=True)


if __name__ == "__main__":
    main()
