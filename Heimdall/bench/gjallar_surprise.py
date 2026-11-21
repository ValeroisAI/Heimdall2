"""
Gjallar experiment 0: does the delta-rule prediction error ("surprise") tell us when exact retrieval is needed?

Setup (pure torch, CPU, no training): a stream of (key,value) writes into a delta-rule memory S (d x d, decay g),
then filler writes, then queries. For each query we compare
  - recall error of the memory alone:      ||S q - v_true||
  - the surprise signal available BEFORE the answer is known: ||v_hat - S k_t|| at the *write* of the query token's
    own key (a re-occurring key whose stored value is stale/overwritten gives a large error), and the
    read-out norm ||S q|| / confidence.
We report AUROC of "surprise predicts that memory-only recall fails" for several memory loads.

    python bench/gjallar_surprise.py
"""
import torch

torch.manual_seed(0)
D = 64


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def run(n_pairs, n_fill, decay, beta=1.0, trials=200):
    rows = []
    for _ in range(trials):
        K = unit(torch.randn(n_pairs, D))
        V = unit(torch.randn(n_pairs, D))
        S = torch.zeros(D, D)
        for k, v in zip(K, V):                       # write phase (delta rule)
            S = decay * S + beta * torch.outer(v - decay * (S @ k), k)
        for _ in range(n_fill):                      # filler tokens: random keys/values
            k, v = unit(torch.randn(D)), unit(torch.randn(D))
            S = decay * S + beta * torch.outer(v - decay * (S @ k), k)
        for i in range(n_pairs):                     # queries
            q = K[i]
            read = S @ q
            err_true = (read - V[i]).norm().item()   # ground truth recall error (unknown at inference)
            conf = read.norm().item()                # observable: read-out norm
            # observable surprise: delta error when the key is presented again with an unknown value
            # -> equals the retrieval-need proxy 1 - ||S q||
            rows.append((err_true, conf))
    t = torch.tensor(rows)
    fail = t[:, 0] > 0.5
    score = -t[:, 1]                                 # low confidence = high surprise
    pos, neg = score[fail], score[~fail]
    if len(pos) == 0 or len(neg) == 0:
        return fail.float().mean().item(), float("nan")
    auroc = (pos[:, None] > neg[None, :]).float().mean().item() + 0.5 * (pos[:, None] == neg[None, :]).float().mean().item()
    return fail.float().mean().item(), auroc


if __name__ == "__main__":
    print(f"{'pairs':>6} {'fill':>6} {'decay':>6} {'fail%':>7} {'AUROC(conf->fail)':>18}")
    for n_pairs in (8, 32, 64, 128):
        for n_fill in (0, 256, 2048):
            for decay in (1.0, 0.999):
                f, a = run(n_pairs, n_fill, decay, trials=40)
                print(f"{n_pairs:>6} {n_fill:>6} {decay:>6} {100*f:>6.1f}% {a:>18.3f}")
