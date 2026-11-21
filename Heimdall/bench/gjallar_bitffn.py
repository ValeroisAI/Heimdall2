"""
Gjallar Experiment: Ternary vs. Binary BitFlip FFN Ablation & Zero-Sparsity Dynamics.

Evaluates:
  1. Dense FFN baseline (BF16 / FP32)
  2. Binary BitFlip FFN (bits=1, ±1)
  3. Ternary BitFlip FFN (bits=2, {-1, 0, +1})
  4. Ternary BitFlip FFN + Low-Rank Corrector (bits=2, rank=8)

Measures:
  - Training loss / convergence speed
  - Observed zero density (z) and theoretical BITCOS storage (2 - z bits/weight)
  - Memory efficiency during training

Usage:
    python bench/gjallar_bitffn.py
"""
import sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.bit import BitLinear, BitFlip, bit_stats

torch.set_num_threads(2)
torch.manual_seed(42)

D_MODEL = 128
D_FFN = 256
VOCAB = 256
SEQ_LEN = 64
BATCH_SIZE = 32
STEPS = 60


class ToyFFNModel(nn.Module):
    def __init__(self, mode="dense", rank=0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.mode = mode
        if mode == "dense":
            self.up = nn.Linear(D_MODEL, D_FFN, bias=False)
            self.gate = nn.Linear(D_MODEL, D_FFN, bias=False)
            self.down = nn.Linear(D_FFN, D_MODEL, bias=False)
        elif mode == "binary":
            self.up = BitLinear(D_MODEL, D_FFN, bits=1, rank=0)
            self.gate = BitLinear(D_MODEL, D_FFN, bits=1, rank=0)
            self.down = BitLinear(D_FFN, D_MODEL, bits=1, rank=0)
        elif mode == "ternary":
            self.up = BitLinear(D_MODEL, D_FFN, bits=2, rank=rank)
            self.gate = BitLinear(D_MODEL, D_FFN, bits=2, rank=rank)
            self.down = BitLinear(D_FFN, D_MODEL, bits=2, rank=rank)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x):
        h = self.embed(x)
        # SwiGLU FFN
        if self.mode == "dense":
            ffn = F.silu(self.gate(h)) * self.up(h)
            h = h + self.down(ffn)
        else:
            ffn = F.silu(self.gate(h)) * self.up(h)
            h = h + self.down(ffn)
        h = self.norm(h)
        return self.head(h)


def run_experiment(mode="dense", rank=0):
    torch.manual_seed(42)
    model = ToyFFNModel(mode=mode, rank=rank)

    # Optimizer setup
    if mode == "dense":
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        bit_opt = None
    else:
        # Separate bit weights from standard parameters
        bit_params = [p for p in model.parameters() if getattr(p, "bit", None)]
        other_params = [p for p in model.parameters() if not getattr(p, "bit", None)]
        opt = torch.optim.AdamW(other_params, lr=3e-3)
        bit_opt = BitFlip(bit_params, lr=0.01, momentum=0.9)

    # Synthetic Markov / copy prediction stream
    # Generates patterned synthetic tokens where next token depends on pattern
    t0 = time.time()
    losses = []
    for step in range(STEPS):
        x = torch.randint(0, VOCAB, (BATCH_SIZE, SEQ_LEN))
        # Pattern: target is token shifted by 1 with modulo rule
        y = (x * 7 + 13) % VOCAB

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        opt.zero_grad(set_to_none=True)
        if bit_opt:
            bit_opt.step()
            bit_opt.zero_grad(set_to_none=True)

        losses.append(loss.item())

    elapsed = time.time() - t0
    final_loss = sum(losses[-10:]) / 10.0

    stats = bit_stats(model)
    zero_ratio = stats.get("sifir_orani", 0.0)
    # BITCOS storage: 2 - z bits per weight
    bitcos_bits = 2.0 - zero_ratio if mode == "ternary" else (1.0 if mode == "binary" else 16.0)

    return {
        "mode": mode,
        "rank": rank,
        "final_loss": round(final_loss, 4),
        "zero_ratio": round(zero_ratio, 4),
        "bitcos_bits": round(bitcos_bits, 3),
        "elapsed_s": round(elapsed, 2)
    }


if __name__ == "__main__":
    configs = [
        ("dense", 0),
        ("binary", 0),
        ("ternary", 0),
        ("ternary", 8),
    ]
    print(f"{'Mode':<18} {'Rank':<6} {'Final Loss':<12} {'Zero Frac (z)':<15} {'BITCOS Bits':<14} {'Time (s)':<8}")
    print("-" * 75)
    for mode, rank in configs:
        res = run_experiment(mode, rank)
        name = f"{mode}" + (f" (LoRA-{rank})" if rank else "")
        print(f"{name:<18} {res['rank']:<6} {res['final_loss']:<12.4f} {res['zero_ratio']:<15.3f} {res['bitcos_bits']:<14.3f} {res['elapsed_s']:<8.2f}")
