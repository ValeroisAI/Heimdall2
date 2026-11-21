"""
Heimdall eğitimi (CUDA / ROCm / CPU).

    python -m heimdall.train --preset temel --data ../stream_coder_100k.bin --tokens 2e9 --out runs/temel
    python -m heimdall.train --preset temel --arch transformer ... --out runs/temel_tf     # karşılaştırma tabanı
    python -m heimdall.train --resume runs/temel/last.pt                                 # devam
"""

import os

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # RDNA'da flash/mem-efficient SDPA

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

from . import kernels
from .config import ARCHS, PRESETS, HeimdallConfig, apply_arch
from .bit import BitFlip, bit_stats, set_bit_kernel
from .data import Loader, Mixture
from .model import HeimdallLM
from .optim import build_optimizers, set_lr, wsd


# The overnight launcher backs off learning rates when this exit code is returned.
NONFINITE_EXIT = 75


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Heimdall eğitimi")
    p.add_argument("--preset", default="temel", choices=list(PRESETS))
    p.add_argument("--arch", choices=ARCHS, help="ön ayarın mimarisini değiştir")
    p.add_argument("--data", nargs="+", help="uint16 .bin dosyaları (yol[:ağırlık])")
    p.add_argument("--val-frac", type=float, default=0.005)
    p.add_argument("--tokens", type=float, help="token sınırı (süre sınırı yoksa varsayılan: 1e9)")
    p.add_argument("--wall-start", type=float, help="süreli WSD başlangıcı (Unix saniye)")
    p.add_argument("--deadline", type=float, help="süreli eğitimin bitişi (Unix saniye)")
    p.add_argument("--seq-len", type=int)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--batch-tokens", type=int)
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--layout", help="katman düzeni, ör. DDDA, DDA, A")
    p.add_argument("--attn-window", type=int)
    p.add_argument("--bits", type=int, choices=[0, 1, 2], help="gizli ağırlıksız: 1 = ikili (±1), 2 = üçlü (−1/0/+1)")
    p.add_argument("--bit-scope", choices=["all", "ffn", "mix"], help="--bits: hangi matrisler bit (ffn = yalnız SwiGLU)")
    p.add_argument("--bit-rank", type=int, help="--bits: bit matrislere yoğun düşük-rank düzeltici (ör. 16)")
    p.add_argument("--bit-kernel", choices=["off", "int8"], default="off",
                   help="--bits: Triton int8 GEMM (eğitim/prefill). Önce bench/bitkernel.py ile hızı ölç")
    p.add_argument("--flip-orani", type=float, default=5e-3, help="--bits: adım başına çevrilen ağırlık oranı (tepe)")
    p.add_argument("--mtp-depth", type=int, help="MTP modül sayısı (0 kapalı; 1-2 önerilir, küçük modelde önce kısa deney)")
    p.add_argument("--mtp-weight", type=float, help="MTP ek kaybının ağırlığı (varsayılan 0.3)")
    p.add_argument("--lr-muon", type=float)
    p.add_argument("--lr-adam", type=float)
    p.add_argument("--adamw-only", action="store_true")
    p.add_argument("--warmup", type=float, default=0.01)
    p.add_argument("--decay-start", type=float, default=0.75)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-ckpt", action="store_true", default=None)
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--kernel", choices=["auto", "torch"], default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="runs/heimdall")
    p.add_argument("--resume")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--save-minutes", type=float, default=0, help="dakika bazlı checkpoint aralığı (0 kapalı)")
    p.add_argument("--retry-index", type=int, default=0, help="sürdürme sırasında yeni veri örnekleme dizisi")
    p.add_argument("--lr-scale", type=float, default=1.0, help="checkpoint sonrası tepe öğrenme hızı çarpanı")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_config(args) -> HeimdallConfig:
    preset = PRESETS[args.preset]
    cfg = apply_arch(preset.model, args.arch or preset.arch)
    bits = preset.bits if args.arch is None and args.bits is None else args.bits
    over = {k: v for k, v in dict(d_model=args.d_model, n_layers=args.n_layers, layout=args.layout,
                                  attn_window=args.attn_window, kernel=args.kernel, bits=bits,
                                  bit_scope=args.bit_scope, bit_rank=args.bit_rank, mtp_depth=args.mtp_depth,
                                  mtp_weight=args.mtp_weight).items() if v is not None}
    return replace(cfg, **over)


def flops_per_token(cfg: HeimdallConfig, n_matmul: int, t: int) -> float:
    """6·N + karıştırıcı (ileri+geri ≈ 3× ileri)."""
    f = 6 * n_matmul
    for kind in cfg.kinds:
        if kind == "A":
            span = min(cfg.attn_window or t, t)
            f += 3 * 2 * 2 * cfg.attn_heads * cfg.attn_head_dim * span / (1 if cfg.attn_window else 2)
        else:
            f += 3 * 2 * 3 * cfg.n_heads * cfg.head_dim * (cfg.chunk_size + cfg.head_dim)
    return f


@torch.no_grad()
def evaluate(model, val, micro, device, amp) -> dict:
    model.eval()
    out = {}
    for name, ids in val:
        tot = 0.0
        for i in range(0, ids.size(0), micro):
            chunk = ids[i:i + micro].to(device)
            with amp:
                tot += model(chunk[:, :-1], chunk[:, 1:]).item() * chunk.size(0)
        out[name] = tot / ids.size(0)
    model.train()
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        given = {a.split("=")[0] for a in (sys.argv[1:] if argv is None else argv) if a.startswith("--")}
        if "--no-grad-ckpt" in given:
            given.add("--grad-ckpt")
        if "--no-compile" in given:
            given.add("--compile")
        for k, v in ckpt["args"].items():
            if k not in ("resume", "device") and "--" + k.replace("_", "-") not in given:
                setattr(args, k, v)
    if not args.data:
        sys.exit("--data gerekli")
    timed = args.deadline is not None
    if timed or args.wall_start is not None:
        if (not timed or args.wall_start is None or not math.isfinite(args.wall_start)
                or not math.isfinite(args.deadline) or args.wall_start >= args.deadline):
            sys.exit("--wall-start < --deadline gerekli (Unix saniye)")
        if time.time() >= args.deadline:
            print("Eğitim süresi doldu.", flush=True)
            return
    if args.tokens is not None and (not math.isfinite(args.tokens) or args.tokens <= 0):
        sys.exit("--tokens pozitif ve sonlu olmalı")
    if not math.isfinite(args.save_minutes) or args.save_minutes < 0:
        sys.exit("--save-minutes sonlu ve >= 0 olmalı")
    if args.retry_index < 0 or not math.isfinite(args.lr_scale) or not 0 < args.lr_scale <= 1:
        sys.exit("--retry-index >= 0 ve 0 < --lr-scale <= 1 gerekli")

    preset = PRESETS[args.preset]
    cfg = HeimdallConfig(**ckpt["config"]) if ckpt else build_config(args)
    args.arch = args.arch or preset.arch
    seq_len = args.seq_len or preset.seq_len
    micro = args.micro_batch or preset.micro_batch
    batch_tokens = args.batch_tokens or preset.batch_tokens
    accum = max(1, batch_tokens // (micro * seq_len))
    lr_muon, lr_adam = args.lr_muon or preset.lr_muon, args.lr_adam or preset.lr_adam
    grad_ckpt = preset.grad_ckpt if args.grad_ckpt is None else args.grad_ckpt

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    backend = kernels.prepare(device, cfg.kernel)

    model = HeimdallLM(cfg).to(device)
    model.grad_ckpt = grad_ckpt
    set_bit_kernel(model, args.bit_kernel)
    optimizers = build_optimizers(model, lr_muon, lr_adam, use_muon=not args.adamw_only, flip_rate=args.flip_orani)
    flipper = next((o for o in optimizers if isinstance(o, BitFlip)), None)
    clip_params = [p for p in model.parameters() if not getattr(p, "bit", 0)]   # bit ağırlıkları kendi optimizer'ında
    bit_params = [p for p in model.parameters() if getattr(p, "bit", 0)]
    step, tokens = 0, 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        for opt, st in zip(optimizers, ckpt["optim"]):
            opt.load_state_dict(st)
        step, tokens = ckpt["step"], ckpt["tokens"]
        del ckpt

    mix = Mixture(args.data, args.val_frac, seed=args.seed + step + args.retry_index * 1_000_000_007)
    for path, _, split in mix.sources:
        if split <= seq_len + 1:
            sys.exit(f"{path}: eğitim bölümü {seq_len + 2:,} tokendan kısa")
    val = mix.val_batches(seq_len, args.eval_batches)
    tokens_per_step = accum * micro * seq_len
    budget = args.tokens if timed else (args.tokens or 1e9)
    total = max(1, int(budget // tokens_per_step)) if budget is not None else None
    step_limit = f"{total:,}" if total is not None else "süreli"
    n_matmul = model.active_params() + model.lm_head.weight.numel()
    train_model = torch.compile(model) if args.compile and device.type != "cpu" else model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "args": vars(args)}, indent=2))
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    hip = f" ROCm {torch.version.hip}" if torch.version.hip else ""
    print(f"Heimdall [{args.arch}, düzen {cfg.layout}] {model.num_params() / 1e6:.1f}M param ({model.active_params() / 1e6:.0f}M aktif) | d={cfg.d_model} "
          f"L={cfg.n_layers} | {gpu}{hip} | {'bf16' if bf16 else 'fp32'} | delta çekirdeği: {backend}")
    print(f"T={seq_len} mikro={micro} birikim={accum} → adım başına {tokens_per_step:,} token, {step_limit} adım "
          f"| Muon {lr_muon} AdamW {lr_adam} | veri: {mix.describe()}", flush=True)

    def save() -> None:
        state = {"config": cfg.to_dict(), "model": model.state_dict(), "optim": [o.state_dict() for o in optimizers],
                 "step": step, "tokens": tokens, "args": vars(args)}
        torch.save(state, out / ".last.tmp")
        (out / ".last.tmp").replace(out / "last.pt")

    loader = Loader(mix, micro, seq_len, device)
    model.train()
    t0 = last_save = time.monotonic()
    tok0 = 0
    initial_step = saved_step = step
    safe_to_save = True
    logf = open(out / "log.jsonl", "a")
    try:
        while total is None or step < total:
            now = time.time()
            if timed and now >= args.deadline:
                break
            progress = step / total if total is not None else 0.0
            if timed:
                progress = max(progress, (now - args.wall_start) / (args.deadline - args.wall_start))
            lr_factor = args.lr_scale * wsd(progress, args.warmup, args.decay_start)
            set_lr(optimizers, lr_factor)
            loss_acc = 0.0
            for _ in range(accum):
                x, y = loader.next()
                with amp:
                    loss = train_model(x, y)
                (loss / accum).backward()
                loss_acc += loss.detach().float() / accum
            gnorm = torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            checks = [loss_acc, gnorm]
            if bit_params:
                checks.append(torch.nn.utils.get_total_norm(
                    [p.grad for p in bit_params if p.grad is not None], norm_type=float("inf")).to(device))
            if not bool(torch.isfinite(torch.stack(checks)).all()):
                print(f"adım {step}: kayıp veya gradyan sonlu değil; son iyi checkpoint korunuyor.", file=sys.stderr)
                raise SystemExit(NONFINITE_EXIT)
            safe_to_save = False
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            tokens += tokens_per_step
            tok0 += tokens_per_step
            safe_to_save = True
            finished = step == total or (timed and time.time() >= args.deadline)
            if step % args.log_every == 0 or finished:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.monotonic() - t0
                tps = tok0 / dt
                rec = {"step": step, "tokens": tokens, "loss": float(loss_acc), "grad_norm": float(gnorm), "tok_s": tps,
                       "tflops": tps * flops_per_token(cfg, n_matmul, seq_len) / 1e12,
                       "mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
                       "lr_factor": lr_factor}
                extra = ""
                if flipper is not None:
                    rec["flip"] = flipper.last_flip_frac
                    rec["sifir_orani"] = bit_stats(model)["sifir_orani"]
                    extra = f" | flip {100 * rec['flip']:.3f}% sıfır {100 * rec['sifir_orani']:.0f}%"
                print(f"adım {step:6d}/{step_limit} | loss {rec['loss']:.4f} | {tps:,.0f} tok/s | ~{rec['tflops']:.1f} TFLOPS"
                      f" | grad {rec['grad_norm']:.2f} | {rec['mem_gb']:.1f} GB | {tokens / 1e6:,.0f}M tok" + extra, flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                t0, tok0 = time.monotonic(), 0
            if (finished or (timed and step == 1) or (args.save_every and step % args.save_every == 0)
                    or (args.save_minutes and time.monotonic() - last_save >= args.save_minutes * 60)):
                save_started = time.monotonic()
                save()
                saved_step = step
                last_save = time.monotonic()
                t0 += last_save - save_started
            if args.eval_every and (step % args.eval_every == 0 or finished):
                eval_started = time.monotonic()
                v = evaluate(model, val, micro, device, amp)
                print("  [val] " + " | ".join(f"{k}: {x:.4f} (ppl {math.exp(x):.1f})" for k, x in v.items()), flush=True)
                logf.write(json.dumps({"step": step, "tokens": tokens, "val": v}) + "\n")
                logf.flush()
                t0 += time.monotonic() - eval_started
    except KeyboardInterrupt:
        if safe_to_save and step > saved_step:
            save()
        if timed:
            raise SystemExit(130)
    else:
        if step > saved_step:
            save()
    finally:
        loader.stop = True
        logf.close()
    if step > initial_step:
        print(f"Kaydedildi: {out / 'last.pt'} (adım {step}, {tokens / 1e6:,.0f}M token)", flush=True)


if __name__ == "__main__":
    main()
