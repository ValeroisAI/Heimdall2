"""
sft_egit.py — Önceden eğitilmiş Heimdall modeli için kısa SFT (Supervised Fine-Tuning) aşaması (~20-30 dk).

Özellikler:
  1. ChatML sohbet şablonu: <|im_start|>system...<|im_start|>user...<|im_start|>assistant...<|im_end|>
  2. Asistan tokenlerine maskelenmiş kayıp (Loss Masking): Sistem ve kullanıcı girdilerinin hedefleri
     -100 yapılarak gradyanların yalnızca asistan yanıtından akması sağlanır.
  3. Süre odaklı çalışma: Belirlenen dakika boyunca çalışır (örn. --minutes 25) ve WSD / Cosine LR uygular.
  4. Kayıt & Dışa Aktarım: runs/gece_sft/last.pt periyodik (her birkaç dakikada bir) yazılır,
     non-finite (NaN/inf) durumunda adım atlanır ve 5 ardışıkta 75 çıkış koduyla durulur;
     bozuk ağırlık asla kaydedilmez.
"""

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.io import DEFAULT_TOKENIZER, load_model, save_bundle  # noqa: E402
from heimdall.optim import build_optimizers, set_lr  # noqa: E402

NONFINITE_EXIT = 75


def parse_args():
    p = argparse.ArgumentParser(description="Heimdall SFT Eğitimi")
    p.add_argument("--ckpt", required=True, help="Ön eğitimli model (.pt ya da bundle dizini)")
    p.add_argument("--train-data", default="veri/sft/train_metin.jsonl", help="OpenAI formatında JSONL")
    p.add_argument("--eval-data", default="veri/sft/eval_metin.jsonl", help="Doğrulama JSONL")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--out", default="runs/gece_sft")
    p.add_argument("--minutes", type=float, default=25.0, help="Eğitim süresi (dakika)")
    p.add_argument("--save-minutes", type=float, default=3.0, help="Periyodik checkpoint kayıt aralığı (dakika)")
    p.add_argument("--max-steps", type=int, default=0, help="0 ise süreye göre belirlenir")
    p.add_argument("--seq-len", type=int, default=1024, help="Maksimum dizi uzunluğu")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=4, help="Gradyan biriktirme adımı (toplam toplu iş: micro*accum)")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--no-compile", action="store_true", help="torch.compile'ı kapat")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def format_chatml_and_mask(messages: List[dict], tok: Tokenizer, max_len: int) -> Tuple[List[int], List[int]]:
    """
    Sohbet mesajlarını ChatML biçimine getirir ve asistan dışındaki tüm tokenleri -100 ile maskeler.
    Döndürür: (input_ids, target_ids)
    """
    full_tokens = []
    full_targets = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue

        if role == "system":
            prefix = f"<|im_start|>system\n{content}<|im_end|>\n"
            t_ids = tok.encode(prefix).ids
            full_tokens.extend(t_ids)
            full_targets.extend([-100] * len(t_ids))
        elif role == "user":
            prefix = f"<|im_start|>user\n{content}<|im_end|>\n"
            t_ids = tok.encode(prefix).ids
            full_tokens.extend(t_ids)
            full_targets.extend([-100] * len(t_ids))
        elif role == "assistant":
            header = "<|im_start|>assistant\n"
            body = f"{content}<|im_end|>\n"
            h_ids = tok.encode(header).ids
            b_ids = tok.encode(body).ids

            # Başlık kısmı maskelenir (prompt sayılır)
            full_tokens.extend(h_ids)
            full_targets.extend([-100] * len(h_ids))

            # Yalnızca asistanın ürettiği içerik ve bitiş etiketi hesaplanır
            full_tokens.extend(b_ids)
            full_targets.extend(b_ids)

    if len(full_tokens) < 2:
        return [], []

    # max_len sınırla
    full_tokens = full_tokens[:max_len + 1]
    full_targets = full_targets[:max_len + 1]

    # Causal LM: input = tokens[:-1], target = targets[1:]
    inp = full_tokens[:-1]
    tgt = full_targets[1:]
    return inp, tgt


class SFTDataset:
    def __init__(self, path: str, tok: Tokenizer, max_len: int, seed: int = 42):
        self.samples = []
        p = Path(path)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    ms = row.get("messages", [])
                    if not ms:
                        continue
                    inp, tgt = format_chatml_and_mask(ms, tok, max_len)
                    # Yalnızca en az bir aktif asistan hedefi içerenleri al
                    if inp and any(t != -100 for t in tgt):
                        self.samples.append((inp, tgt))
                except Exception:
                    continue
        self.rng = np.random.default_rng(seed)
        self.indices = list(range(len(self.samples)))
        self.rng.shuffle(self.indices)
        self.pos = 0

    def __len__(self):
        return len(self.samples)

    def next_batch(self, batch_size: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.samples:
            raise RuntimeError("Veri kümesi boş!")
        batch_items = []
        for _ in range(batch_size):
            if self.pos >= len(self.indices):
                self.rng.shuffle(self.indices)
                self.pos = 0
            idx = self.indices[self.pos]
            self.pos += 1
            batch_items.append(self.samples[idx])

        max_l = max(len(inp) for inp, _ in batch_items)
        x_pad = torch.full((batch_size, max_l), pad_id, dtype=torch.long)
        y_pad = torch.full((batch_size, max_l), -100, dtype=torch.long)

        for i, (inp, tgt) in enumerate(batch_items):
            x_pad[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            y_pad[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)

        return x_pad, y_pad


@torch.no_grad()
def evaluate_sft(model, dataset: SFTDataset, pad_id: int, device, amp, num_batches: int = 16) -> float:
    if len(dataset) == 0:
        return 0.0
    model.eval()
    tot_loss = 0.0
    tot_tokens = 0
    for _ in range(num_batches):
        x, y = dataset.next_batch(4, pad_id)
        x, y = x.to(device), y.to(device)
        with amp:
            loss = model(x, y)
        active = (y != -100).sum().item()
        if active > 0:
            tot_loss += loss.item() * active
            tot_tokens += active
    model.train()
    return tot_loss / max(tot_tokens, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"SFT Başlıyor: Model={args.ckpt} | Cihaz={device} ({'bf16' if bf16 else 'fp32'})")
    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("<pad>") or 0

    # Model yükle
    model = load_model(args.ckpt, device=device)
    model.train()

    # Veri yükle
    print(f"Eğitim verisi okunuyor: {args.train_data}...")
    train_ds = SFTDataset(args.train_data, tok, args.seq_len, seed=args.seed)
    print(f"  Bulunan geçerli örnek sayısı: {len(train_ds):,}")
    if len(train_ds) == 0:
        sys.exit(f"HATA: {args.train_data} içinde geçerli SFT örneği bulunamadı!")

    eval_ds = SFTDataset(args.eval_data, tok, args.seq_len, seed=args.seed + 1)
    print(f"Doğrulama verisi: {len(eval_ds):,} örnek")

    # Optimizer (AdamW + varsa BitFlip)
    optimizers = build_optimizers(model, lr_muon=args.lr, lr_adam=args.lr, use_muon=False, flip_rate=1e-3)
    clip_params = [p for p in model.parameters() if not getattr(p, "bit", 0)]

    train_model = torch.compile(model) if (not args.no_compile and device.type != "cpu") else model

    # Süre ve adım hesabı
    max_seconds = args.minutes * 60.0
    start_time = time.time()
    end_time = start_time + max_seconds

    # Yaklaşık adım sayısı tahmini (warmup ve lr decay için)
    est_total_steps = args.max_steps if args.max_steps > 0 else max(200, int(max_seconds / 2.0))
    warmup_steps = max(10, int(est_total_steps * args.warmup_ratio))

    def get_lr_scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        # Cosine decay to 0.1
        progress = float(step - warmup_steps) / float(max(1, est_total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    step = 0
    trained_tokens = 0
    last_log_time = time.time()
    last_save_time = time.monotonic()
    save_interval = args.save_minutes * 60.0
    consecutive_nan = 0
    log_file = open(out_dir / "sft_log.jsonl", "a", encoding="utf-8")

    def save_checkpoint(is_final: bool = False):
        state = {
            "config": model.cfg.to_dict(),
            "model": model.state_dict(),
            "step": step,
            "tokens": trained_tokens,
            "sft": True
        }
        tmp_path = out_dir / ".last.tmp"
        torch.save(state, tmp_path)
        tmp_path.replace(out_dir / "last.pt")
        if is_final:
            print(f"\nSFT Checkpoint kaydedildi: {out_dir / 'last.pt'}")

    print(f"Hedef süre: {args.minutes:.1f} dakika (~{est_total_steps} adım) | LR: {args.lr} | Kayıt aralığı: {args.save_minutes:.1f} dk")

    try:
        while True:
            now = time.time()
            if now >= end_time:
                print(f"\nSüre doldu ({args.minutes:.1f} dk). Eğitim tamamlanıyor.")
                break
            if args.max_steps > 0 and step >= args.max_steps:
                print(f"\nAzami adım ({args.max_steps}) tamamlandı.")
                break

            scale = get_lr_scale(step)
            set_lr(optimizers, scale)

            loss_acc = 0.0
            step_active_tokens = 0

            for _ in range(args.accum):
                x, y = train_ds.next_batch(args.micro_batch, pad_id)
                x, y = x.to(device), y.to(device)
                with amp:
                    loss = train_model(x, y)
                (loss / args.accum).backward()
                loss_acc += float(loss.detach()) / args.accum
                step_active_tokens += int((y != -100).sum().item())

            gnorm = torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            checks = [loss_acc, gnorm]
            if not bool(torch.isfinite(torch.stack(checks)).all()):
                consecutive_nan += 1
                print(f"UYARI: Adım {step}: Kayıp ({float(loss_acc):.4f}) veya gradyan ({float(gnorm):.4f}) sonlu değil! "
                      f"(ardışık: {consecutive_nan}/5)", file=sys.stderr)
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                if consecutive_nan >= 5:
                    print("HATA: 5 ardışık non-finite adım tespit edildi. Bozuk ağırlıklar kaydedilmeden çıkılıyor.", file=sys.stderr)
                    sys.exit(NONFINITE_EXIT)
                continue

            consecutive_nan = 0
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            trained_tokens += step_active_tokens

            # Periyodik kayıt (her save_minutes dakikada bir)
            if time.monotonic() - last_save_time >= save_interval:
                save_checkpoint()
                last_save_time = time.monotonic()

            # Loglama
            if step % 10 == 0 or time.time() - last_log_time > 15.0:
                elapsed = time.time() - start_time
                remain_min = max(0.0, (end_time - time.time()) / 60.0)
                tok_per_sec = trained_tokens / max(elapsed, 0.1)
                cur_lr = args.lr * scale
                print(f"Adım {step:4d} | Kayıp: {loss_acc:6.4f} | LR: {cur_lr:.2e} | "
                      f"Hedef Token: {trained_tokens / 1e6:5.2f}M ({tok_per_sec / 1e3:4.1f}K/s) | Kalan: {remain_min:4.1f} dk",
                      flush=True)
                log_file.write(json.dumps({"step": step, "loss": loss_acc, "lr": cur_lr,
                                           "tokens": trained_tokens, "elapsed": elapsed}) + "\n")
                log_file.flush()
                last_log_time = time.time()

            # Doğrulama
            if step % args.eval_every == 0 and len(eval_ds) > 0:
                eval_loss = evaluate_sft(model, eval_ds, pad_id, device, amp)
                print(f"  >>> [Doğrulama {step}] Maskeli Asistan Kaybı: {eval_loss:6.4f} (ppl: {math.exp(min(eval_loss, 20)):.2f})", flush=True)

    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu, son geçerli ağırlıklar kaydediliyor...")

    log_file.close()

    # 1. Final Checkpoint kaydet (.pt)
    save_checkpoint(is_final=True)

    # 2. Servis paketi dışa aktar (export/)
    export_dir = out_dir / "export"
    save_bundle(model, export_dir, tokenizer=args.tokenizer,
                meta={"step": step, "sft_tokens": trained_tokens, "base_ckpt": args.ckpt})
    print(f"Servis paketi hazırlandı: {export_dir}")
    print("SFT Aşaması Başarıyla Tamamlandı.")


if __name__ == "__main__":
    main()
