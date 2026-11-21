"""
ornek_uret.py — 10 sabit, çeşitli ve etkileyici istem (prompt) üzerinden metin üretir ve Markdown dosyasına yazar.

İstem alanları:
  1. Python Algoritma (Palindrom / String)
  2. Sistem Programlama (Sanal Bellek & Sayfalama)
  3. Adım Adım Matematik / Akıl Yürütme (Havuz problemi)
  4. Yaratıcı Yazarlık (Gece çalışan saat ustası)
  5. ChatML Sohbet (TCP vs UDP)
  6. Yapılandırılmış JSON Çıkarma
  7. Mantık Bulmacası (Etiketli kutular)
  8. Nesne Yönelimli Tasarım (LRU Cache O(1))
  9. Fizik / Bilim (Rayleigh Saçılması)
 10. Türkçe Alan Testi (Dikkat Mekanizması)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.io import DEFAULT_TOKENIZER, load_model  # noqa: E402

PROMPTS: List[Tuple[str, str, str]] = [
    (
        "1. Algoritma & Python",
        "Palindromik Alt Dizi",
        "Implement a Python function `def find_longest_palindromic_substring(s: str) -> str:` with a brief explanation and test cases:\n```python\n"
    ),
    (
        "2. Sistem Programlama & İşletim Sistemleri",
        "Sanal Bellek ve Sayfalama",
        "Explain how virtual memory, page tables, and TLBs (Translation Lookaside Buffers) work together in modern operating systems:"
    ),
    (
        "3. Matematiksel Akıl Yürütme",
        "Havuz ve Boru Problemi",
        "Question: A water tank has two inlet pipes A and B, and one outlet pipe C. Pipe A fills the tank in 4 hours, Pipe B in 6 hours, and Pipe C empties it in 3 hours. If all three pipes are opened simultaneously, how long does it take to completely fill the empty tank? Show each step of calculation clearly.\nSolution:"
    ),
    (
        "4. Yaratıcı & Edebi Yazı",
        "Gece Çalışan Saat Ustası",
        "Write a vivid, atmospheric short paragraph about an elderly watchmaker working late at night in a silent workshop surrounded by rhythmic ticking clocks:"
    ),
    (
        "5. ChatML Sohbet Formatı",
        "Ağ Protokolleri (TCP vs UDP)",
        "<|im_start|>system\nYou are Heimdall, a precise and knowledgeable AI assistant created by Valerois.<|im_end|>\n<|im_start|>user\nWhat is the fundamental difference between TCP and UDP, and in what scenarios would a developer choose UDP over TCP?<|im_end|>\n<|im_start|>assistant\n"
    ),
    (
        "6. Bilgi Çıkarımı (Structured JSON)",
        "Metinden JSON Çıkarma",
        "Extract the person's name, role, organization, and email address from the text below into valid JSON:\n\nText: 'Dr. Elena Rostova is the Principal Cryptographer at QuantumSphere Labs. For vulnerability disclosures, contact her team at elena.rostova@quantumsphere.org.'\n\nJSON:\n```json\n"
    ),
    (
        "7. Mantık & Problem Çözme",
        "Yanlış Etiketli Kutular Bulmacası",
        "Puzzle: Three boxes are labeled 'Apples', 'Oranges', and 'Apples & Oranges'. You are told that EVERY box is labeled incorrectly. You may draw exactly one fruit from one box of your choice without looking inside. How can you determine the correct label of all three boxes? Walk through the logic step by step.\nAnswer:"
    ),
    (
        "8. Veri Yapıları & Nesneye Yönelik Tasarım",
        "O(1) LRU Cache",
        "Write a complete Python implementation of an LRU (Least Recently Used) Cache supporting `get(key)` and `put(key, value)` operations in O(1) time complexity using a doubly-linked list and a dictionary:\n```python\n"
    ),
    (
        "9. Fizik & Doğa Bilimleri",
        "Gökyüzünün Rengi ve Rayleigh Saçılması",
        "Explain why the sky appears blue in the middle of the day, but turns orange and red during sunrise and sunset, referencing the physics of Rayleigh scattering:"
    ),
    (
        "10. Türkçe & Çok Dilli Yetenek",
        "Dikkat Mekanizması (Attention)",
        "<|im_start|>system\nSen Valerois tarafından geliştirilmiş Heimdall yapay zeka modelisin. Soruları net ve anlaşılır Türkçe ile yanıtla.<|im_end|>\n<|im_start|>user\nDerin öğrenmede 'dikkat mekanizması' (attention) nedir ve geleneksel tekrarlayan sinir ağlarına (RNN) kıyasla neden çığır açmıştır?<|im_end|>\n<|im_start|>assistant\n"
    ),
]


def sample_token(logits: torch.Tensor, temp: float = 0.7, top_p: float = 0.9) -> int:
    if temp <= 0:
        return int(logits.argmax())
    logits = logits.float() / temp
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    return int(torch.multinomial(probs / probs.sum(), 1))


def generate_response(model, tok: Tokenizer, prompt: str, max_new: int, temp: float, device, amp) -> Tuple[str, float]:
    ids = tok.encode(prompt).ids
    input_tensor = torch.tensor([ids], device=device)
    eos_id = tok.token_to_id("<eos>")
    im_end_id = tok.token_to_id("<|im_end|>")

    generated = []
    t0 = time.time()

    with torch.no_grad(), amp:
        cache = model.new_cache(1)
        logits = model(input_tensor, cache=cache, last_only=True)[0, -1]
        for _ in range(max_new):
            nxt = sample_token(logits, temp=temp)
            if nxt in (eos_id, im_end_id):
                break
            generated.append(nxt)
            logits = model(torch.tensor([[nxt]], device=device), cache=cache)[0, -1]

    elapsed = max(time.time() - t0, 1e-4)
    speed = len(generated) / elapsed
    text = tok.decode(generated)
    return text.strip(), speed


def main():
    p = argparse.ArgumentParser(description="10 Sabit Örnek İstem Değerlendirmesi")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-md", default="runs/degerlendirme/ornek_uretilenler.md")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    print(f"Model yükleniyor: {args.ckpt}...")
    model = load_model(args.ckpt, device=dev)
    tok = Tokenizer.from_file(args.tokenizer)

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        f"# Heimdall Model Çıktı Değerlendirmesi ({len(PROMPTS)} Sabit Örnek)",
        "",
        f"- **Model:** `{args.ckpt}`",
        f"- **Parametre Sayısı:** ~{model.num_params() / 1e6:.1f}M",
        f"- **Mimari Düzen:** `{model.cfg.layout}` (bits={model.cfg.bits})",
        f"- **Tarih:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Üretim Ayarları:** `temperature={args.temp}`, `max_new_tokens={args.max_new}`",
        "",
        "---",
        ""
    ]

    print(f"\n10 Sabit İstem Çalıştırılıyor (temp={args.temp}):\n")

    for i, (cat, title, prompt) in enumerate(PROMPTS, 1):
        print(f"[{i:2d}/10] {cat}: {title}...")
        resp, speed = generate_response(model, tok, prompt, args.max_new, args.temp, dev, amp)

        md_lines.append(f"## {cat}: {title}")
        md_lines.append("")
        md_lines.append("**Girdi (Prompt):**")
        md_lines.append("```text")
        md_lines.append(prompt)
        md_lines.append("```")
        md_lines.append("")
        md_lines.append(f"**Model Çıktısı** *({speed:.1f} token/s):*")
        md_lines.append("```text")
        md_lines.append(resp if resp else "[Boş veya anında EOS]")
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nBaşarıyla yazıldı: {out_path}")


if __name__ == "__main__":
    main()
