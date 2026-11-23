#!/usr/bin/env bash
# Döngü deneyi (GLM-5.4 tarzı) + gece modelinin SFT'si, tek komut (Ubuntu / ROCm):
#
#   bash bench/dongu_deney.sh                       # ~2.5-3 saat
#   TOKENS=1e8 bash bench/dongu_deney.sh            # daha kısa (~1.5 saat)
#   SFT=0 bash bench/dongu_deney.sh                 # SFT'yi atla
#
# Soru: hazır bir modelin orta katmanlarını iki kez çalıştırmak (parametre eklemeden derinlik), aynı sürekli eğitimde
# kaybı düşürür mü? İki kol da runs/gece2/last.pt'den başlar, aynı veri, aynı token sayısı, aynı öğrenme oranı:
#   kontrol  döngüsüz sürekli eğitim (fazladan eğitimin kendi etkisi)
#   dongu    4..7 katmanları (ortadaki DDDA bloğu) iki kez; 2. geçiş token başına kapıyla yazar, kapı kapalı başlar
# Fark = döngünün etkisi. Döngü token başına ~%33 fazla hesap ister: tablo süreyi de gösterir.
# Ardından gece modeli (runs/gece2/last.pt) bench/sft_gece.sh ile ~25 dk SFT edilir.
# Yarıda kalırsa aynı komutu yeniden çalıştırın: biten adımlar atlanır, yarım kalan kol son kayıttan sürer.
#
# Ayarlar: TOKENS=2e8 LR_SCALE=0.3 DONGU=4:8 LOOP_MICRO= (OOM olursa LOOP_MICRO=4) SFT=1 MINUTES=25
#          VERI="veri/fineweb.bin:0.5 veri/cosmo.bin:0.5" (gece koşusuyla aynı karışım olmalı)
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=${PYTHON:-}
if [[ -z "$python_bin" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then python_bin="$VIRTUAL_ENV/bin/python"
    elif [[ -x .venv101/bin/python ]]; then python_bin=.venv101/bin/python
    elif [[ -x .venv/bin/python ]]; then python_bin=.venv/bin/python
    else python_bin=python3
    fi
fi
export PATH="$(cd "$(dirname "$python_bin")" && pwd):$PATH"    # sft_gece.sh "python3"ü aynı ortamdan bulsun
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export PYTHONUNBUFFERED=1

BASE=${BASE:-runs/gece2/last.pt}
TOKENS=${TOKENS:-2e8}
LR_SCALE=${LR_SCALE:-0.3}
DONGU=${DONGU:-4:8}
LOOP_MICRO=${LOOP_MICRO:-}
SFT=${SFT:-1}
read -r -a VERI <<< "${VERI:-veri/fineweb.bin:0.5 veri/cosmo.bin:0.5}"

die() { echo "dongu_deney: $*" >&2; exit 1; }
[[ -s "$BASE" ]] || die "$BASE yok (önce gece koşusu)"
for spec in "${VERI[@]}"; do [[ -s "${spec%%:*}" ]] || die "veri yok: ${spec%%:*}"; done
"$python_bin" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || die "GPU görünmüyor ($python_bin)"

common=(--preset gece --baslangic "$BASE" --tokens "$TOKENS" --lr-scale "$LR_SCALE" --warmup 0.05 --decay-start 0.5
        --data "${VERI[@]}" --log-every 20 --eval-every 200 --eval-batches 16 --save-every 0 --save-minutes 10)

run_arm() {   # ad, çıktı, ek argümanlar...
    local name=$1 out=$2; shift 2
    if [[ -d "$out/export" ]]; then echo "== $name: bitmiş, atlanıyor ($out)"; return; fi
    echo "== $name: $(date '+%H:%M') başlıyor ($out)"
    local t0=$(date +%s)
    if [[ -s "$out/last.pt" ]]; then
        "$python_bin" -u -m heimdall.train --resume "$out/last.pt" "$@"
    else
        "$python_bin" -u -m heimdall.train "${common[@]}" --out "$out" "$@"
    fi
    echo $(( $(date +%s) - t0 + $(cat "$out/sure_s" 2>/dev/null || echo 0) )) > "$out/sure_s.tmp" && mv "$out/sure_s.tmp" "$out/sure_s"
    "$python_bin" -m heimdall.export --ckpt "$out/last.pt" --out "$out/export"
}

run_arm kontrol runs/dongu_kontrol
loop_args=(--dongu "$DONGU")
[[ -n "$LOOP_MICRO" ]] && loop_args+=(--micro-batch "$LOOP_MICRO")
run_arm dongu runs/dongu_dongu "${loop_args[@]}"

if [[ "$SFT" == 1 ]]; then
    if [[ -d runs/gece_sft/export ]]; then
        echo "== SFT: bitmiş, atlanıyor (runs/gece_sft)"
    else
        echo "== SFT: $(date '+%H:%M') başlıyor (gece modeli)"
        CKPT="$BASE" PYTHON="$python_bin" bash bench/sft_gece.sh || echo "dongu_deney: SFT başarısız (deney sonuçları etkilenmez)" >&2
    fi
fi

"$python_bin" - "$BASE" <<'PY'
import json, statistics, sys
from pathlib import Path

def last_val(log):
    v = None
    for line in Path(log).read_text().splitlines():
        r = json.loads(line)
        if "val" in r:
            v = r
    return v

def speed(log):
    s = [r["tok_s"] for r in map(json.loads, Path(log).read_text().splitlines()) if "tok_s" in r]
    return statistics.median(s[len(s) // 4:]) if s else float("nan")   # ilk çeyrek: derleme/ısınma

base_log = Path(sys.argv[1]).parent / "log.jsonl"
rows = [("gece2 (başlangıç)", base_log, None)] + [(n, Path(f"runs/dongu_{n}/log.jsonl"), Path(f"runs/dongu_{n}/sure_s"))
                                                 for n in ("kontrol", "dongu")]
vals = {}
print(f"\n{'kol':20} {'fineweb':>9} {'cosmo':>9} {'tok/s':>9} {'süre':>8}")
for name, log, sure in rows:
    if not log.exists() or not (v := last_val(log)):
        print(f"{name:20} (sonuç yok)")
        continue
    val = v["val"]
    vals[name] = val
    fw = next((x for k, x in val.items() if "fineweb" in k), float("nan"))
    co = next((x for k, x in val.items() if "cosmo" in k), float("nan"))
    t = f"{int(sure.read_text()) / 60:.0f} dk" if sure and sure.exists() else "-"
    print(f"{name:20} {fw:9.4f} {co:9.4f} {speed(log):9,.0f} {t:>8}")
if "kontrol" in vals and "dongu" in vals:
    for k in vals["kontrol"]:
        d = vals["dongu"][k] - vals["kontrol"][k]
        print(f"döngü − kontrol, {k}: {d:+.4f} nat ({'döngü iyi' if d < 0 else 'döngü kötü ya da eşit'})")
    print("Not: döngü token başına ~%33 fazla hesap harcar. Fark ~0.01 nat'tan küçükse gürültü sayın (tek tohum).")
PY
