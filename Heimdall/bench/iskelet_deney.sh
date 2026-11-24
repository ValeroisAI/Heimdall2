#!/usr/bin/env bash
# İskelet deneyi (Ubuntu / ROCm): hazır bir modelin FFN/gömme/normları donuk, dikkatin yerine YENİ karıştırıcı konunca
# bilgi ne kadar korunuyor?  Beş kol, aynı veri ve aynı eğitim bütçesi, ölçüt WikiText perplexity (düşük iyi).
#
#   bash bench/iskelet_deney.sh                          # SmolLM2-135M, ~1-1.5 saat
#   MODEL=Qwen/Qwen3-0.6B bash bench/iskelet_deney.sh    # başka model (Llama/Qwen2/Qwen3/SmolLM ailesi)
#   KOLLAR="bifrost delta" STEPS_A=300 STEPS_B=200 bash bench/iskelet_deney.sh   # kısa deneme
#
# Kollar:
#   bifrost   mevcut dönüşüm: q/k/v/o YENİDEN KULLANILIR, pencere 64 dikkat + delta belleği (referans)
#   delta     her dikkat katmanı yerine SIFIRDAN GatedDeltaNet (q/k/v/o atılır)
#   conv      her dikkat katmanı yerine SIFIRDAN kapılı kısa evrişim
#   delta_h   delta, ama her 4. katman öğretmen dikkati kalır (DDDA düzeni)
#   conv_h    conv,  ama her 4. katman öğretmen dikkati kalır
# Aşama A: katman katman, öğretmenin dikkat çıktısını taklit. Aşama B: uçtan uca KL damıtma. İkisinde de yalnız yeni
# karıştırıcılar eğitilir (donuk bilgi: FFN, gömme, normlar).
# Yarıda kalırsa aynı komutu yeniden çalıştırın: biten kollar atlanır.
#
# Ayarlar: MODEL STEPS_A=600 STEPS_B=400 SEQ=1024 BATCH=4 LR=3e-3 EVAL_LEN=2048 EVAL_N=16 KOLLAR="bifrost delta conv delta_h conv_h"
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
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

MODEL=${MODEL:-HuggingFaceTB/SmolLM2-135M}
STEPS_A=${STEPS_A:-600}
STEPS_B=${STEPS_B:-400}
SEQ=${SEQ:-1024}
BATCH=${BATCH:-4}
LR=${LR:-3e-3}
EVAL_LEN=${EVAL_LEN:-2048}
EVAL_N=${EVAL_N:-16}
KOLLAR=${KOLLAR:-"bifrost delta conv delta_h conv_h"}
TAG=$(echo "$MODEL" | tr '/' '_')

"$python_bin" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { echo "iskelet_deney: GPU görünmüyor ($python_bin)" >&2; exit 1; }

# her 4. katman (3, 7, 11, ...) öğretmen dikkati kalsın
n_layers=$("$python_bin" - "$MODEL" <<'PY'
import json, sys
from heimdall.donustur import fetch_model
from pathlib import Path
print(json.loads((Path(fetch_model(sys.argv[1])) / "config.json").read_text())["num_hidden_layers"])
PY
)
keep=$(seq 3 4 $((n_layers - 1)) | tr '\n' ' ')
echo "iskelet_deney: $MODEL, $n_layers katman | A $STEPS_A + B $STEPS_B adım, $BATCH×$SEQ | korunan (_h kolları): $keep"

common=(--model "$MODEL" --wikitext veri/wikitext --wikitext-tokens 20e6 --seq-len "$SEQ" --batch "$BATCH"
        --steps-a "$STEPS_A" --steps-b "$STEPS_B" --lr "$LR" --eval-len "$EVAL_LEN" --eval-n "$EVAL_N")

for arm in $KOLLAR; do
    out="runs/iskelet_${TAG}_${arm}"
    if [[ -f "$out/sonuc.json" ]] && grep -q donusum_asama_B "$out/sonuc.json"; then
        echo "== $arm: bitmiş, atlanıyor ($out)"
        continue
    fi
    case "$arm" in
        bifrost) extra=() ;;
        delta)   extra=(--iskelet delta) ;;
        conv)    extra=(--iskelet conv) ;;
        delta_h) extra=(--iskelet delta --keep-global $keep) ;;
        conv_h)  extra=(--iskelet conv --keep-global $keep) ;;
        *) echo "bilinmeyen kol: $arm" >&2; exit 1 ;;
    esac
    echo "== $arm: $(date '+%H:%M') başlıyor ($out)"
    "$python_bin" -u -m heimdall.donustur "${common[@]}" "${extra[@]}" --out "$out" 2>&1 | tee -a "$out.log"
done

"$python_bin" - "$TAG" $KOLLAR <<'PY'
import json, sys
from pathlib import Path
tag, arms = sys.argv[1], sys.argv[2:]
rows, teacher = [], None
for arm in arms:
    f = Path(f"runs/iskelet_{tag}_{arm}/sonuc.json")
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    teacher = teacher or r["ogretmen"]["ppl"]
    a, b = r.get("donusum_asama_A"), r.get("donusum_asama_B")
    buckets = [k for k in (b or {}) if "-" in k]
    rows.append((arm, r["donusum_egitimsiz"]["ppl"], a and a["ppl"], b and b["ppl"], b and b.get(buckets[-1]) if buckets else None))
if rows:
    print(f"\nöğretmen WikiText ppl: {teacher:.2f}  (düşük iyi; uzun bağlam = son kova)")
    print(f"{'kol':10} {'eğitimsiz':>10} {'A sonrası':>10} {'B sonrası':>10} {'son kova':>10} {'öğretmene göre':>15}")
    for arm, u, a, b, last in rows:
        rel = f"{100 * (b / teacher - 1):+.1f}%" if b else "-"
        print(f"{arm:10} {u:10.2f} {a or float('nan'):10.2f} {b or float('nan'):10.2f} {last or float('nan'):10.2f} {rel:>15}")
    print("\nbifrost: q/k/v/o yeniden kullanılır. delta/conv: SIFIRDAN yeni karıştırıcı. _h: her 4. katman öğretmen dikkati.")
    print("Karar: yeni karıştırıcı kolları bifrost'a yakınsa 'iskelet + donuk FFN' fikri çalışır; çok uzaktaysa bilgi karıştırıcıya sıkı bağlıdır.")
PY
