#!/usr/bin/env bash
# sft_gece.sh — Ön eğitim sonrası kısa SFT (Supervised Fine-Tuning) aşaması (~20-30 dk).
#
# Akış:
#   1. SFT veri kümesini denetler (yoksa bench/sft_veri.py ile hazırlar)
#   2. En son model kontrol noktasını (runs/gece2/last.pt veya runs/gece/last.pt) bulur
#   3. ChatML şablonu ve asistan tokenlerine maskelenmiş kayıp ile ~25 dk eğitir
#   4. runs/gece_sft/last.pt ve servis paketini (runs/gece_sft/export/) oluşturur
#
# Kullanım:
#   bash bench/sft_gece.sh
#   CKPT=runs/gece2/last.pt MINUTES=20 bash bench/sft_gece.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON:-python3}
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    PYTHON_BIN=python
fi

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"

# 1. Kontrol noktası (Checkpoint) belirleme
CKPT=${CKPT:-""}
if [ -z "$CKPT" ]; then
    if [ -f "runs/gece2/last.pt" ]; then
        CKPT="runs/gece2/last.pt"
    elif [ -f "runs/gece/last.pt" ]; then
        CKPT="runs/gece/last.pt"
    elif [ -d "runs/gece2/export" ]; then
        CKPT="runs/gece2/export"
    else
        echo "HATA: Başlangıç kontrol noktası bulunamadı! (runs/gece2/last.pt veya runs/gece/last.pt bekleniyor)." >&2
        echo "Lütfen CKPT=/yol/last.pt bash bench/sft_gece.sh şeklinde verin." >&2
        exit 1
    fi
fi
echo "=== SFT Başlangıç Modeli: $CKPT"

# 2. Tokenizer seçimi
TOK=${TOK:-8k}
if [ -f "../04_TOKENIZERLAR/valerois_tokenizer_8k.json" ]; then
    TOK_DIR="../04_TOKENIZERLAR"
elif [ -f "04_TOKENIZERLAR/valerois_tokenizer_8k.json" ]; then
    TOK_DIR="04_TOKENIZERLAR"
else
    TOK_DIR="../04_TOKENIZERLAR"
fi

if [ "$TOK" = "32k" ]; then
    TOKENIZER="$TOK_DIR/valerois_tokenizer_32k.json"
elif [ "$TOK" = "8k" ]; then
    TOKENIZER="$TOK_DIR/valerois_tokenizer_8k.json"
else
    TOKENIZER="$TOK"
fi
[[ -s "$TOKENIZER" ]] || { echo "Tokenizer bulunamadı: $TOKENIZER" >&2; exit 1; }

# 3. SFT Verisi Kontrolü (yoksa sft_veri.py ile üret)
SFT_DIR="veri/sft"
TRAIN_DATA="$SFT_DIR/train_metin.jsonl"
EVAL_DATA="$SFT_DIR/eval_metin.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "SFT verisi ($TRAIN_DATA) bulunamadı. bench/sft_veri.py ile hazırlanıyor..."
    mkdir -p "$SFT_DIR"
    "$PYTHON_BIN" bench/sft_veri.py --out "$SFT_DIR" --d3-max 12000 --max-token-ornek 4000
fi

# 4. SFT Eğitimi
MINUTES=${MINUTES:-25}
OUT=${SFT_OUT:-runs/gece_sft}
LR=${LR:-3e-5}
MICRO_BATCH=${MICRO_BATCH:-4}
ACCUM=${ACCUM:-4}
SEQ_LEN=${SEQ_LEN:-1024}

echo "=== SFT Aşaması Başlıyor: $(date) ==="
echo "Hedef süre: ${MINUTES} dakika | Çıktı: $OUT | LR: $LR | Dizi: $SEQ_LEN"

"$PYTHON_BIN" bench/sft_egit.py \
    --ckpt "$CKPT" \
    --train-data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --tokenizer "$TOKENIZER" \
    --out "$OUT" \
    --minutes "$MINUTES" \
    --lr "$LR" \
    --micro-batch "$MICRO_BATCH" \
    --accum "$ACCUM" \
    --seq-len "$SEQ_LEN" \
    --eval-every 50

echo ""
echo "================================================================================"
echo "SFT Eğitimi Tamamlandı: $(date)"
echo "  Kayıtlı Checkpoint: $OUT/last.pt"
echo "  Kullanıma Hazır Servis Paketi: $OUT/export"
echo ""
echo "Değerlendirme için:"
echo "  bash bench/degerlendir_gece.sh --ckpt $OUT/last.pt"
echo "================================================================================"
