#!/usr/bin/env bash
# degerlendir_gece.sh — Gece koşusu sonrası hızlı model değerlendirmesi.
#
# Adımlar:
#   1. Doğrulama kaybı (Val loss / NLL) [bench/baglam.py nll]
#   2. Parola arama (Passkey retrieval) [bench/baglam.py sifre]
#   3. lm-evaluation-harness (ARC-easy, PIQA vb. hızlı alt küme) [bench/lmeval_heimdall.py]
#   4. 10 Sabit Örnek İstem (Prompt) üretimi ve Markdown raporu [bench/ornek_uret.py]
#
# Kullanım:
#   bash bench/degerlendir_gece.sh --ckpt runs/gece_sft/last.pt
#   bash bench/degerlendir_gece.sh --ckpt runs/gece2/last.pt
#   CKPT=runs/gece_sft/last.pt bash bench/degerlendir_gece.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON:-python3}
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    PYTHON_BIN=python
fi

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# 1. Komut satırı argümanları ve ortam değişkenleri
CKPT=${CKPT:-""}
OUT_DIR=${OUT_DIR:-""}
TOK=${TOK:-""}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --ckpt=*)
            CKPT="${1#*=}"
            shift
            ;;
        --out|--out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --out=*|--out-dir=*)
            OUT_DIR="${1#*=}"
            shift
            ;;
        --tok)
            TOK="$2"
            shift 2
            ;;
        --tok=*)
            TOK="${1#*=}"
            shift
            ;;
        *)
            if [ -z "$CKPT" ]; then
                CKPT="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CKPT" ]; then
    if [ -f "runs/gece_sft/last.pt" ]; then
        CKPT="runs/gece_sft/last.pt"
    elif [ -f "runs/gece2/last.pt" ]; then
        CKPT="runs/gece2/last.pt"
    elif [ -f "runs/gece/last.pt" ]; then
        CKPT="runs/gece/last.pt"
    elif [ -d "runs/gece_sft/export" ]; then
        CKPT="runs/gece_sft/export"
    elif [ -d "runs/gece2/export" ]; then
        CKPT="runs/gece2/export"
    else
        echo "HATA: Değerlendirilecek model checkpoint'i bulunamadı!" >&2
        echo "Kullanım: bash bench/degerlendir_gece.sh --ckpt runs/gece2/last.pt" >&2
        exit 1
    fi
fi

OUT_DIR=${OUT_DIR:-"runs/degerlendirme"}
mkdir -p "$OUT_DIR"

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

echo "================================================================================"
echo "HEIMDALL GECE KOŞUSU MODEL DEĞERLENDİRMESİ"
echo "  Tarih:        $(date)"
echo "  Checkpoint:   $CKPT"
echo "  Tokenizer:    $TOKENIZER"
echo "  Rapor Dizini: $OUT_DIR"
echo "================================================================================"

# 3. Adım: Doğrulama Kaybı (Val Loss / NLL)
echo ""
echo ">>> [1/4] Doğrulama Kaybı (NLL & PPL) Ölçülüyor..."
VAL_VERI=""
if [ -f "veri/fineweb.bin" ]; then
    VAL_VERI="veri/fineweb.bin"
elif [ -f "veri/cosmopedia.bin" ]; then
    VAL_VERI="veri/cosmopedia.bin"
elif [ -f "../stream_coder_100k.bin" ]; then
    VAL_VERI="../stream_coder_100k.bin"
elif [ -f "stream_coder_100k.bin" ]; then
    VAL_VERI="stream_coder_100k.bin"
fi

if [ -n "$VAL_VERI" ]; then
    "$PYTHON_BIN" bench/baglam.py nll --ckpt "$CKPT" --veri "$VAL_VERI" --uzunluk 2048 --n 8 | tee "$OUT_DIR/val_loss.log"
else
    echo "  [Atlandı] Doğrulama için .bin dosyası bulunamadı."
fi

# 4. Adım: Passkey Retrieval (Parola Arama & Bağlam Dayanıklılığı)
echo ""
echo ">>> [2/4] Parola Arama (Passkey Retrieval, 512..4096 token) Test Ediliyor..."
"$PYTHON_BIN" bench/baglam.py sifre --ckpt "$CKPT" --uzunluklar 512 1024 2048 4096 --deneme 3 | tee "$OUT_DIR/passkey.log"

# 5. Adım: lm-evaluation-harness (ARC-easy, PIQA)
echo ""
echo ">>> [3/4] Standart Görevler (lm-eval: arc_easy, piqa)..."
if "$PYTHON_BIN" -c "import lm_eval" 2>/dev/null; then
    "$PYTHON_BIN" bench/lmeval_heimdall.py \
        --ckpt "$CKPT" \
        --tasks arc_easy,piqa \
        --limit 100 \
        --out "$OUT_DIR/lmeval.json" | tee "$OUT_DIR/lmeval.log"
else
    echo "  [Bilgi] lm-eval kütüphanesi kurulu değil (pip install lm-eval). Bu adım atlanıyor."
fi

# 6. Adım: 10 Sabit Örnek İstem Üretimi (Markdown Raporu)
echo ""
echo ">>> [4/4] 10 Sabit Örnek İstem için Modelden Metin Üretiliyor..."
MD_OUT="$OUT_DIR/ornek_uretilenler.md"
"$PYTHON_BIN" bench/ornek_uret.py \
    --ckpt "$CKPT" \
    --tokenizer "$TOKENIZER" \
    --out-md "$MD_OUT" \
    --max-new 200 \
    --temp 0.7

echo ""
echo "================================================================================"
echo "Değerlendirme Tamamlandı!"
echo "  - Doğrulama Kaybı Günlüğü: $OUT_DIR/val_loss.log"
echo "  - Passkey Raporu:          $OUT_DIR/passkey.log"
echo "  - 10 Örnek Markdown Dosyası: $MD_OUT"
echo "================================================================================"
