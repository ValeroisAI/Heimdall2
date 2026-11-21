#!/usr/bin/env bash
# veri_gece.sh — Gece koşusu için yüksek kaliteli veri kümesi karışımı (~2.4B token).
# FineWeb-Edu, Cosmopedia, GitHub Code, OpenWebMath ve Cosmopedia WikiHow (talimat)
# uint16 .bin dosyalarına dönüştürülür; disk kontrolü ve devam ettirme (resumable) destekler.
#
# Kullanım:
#   bash bench/veri_gece.sh
#   TOK=32k bash bench/veri_gece.sh        # 32K tokenizer ile
#   D=/baska/yer bash bench/veri_gece.sh   # Farklı veri dizini
#
set -euo pipefail
cd "$(dirname "$0")/.."

D=${D:-veri}
mkdir -p "$D"

PYTHON_BIN=${PYTHON:-python3}
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    PYTHON_BIN=python
fi

# 1. Disk alanı kontrolü (En az 35 GB boş alan gerekli)
MIN_GB=35
if command -v df &>/dev/null; then
    AVAILABLE_KB=$(df -k "$D" | awk 'NR==2 {print $4}')
    AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
    if [ "$AVAILABLE_GB" -lt "$MIN_GB" ]; then
        echo "HATA: Yetersiz disk alanı! Gerekli: ${MIN_GB} GB, Mevcut: ${AVAILABLE_GB} GB ($D)" >&2
        exit 1
    fi
    echo "Disk kontrolü başarılı: ${AVAILABLE_GB} GB boş alan mevcut (gerekli: ${MIN_GB} GB)."
fi

# 2. Tokenizer seçimi (varsayılan: 8k)
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

if [ ! -f "$TOKENIZER" ]; then
    echo "HATA: Tokenizer dosyası bulunamadı: $TOKENIZER" >&2
    exit 1
fi
echo "Kullanılan Tokenizer: $TOKENIZER (mod: $TOK)"

# 3. Veri Kümeleri: Tamamı açık lisanslı, gerçekte var olan Hugging Face depoları
#
# (a) FineWeb-Edu (ODC-By 1.0): Yüksek kaliteli eğitim metinleri (~1.2B token, %50)
FW_PARQUETS="sample/10BT/000_00000.parquet sample/10BT/001_00000.parquet sample/10BT/002_00000.parquet"
if [ -f "$D/fineweb.bin" ]; then
    echo "[1/5] Mevcut: $D/fineweb.bin ($(du -h "$D/fineweb.bin" 2>/dev/null | cut -f1 || ls -lh "$D/fineweb.bin" | awk '{print $5}')), atlanıyor."
else
    echo "[1/5] İndiriliyor ve tokenleştiriliyor: FineWeb-Edu (~1200M token) -> $D/fineweb.bin"
    "$PYTHON_BIN" -m heimdall.hazirla \
        --repo HuggingFaceFW/fineweb-edu \
        --dosya $FW_PARQUETS \
        --kolon text \
        --out "$D/fineweb.bin" \
        --max-token 1200e6 \
        --tokenizer "$TOKENIZER"
fi

# (b) Cosmopedia web_samples_v2 (Apache-2.0): Sentetik ders kitabı & web bilgisi (~480M token, %20)
CS_PARQUETS="data/web_samples_v2/train-00000-of-00118.parquet data/web_samples_v2/train-00001-of-00118.parquet data/web_samples_v2/train-00002-of-00118.parquet data/web_samples_v2/train-00003-of-00118.parquet data/web_samples_v2/train-00004-of-00118.parquet data/web_samples_v2/train-00005-of-00118.parquet"
if [ -f "$D/cosmopedia.bin" ]; then
    echo "[2/5] Mevcut: $D/cosmopedia.bin ($(du -h "$D/cosmopedia.bin" 2>/dev/null | cut -f1 || ls -lh "$D/cosmopedia.bin" | awk '{print $5}')), atlanıyor."
else
    echo "[2/5] İndiriliyor ve tokenleştiriliyor: Cosmopedia (~480M token) -> $D/cosmopedia.bin"
    "$PYTHON_BIN" -m heimdall.hazirla \
        --repo HuggingFaceTB/cosmopedia \
        --dosya $CS_PARQUETS \
        --kolon text \
        --out "$D/cosmopedia.bin" \
        --max-token 480e6 \
        --tokenizer "$TOKENIZER"
fi

# (c) GitHub Code Clean (Açık kaynak): Python, JS, Rust, Go vb. temiz kod (~480M token, %20)
KD_PARQUETS="data/train-00000-of-00880.parquet data/train-00001-of-00880.parquet data/train-00002-of-00880.parquet data/train-00003-of-00880.parquet"
if [ -f "$D/kod.bin" ]; then
    echo "[3/5] Mevcut: $D/kod.bin ($(du -h "$D/kod.bin" 2>/dev/null | cut -f1 || ls -lh "$D/kod.bin" | awk '{print $5}')), atlanıyor."
else
    echo "[3/5] İndiriliyor ve tokenleştiriliyor: GitHub Code (~480M token) -> $D/kod.bin"
    "$PYTHON_BIN" -m heimdall.hazirla \
        --repo codeparrot/github-code-clean \
        --dosya $KD_PARQUETS \
        --kolon code \
        --out "$D/kod.bin" \
        --max-token 480e6 \
        --tokenizer "$TOKENIZER"
fi

# (d) OpenWebMath (ODC-By 1.0): Matematik, LaTeX & analitik akıl yürütme (~140M token, %6)
MTH_PARQUETS="data/train-00000-of-00114-5a023365406cb9c4.parquet data/train-00001-of-00114-e32fc2813a15f61c.parquet"
if [ -f "$D/matematik.bin" ]; then
    echo "[4/5] Mevcut: $D/matematik.bin ($(du -h "$D/matematik.bin" 2>/dev/null | cut -f1 || ls -lh "$D/matematik.bin" | awk '{print $5}')), atlanıyor."
else
    echo "[4/5] İndiriliyor ve tokenleştiriliyor: OpenWebMath (~140M token) -> $D/matematik.bin"
    "$PYTHON_BIN" -m heimdall.hazirla \
        --repo open-web-math/open-web-math \
        --dosya $MTH_PARQUETS \
        --kolon text \
        --out "$D/matematik.bin" \
        --max-token 140e6 \
        --tokenizer "$TOKENIZER"
fi

# (e) Cosmopedia WikiHow (Apache-2.0): Yapılandırılmış adım adım talimat ve rehber metinleri (~100M token, %4)
WH_PARQUETS="data/wikihow/train-00000-of-00002.parquet data/wikihow/train-00001-of-00002.parquet"
if [ -f "$D/talimat.bin" ]; then
    echo "[5/5] Mevcut: $D/talimat.bin ($(du -h "$D/talimat.bin" 2>/dev/null | cut -f1 || ls -lh "$D/talimat.bin" | awk '{print $5}')), atlanıyor."
else
    echo "[5/5] İndiriliyor ve tokenleştiriliyor: Cosmopedia WikiHow Talimat (~100M token) -> $D/talimat.bin"
    "$PYTHON_BIN" -m heimdall.hazirla \
        --repo HuggingFaceTB/cosmopedia \
        --dosya $WH_PARQUETS \
        --kolon text \
        --out "$D/talimat.bin" \
        --max-token 100e6 \
        --tokenizer "$TOKENIZER"
fi

# 4. Özet ve Eğitim Komutları
DATA_SPEC="$D/fineweb.bin:0.50 $D/cosmopedia.bin:0.20 $D/kod.bin:0.20 $D/matematik.bin:0.06 $D/talimat.bin:0.04"

echo ""
echo "================================================================================"
echo "Gece Eğitimi Veri Hazırlığı Tamamlandı (~2.4B Token Toplam Karışım):"
echo "  1. $D/fineweb.bin     ~1200M token (%50) [FineWeb-Edu ODC-By]"
echo "  2. $D/cosmopedia.bin   ~480M token (%20) [Cosmopedia Apache-2.0]"
echo "  3. $D/kod.bin          ~480M token (%20) [GitHub Code Clean Permissive]"
echo "  4. $D/matematik.bin    ~140M token (%6)  [OpenWebMath ODC-By]"
echo "  5. $D/talimat.bin      ~100M token (%4)  [Cosmopedia WikiHow Apache-2.0]"
echo "================================================================================"
echo ""
echo "Eğitim için --data parametresi:"
echo "  --data $DATA_SPEC"
echo ""
echo "gece2.sh çalıştırma komutu:"
echo "  bash bench/gece2.sh --hours 11 $DATA_SPEC"
echo "================================================================================"
