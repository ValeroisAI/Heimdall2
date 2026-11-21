#!/usr/bin/env bash
# Ubuntu ön kontrolü: RX 9070 XT (gfx1201), PyTorch 2.14+rocm10.1, torch.compile.
# Gece koşusundan ÖNCE:  bash bench/onkontrol_rocm.sh
# Ayar: NEED_GB=30  NEED_RAM_GB=16  SKIP_GPU=1  PYTHON=python3
# Her satır PASS, WARN ya da FAIL. FAIL satırının altındaki komut, o kontrolün düzeltmesidir.
# Çıkış: 0 = FAIL yok, 1 = en az bir FAIL.
set -u
cd "$(dirname "$0")/.."

NEED_GB=${NEED_GB:-30}
NEED_RAM_GB=${NEED_RAM_GB:-16}
SKIP_GPU=${SKIP_GPU:-0}
PYBIN=${PYTHON:-}
if [[ -z "$PYBIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then PYBIN="$VIRTUAL_ENV/bin/python"   # etkin ortam kazanır
  elif [[ -x .venv101/bin/python ]]; then PYBIN=.venv101/bin/python
  elif [[ -x .venv/bin/python ]]; then PYBIN=.venv/bin/python
  elif command -v python3 >/dev/null 2>&1; then PYBIN=python3
  else PYBIN=python
  fi
fi

ok_n=0
bad_n=0
warn_n=0
pass() { printf 'PASS  %s\n' "$1"; ok_n=$((ok_n + 1)); }
fail() { printf 'FAIL  %s\n      düzelt: %s\n' "$1" "$2"; bad_n=$((bad_n + 1)); }
warn() { printf 'WARN  %s\n      not: %s\n' "$1" "$2"; warn_n=$((warn_n + 1)); }

FIX_GROUPS='sudo usermod -aG render,video "$USER"; echo "oturumu kapatıp tekrar gir (newgrp yetmez)"'
FIX_ROCM='sudo apt-get update && sudo apt-get install -y rocminfo rocm-smi-lib; export PATH=/opt/rocm/bin:$PATH'
FIX_TORCH='pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.14.0+rocm10.1.0"'
FIX_LIMITS='ulimit -n 1048576; ulimit -l unlimited; sudo tee /etc/security/limits.d/99-rocm.conf >/dev/null <<EOF
* soft memlock unlimited
* hard memlock unlimited
* soft nofile 1048576
* hard nofile 1048576
EOF
echo "limits için oturumu kapatıp tekrar gir"'
LAUNCH='systemd-inhibit --what=idle:sleep --why="heimdall gece2" --mode=block bash bench/gece2.sh --hours 11 veri/fineweb.bin:0.7 veri/cosmo.bin:0.3'

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"   # PYTORCH_CUDA_ALLOC_CONF artık uyarı veriyor
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export PYTHONUNBUFFERED=1

echo "=== Heimdall ROCm ön kontrol $(date '+%F %T') ==="
echo "kök: $(pwd) | python: $PYBIN"

# --- işletim sistemi
if [[ "$(uname -s 2>/dev/null || echo '?')" == Linux ]]; then
  pass "Linux ($(uname -sr))"
else
  fail "bu betik Ubuntu içindir (şimdi: $(uname -s 2>/dev/null || echo bilinmiyor))" \
    "eğitimi RX 9070 XT takılı Ubuntu makinede çalıştır; Windows'tan yalnızca dosyaları kopyala"
fi

# --- rocminfo / rocm-smi
ROCMINFO=$(command -v rocminfo 2>/dev/null || true)
[[ -z "$ROCMINFO" && -x /opt/rocm/bin/rocminfo ]] && ROCMINFO=/opt/rocm/bin/rocminfo
ROCMSMI=$(command -v rocm-smi 2>/dev/null || true)
[[ -z "$ROCMSMI" && -x /opt/rocm/bin/rocm-smi ]] && ROCMSMI=/opt/rocm/bin/rocm-smi
if [[ -n "$ROCMINFO" ]]; then
  pass "rocminfo ($ROCMINFO)"
else
  fail "rocminfo yok" "$FIX_ROCM"
fi
if [[ -n "$ROCMSMI" ]]; then
  pass "rocm-smi ($ROCMSMI)"
else
  fail "rocm-smi yok" "$FIX_ROCM"
fi
if [[ "$ROCMINFO" == /opt/rocm/* || "$ROCMSMI" == /opt/rocm/* ]]; then
  case ":$PATH:" in
    *:/opt/rocm/bin:*) ;;
    *) warn "ROCm PATH'te değil" 'echo "export PATH=/opt/rocm/bin:\$PATH" >> ~/.bashrc && export PATH=/opt/rocm/bin:$PATH' ;;
  esac
fi

# --- gfx1201
gfx_hit=""
if [[ -n "$ROCMINFO" ]]; then
  gfx_hit=$("$ROCMINFO" 2>/dev/null | grep -o 'gfx1201' | head -1 || true)
fi
if [[ "$gfx_hit" == gfx1201 ]]; then
  pass "gfx1201 rocminfo'da görünüyor"
elif [[ -z "$ROCMINFO" ]]; then
  fail "gfx1201 kontrol edilemedi (rocminfo yok)" "$FIX_ROCM; sonra: rocminfo | grep -i gfx"
else
  fail "gfx1201 görünmüyor" "$FIX_GROUPS; $ROCMINFO | grep -i gfx; hâlâ yoksa (son çare, tekerlek gfx1201 ise gerekmez): export HSA_OVERRIDE_GFX_VERSION=12.0.1"
fi

# Yanlış ISA override'ı "no kernel image" yapar. gfx1201 yerliyse override kapalı kalsın.
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  if [[ "${HSA_OVERRIDE_GFX_VERSION}" == "12.0.1" && "$gfx_hit" == gfx1201 ]]; then
    warn "HSA_OVERRIDE_GFX_VERSION=12.0.1 gereksiz (gfx1201 zaten görünüyor)" "unset HSA_OVERRIDE_GFX_VERSION"
  elif [[ "${HSA_OVERRIDE_GFX_VERSION}" != "12.0.1" ]]; then
    fail "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} (RX 9070 XT için 12.0.1 dışında bir değer yanlış ISA seçer)" \
      "unset HSA_OVERRIDE_GFX_VERSION"
  else
    warn "HSA_OVERRIDE_GFX_VERSION=12.0.1 ayarlı, rocminfo gfx1201 göstermiyor" "$FIX_GROUPS"
  fi
else
  pass "HSA_OVERRIDE_GFX_VERSION ayarlı değil (gfx1201 tekerleği için doğru)"
fi

# --- gruplar ve aygıt düğümleri
groups_now=$(id -nG 2>/dev/null || true)
has_group() { echo " $groups_now " | grep -q " $1 "; }
if has_group render && has_group video; then
  pass "kullanıcı render ve video gruplarında ($(id -un))"
else
  miss=""
  has_group render || miss="render"
  has_group video || miss="${miss:+$miss,}video"
  fail "grup eksik: $miss" "$FIX_GROUPS"
fi
kfd_ok=0
dri_ok=0
[[ -r /dev/kfd && -w /dev/kfd ]] && kfd_ok=1
for n in /dev/dri/renderD*; do
  [[ -e "$n" && -r "$n" && -w "$n" ]] && dri_ok=1
done
if [[ $kfd_ok -eq 1 && $dri_ok -eq 1 ]]; then
  pass "/dev/kfd ve /dev/dri/renderD* okunup yazılıyor"
else
  fail "/dev/kfd ya da render düğümü erişilmiyor" "$FIX_GROUPS; ls -l /dev/kfd /dev/dri/renderD*"
fi

# --- disk
avail_kb=$(df -Pk . 2>/dev/null | awk 'NR==2 {print $4}')
if [[ -z "${avail_kb:-}" ]]; then
  fail "disk ölçülemedi (df .)" "df -h ."
else
  avail_gb=$((avail_kb / 1024 / 1024))
  if [[ $avail_gb -ge $NEED_GB ]]; then
    pass "boş disk ${avail_gb} GB (eşik ${NEED_GB} GB, $(df -Pk . | awk 'NR==2 {print $6}'))"
  else
    fail "boş disk ${avail_gb} GB < ${NEED_GB} GB" \
      "df -h .; rm -rf runs/boyut_* ; pip cache ve yarım parquet'i sil. Veri+checkpoint+torch.compile önbelleği için en az ${NEED_GB} GB bırak"
  fi
fi

# --- RAM
if [[ -r /proc/meminfo ]]; then
  ram_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if [[ -z "${ram_kb:-}" ]]; then
    warn "MemAvailable okunamadı" "free -h"
    ram_gb=""
  else
    ram_gb=$((ram_kb / 1024 / 1024))
  fi
  if [[ -z "${ram_gb:-}" ]]; then
    :
  elif [[ $ram_gb -lt 8 ]]; then
    fail "boş RAM ${ram_gb} GB (en az 8 GB; hedef ${NEED_RAM_GB} GB)" "free -h; tarayıcı ve diğer eğitimi kapat"
  elif [[ $ram_gb -lt $NEED_RAM_GB ]]; then
    warn "boş RAM ${ram_gb} GB < hedef ${NEED_RAM_GB} GB" "free -h; gece boyunca başka iş açma"
    pass "boş RAM ${ram_gb} GB (alt sınır 8 GB geçti)"
  else
    pass "boş RAM ${ram_gb} GB"
  fi
else
  warn "RAM okunamadı (/proc/meminfo yok)" "free -h"
fi

# --- ulimit
nofile=$(ulimit -n 2>/dev/null || echo 0)
memlock=$(ulimit -l 2>/dev/null || echo 0)
if [[ "$nofile" == unlimited || "$nofile" -ge 4096 ]]; then
  pass "ulimit -n = $nofile"
else
  fail "ulimit -n = $nofile (en az 4096)" "$FIX_LIMITS"
fi
if [[ "$memlock" == unlimited ]]; then
  pass "ulimit -l = unlimited"
elif [[ "$memlock" -ge 65536 ]]; then
  warn "ulimit -l = ${memlock} KB (ROCm için unlimited daha güvenli)" "$FIX_LIMITS"
else
  fail "ulimit -l = ${memlock} KB (ROCm büyük tahsis için unlimited ister)" "$FIX_LIMITS"
fi

# --- uyku engeli
if command -v systemd-inhibit >/dev/null 2>&1; then
  if systemd-inhibit --what=idle:sleep --why="heimdall onkontrol" --mode=block true; then
    pass "systemd-inhibit idle:sleep çalışıyor"
  else
    fail "systemd-inhibit denemesi başarısız" "sudo apt-get install -y systemd; $LAUNCH"
  fi
else
  fail "systemd-inhibit yok" "sudo apt-get install -y systemd; $LAUNCH"
fi

# --- gece2'nin ihtiyaç duyduğu araçlar
if command -v timeout >/dev/null 2>&1 && command -v flock >/dev/null 2>&1; then
  pass "timeout ve flock var (gece2 bunlara bağlı)"
else
  fail "timeout veya flock yok" "sudo apt-get install -y coreutils util-linux"
fi
if command -v tmux >/dev/null 2>&1; then
  pass "tmux var"
elif command -v nohup >/dev/null 2>&1; then
  warn "tmux yok, nohup var" "sudo apt-get install -y tmux   # SSH kopsa da oturum kalır. nohup yedektir"
else
  fail "ne tmux ne nohup var" "sudo apt-get install -y tmux coreutils"
fi

# --- tokenizer ve veri uyarısı
tok=../04_TOKENIZERLAR/valerois_tokenizer_8k.json
if [[ -s $tok ]]; then
  pass "8K tokenizer ($tok)"
else
  fail "8K tokenizer yok: $tok" "repo kökü Heimdall/Heimdall olmalı; dosya ../04_TOKENIZERLAR/valerois_tokenizer_8k.json"
fi
if [[ -s veri/fineweb.bin && -s veri/cosmo.bin ]]; then
  pass "veri/fineweb.bin ve veri/cosmo.bin var"
else
  warn "veri/fineweb.bin veya veri/cosmo.bin yok (gece2 örneği bu iki dosyayı ister)" \
    'python -m heimdall.hazirla --repo HuggingFaceFW/fineweb-edu --dosya sample/10BT/000_00000.parquet --kolon text --out veri/fineweb.bin --max-token 500e6; python -m heimdall.hazirla --repo HuggingFaceTB/cosmopedia --dosya data/web_samples_v2/train-00000-of-00118.parquet --kolon text --out veri/cosmo.bin --max-token 200e6'
fi

# --- ısı / güç / dolu VRAM (torch'tan önce, kart meşgulse duman testi koşuyu düşürmesin)
smi_txt=$(mktemp)
if [[ -n "$ROCMSMI" ]]; then
  "$ROCMSMI" --showtemp --showpower --showmeminfo vram >"$smi_txt" 2>&1 || true
  if grep -qi 'junction' "$smi_txt" || grep -qi 'Temperature' "$smi_txt"; then
    pass "rocm-smi sıcaklık/güç/VRAM cevap verdi"
  else
    fail "rocm-smi sıcaklık okuyamadı" "$FIX_GROUPS; $ROCMSMI --showtemp"
  fi
  # Kabuk aritmetiği ondalık bilmez; eşikleri python'a bırakmadan awk ile tam sayıya indir.
  # "GPU[0] : Temperature (Sensor junction) (C): 44.0" — sayı son alanda.
  junc=$(awk -F: '/[Jj]unction/ && /C/ {gsub(/[^0-9.]/,"",$NF); if ($NF != "") print int($NF); exit}' "$smi_txt")
  edge=$(awk -F: '/[Ee]dge/ && /C/ {gsub(/[^0-9.]/,"",$NF); if ($NF != "") print int($NF); exit}' "$smi_txt")
  hot=${junc:-${edge:-}}
  if [[ -n "$hot" ]]; then
    if [[ "$hot" -ge 95 ]]; then
      fail "GPU ${hot}°C (geceye bu ısıyla girme)" "$ROCMSMI --showtemp; fan ve hava akışını düzelt, 80°C altına inmeden başlama"
    elif [[ "$hot" -ge 80 ]]; then
      warn "GPU ${hot}°C (boşta yüksek)" "$ROCMSMI --showtemp; kasa açık kalsın, yükte 95°C üstünü izle"
    else
      pass "GPU sıcaklığı ${hot}°C"
    fi
  else
    warn "sıcaklık satırı ayrışmadı" "$ROCMSMI --showtemp"
  fi
  used_b=$(awk -F: '/VRAM Total Used Memory/ {gsub(/[^0-9]/,"",$NF); print $NF; exit}' "$smi_txt")
  total_b=$(awk -F: '/VRAM Total Memory \(B\)/ && !/Used/ {gsub(/[^0-9]/,"",$NF); print $NF; exit}' "$smi_txt")
  if [[ -n "${used_b:-}" && -n "${total_b:-}" && "$total_b" -gt 0 ]]; then
    used_gb=$((used_b / 1024 / 1024 / 1024))
    total_gb=$((total_b / 1024 / 1024 / 1024))
    if [[ $used_gb -ge 2 ]]; then
      fail "VRAM dolu: ${used_gb}/${total_gb} GB (başka süreç kartı tutuyor)" \
        "$ROCMSMI --showpids; eğitim sende değilse onu kapat. Heimdall hafta koşusuysa: bash bench/dur.sh"
    else
      pass "VRAM boş görünüyor (${used_gb}/${total_gb} GB kullanılıyor)"
    fi
  fi
  pwr=$(awk -F: '/Average Graphics Package Power/ {gsub(/[^0-9.]/,"",$NF); if ($NF != "") print int($NF); exit}' "$smi_txt")
  if [[ -n "${pwr:-}" ]]; then
    if [[ "$pwr" -eq 0 && -z "$hot" ]]; then
      fail "güç ve sıcaklık 0 (sürücü kartla konuşmuyor)" "$FIX_ROCM; $ROCMSMI"
    else
      pass "anlık paket gücü ~${pwr} W"
    fi
  fi
fi
rm -f "$smi_txt"

if command -v pgrep >/dev/null 2>&1 && pgrep -f 'heimdall[.]train' >/dev/null 2>&1; then
  warn "heimdall.train zaten çalışıyor" "GPU duman testi atlandı; koşuyu düşürmesin. python bench/izle.py gece2.log --out runs/gece2"
  SKIP_GPU=1
fi

# Ortak Python çalıştırıcı. Betik stdin'den gelir. Başarı: çıkış 0 ve stdout'ta 'ok' ile başlayan satır.
run_py() {
  local title="$1" fix="$2" secs="$3" out rc err
  out=$(mktemp)
  # Kod bir DOSYADAN çalışmalı: @triton.jit stdin'den okunan betikte kaynağı bulamaz ("should be defined in a Python file").
  src=$(mktemp --suffix=.py)
  cat >"$src"
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$PYBIN" "$src" >"$out" 2>&1
  else
    "$PYBIN" "$src" >"$out" 2>&1
  fi
  rc=$?
  rm -f "$src"
  if [[ $rc -eq 0 ]] && grep -q '^ok' "$out"; then
    pass "$title — $(grep '^ok' "$out" | head -1)"
  else
    err=$(tail -n 1 "$out" | tr '\r' ' ' | cut -c1-200)
    if [[ $rc -eq 124 ]]; then
      fail "$title (zaman aşımı ${secs}s)" "$fix | son: $err"
    else
      fail "$title" "$fix | son: $err"
    fi
  fi
  rm -f "$out"
}

if [[ "$SKIP_GPU" == 1 ]]; then
  warn "GPU duman testleri atlandı (SKIP_GPU=1)" "kart boşalınca SKIP_GPU vermeden yeniden çalıştır"
else
  if ! command -v "$PYBIN" >/dev/null 2>&1 && [[ ! -x "$PYBIN" ]]; then
    fail "python yok ($PYBIN)" "python3 -m venv .venv && source .venv/bin/activate && $FIX_TORCH"
  else
    run_py "torch.cuda + bf16 matmul" "$FIX_TORCH; $FIX_GROUPS" 180 <<'PY'
import torch
hip = torch.version.hip or ""
ver = torch.__version__
if not torch.cuda.is_available():
    raise SystemExit("cuda kullanılamıyor: " + ver + " hip=" + hip)
if "2.14" not in ver or not hip:
    raise SystemExit("beklenen tekerlek 2.14+rocm, gelen: " + ver + " hip=" + hip)
name = torch.cuda.get_device_name(0)
arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
blob = (name + " " + arch).lower()
if "9070" not in blob and "gfx1201" not in blob:
    raise SystemExit("yanlış GPU: " + name + " " + arch)
if not torch.cuda.is_bf16_supported():
    raise SystemExit("bf16 yok")
a = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16)
b = torch.randn(768, 768, device="cuda", dtype=torch.bfloat16)
y = a @ b
torch.cuda.synchronize()
if not torch.isfinite(y.float()).all():
    raise SystemExit("bf16 matmul sonlu değil")
print("ok", ver, "hip", hip, name, arch)
PY
    run_py "torch.compile bf16" "$FIX_TORCH; takılırsa gece2'ye --no-compile ekleme, önce: python bench/rocm_teshis.py" 240 <<'PY'
import torch
def f(x):
    return x @ x.transpose(0, 1)
g = torch.compile(f)
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
y = g(x)
torch.cuda.synchronize()
if not torch.isfinite(y.float()).all():
    raise SystemExit("compile çıktısı sonlu değil")
print("ok compile", torch.__version__)
PY
    run_py "Triton çekirdek" "$FIX_TORCH" 240 <<'PY'
import torch
import triton
import triton.language as tl

@triton.jit
def k_add(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1, mask=m)

n = 1024
x = torch.ones(n, device="cuda")
y = torch.empty_like(x)
k_add[(1,)](x, y, n, BLOCK=1024)
torch.cuda.synchronize()
if not torch.allclose(y, x + 1):
    raise SystemExit("triton sonucu yanlış")
print("ok triton", triton.__version__)
PY
    # flash-linear-attention isteğe bağlı: yoksa eğitim PyTorch'a düşer (WARN). Varsa öz-test geçmeli.
    fla_out=$(mktemp)
    if command -v timeout >/dev/null 2>&1; then
      timeout 240 "$PYBIN" - >"$fla_out" 2>&1 <<'PY'
import torch
from heimdall.kernels import prepare
msg = prepare(torch.device("cuda"), "auto")
print(msg)
PY
    else
      "$PYBIN" - >"$fla_out" 2>&1 <<'PY'
import torch
from heimdall.kernels import prepare
msg = prepare(torch.device("cuda"), "auto")
print(msg)
PY
    fi
    fla_rc=$?
    fla_msg=$(tail -n 1 "$fla_out" | tr '\r' ' ' | cut -c1-200)
    if [[ $fla_rc -eq 0 ]] && grep -q 'flash-linear-attention (' "$fla_out"; then
      pass "flash-linear-attention öz-test — $fla_msg"
    elif [[ $fla_rc -eq 0 ]] && grep -q 'kurulu değil' "$fla_out"; then
      warn "flash-linear-attention kurulu değil (eğitim PyTorch çekirdeğine düşer, daha yavaş)" \
        'pip install "flash-linear-attention>=0.2"'
    else
      fail "flash-linear-attention öz-test" "pip install \"flash-linear-attention>=0.2\"; olmadıysa TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1. son: $fla_msg"
    fi
    rm -f "$fla_out"
  fi
fi

echo
echo "--- başlatma (tmux içinde; inhibit gece2'nin içinde YOK, bu yüzden sarmal şart) ---"
echo "  tmux new -s gece"
echo "  source ${VIRTUAL_ENV:-.venv}/bin/activate"
echo "  $LAUNCH 2>&1 | tee -a gece2.log"
echo "  # başka pencere: python bench/izle.py gece2.log --out runs/gece2"
echo "  # SSH kopunca: tmux attach -t gece"
if ! command -v tmux >/dev/null 2>&1 && command -v nohup >/dev/null 2>&1; then
  echo "  # tmux yoksa: nohup $LAUNCH > gece2.log 2>&1 & echo \$! > gece2.pid"
fi
echo
printf 'SONUÇ: %d PASS, %d WARN, %d FAIL\n' "$ok_n" "$warn_n" "$bad_n"
if [[ $bad_n -gt 0 ]]; then
  echo "Geceyi başlatma. Her FAIL satırındaki düzelt: komutunu uygula ve bu betiği yeniden çalıştır."
  exit 1
fi
echo "FAIL yok. WARN satırlarını oku; sonra GECE_RUNBOOK.md adım 4."
exit 0
