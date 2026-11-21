#!/usr/bin/env bash
# Run under a sleep inhibitor on Ubuntu/ROCm:
# systemd-inhibit --what=idle:sleep bash bench/gece2.sh --hours 11 veri/fineweb.bin:0.7 veri/cosmo.bin:0.3
# Data paths are relative to Heimdall/ and must use valerois_tokenizer_8k.json.
# OUT=runs/gece2 SAVE_MINUTES=5 PYTHON=python3 are optional. Reuse OUT to resume.
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=${PYTHON:-}
if [[ -z "$python_bin" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then python_bin="$VIRTUAL_ENV/bin/python"      # etkin ortam kazanır
    elif [[ -x .venv101/bin/python ]]; then python_bin=.venv101/bin/python
    elif [[ -x .venv/bin/python ]]; then python_bin=.venv/bin/python
    else python_bin=python3
    fi
fi
out=${OUT:-runs/gece2}
hours=${HOURS:-11}
save_minutes=${SAVE_MINUTES:-5}
tokenizer=../04_TOKENIZERLAR/valerois_tokenizer_8k.json
window="$out/gece2.window"
checkpoint="$out/last.pt"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
# gfx1201 is native on ROCm 10.1. A caller-supplied HSA_OVERRIDE_GFX_VERSION is inherited.

die() { echo "gece2: $*" >&2; exit 1; }
if [[ ${1:-} == --hours ]]; then
    (( $# >= 2 )) || die "--hours bir sayı gerektiriyor"
    hours=$2
    shift 2
fi
mkdir -p -- "$out"
command -v flock >/dev/null || die "flock (util-linux) gerekli"
exec 9>"$out/.gece2.lock"
flock -n 9 || die "Bu OUT dizininde başka bir gece2 çalışıyor: $out"
if [[ -d "$out/export" ]]; then
    echo "Servis paketi: $out/export"
    exit "$(cat "$out/export/gece2.status")"
fi
[[ -s "$tokenizer" ]] || die "8K tokenizer bulunamadı: $tokenizer"

new_run=0
if [[ -f "$window" ]]; then
    mapfile -t run < "$window"
    start=${run[0]}
    hard_deadline=${run[1]}
    data=("${run[@]:2}")
    [[ ${#data[@]} -gt 0 ]] || die "Kayıtlı eğitim verisi boş: $window"
    if (( $# > 0 )) && [[ "$(printf '%s\n' "$@")" != "$(printf '%s\n' "${data[@]}")" ]]; then
        die "Veri listesi kayıtlı koşudan farklı. Yeni bir OUT kullanın."
    fi
else
    [[ ! -e "$checkpoint" ]] || die "Mevcut checkpoint gece2'ye ait değil. Yeni bir OUT kullanın."
    (( $# > 0 )) || die "8K veri gerekli. Örnek: bash bench/gece2.sh veri/fineweb.bin:0.7 veri/cosmo.bin:0.3"
    new_run=1
    data=("$@")
    start=$(date +%s)
    hard_deadline=$("$python_bin" - "$hours" "$start" <<'PY'
import math
import sys

try:
    hours = float(sys.argv[1])
except ValueError:
    sys.exit("--hours bir sayı gerektiriyor")
if not math.isfinite(hours) or hours * 3600 < 360:
    sys.exit("--hours en az 0.1 olmalı")
print(int(sys.argv[2]) + int(hours * 3600))
PY
)
fi
train_deadline=$((hard_deadline - 300))

remaining=$((train_deadline - $(date +%s)))
scan_status=0
if (( remaining > 0 )); then
    timeout --signal=KILL "${remaining}s" "$python_bin" - "${data[@]}" <<'PY' || scan_status=$?
import math
from pathlib import Path
import sys

import numpy as np
import torch

from heimdall.config import PRESETS
from heimdall.data import parse_spec

if not torch.version.hip or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    sys.exit("ROCm PyTorch ve BF16 destekli AMD GPU gerekli")
preset = PRESETS["gece"]
for spec in sys.argv[1:]:
    path, weight = parse_spec(spec)
    if weight is not None and (not math.isfinite(weight) or weight <= 0):
        sys.exit(f"Geçersiz veri ağırlığı: {spec}")
    size = Path(path).stat().st_size
    if size % 2 or int(size // 2 * 0.995) <= preset.seq_len + 1:
        sys.exit(f"Geçersiz veya kısa uint16 veri dosyası: {path}")
    if np.memmap(path, dtype=np.uint16, mode="r").max() >= preset.model.vocab_size:
        sys.exit(f"8K sözlük dışında token kimliği: {path}")
PY
fi
if (( scan_status != 0 )); then
    [[ -s "$checkpoint" ]] || die "Veri ön kontrolü başarısız (kod $scan_status); checkpoint yok."
    echo "gece2: veri ön kontrolü başarısız (kod $scan_status); son checkpoint dışa aktarılıyor." >&2
fi
if (( new_run )); then
    # Retries share the original WSD window and the validated input list.
    printf '%s\n' "$start" "$hard_deadline" "${data[@]}" > "$window.tmp"
    mv -- "$window.tmp" "$window"
fi

status=0
retry=0
nan_count=0
lr_scales=(1.0 0.5 0.25 0.125 0.0625 0.03125)
if [[ -f "$out/gece2.retry" ]]; then
    read -r retry nan_count < "$out/gece2.retry"
fi
echo "gece2: eğitim $(date -d "@$train_deadline"), son teslim $(date -d "@$hard_deadline")"
while (( scan_status == 0 && $(date +%s) < train_deadline )); do
    if [[ -s "$checkpoint" ]]; then
        args=(--resume "$checkpoint")
    else
        args=(--preset gece --compile --device cuda --out "$out"
              --save-every 0 --log-every 20 --eval-every 200 --eval-batches 16 --data "${data[@]}")
    fi
    remaining=$((train_deadline + 45 - $(date +%s)))
    status=0
    timeout --signal=INT --kill-after=15s "${remaining}s" "$python_bin" -u -m heimdall.train "${args[@]}" \
        --wall-start "$start" --deadline "$train_deadline" --save-minutes "$save_minutes" \
        --retry-index "$retry" --lr-scale "${lr_scales[nan_count]}" || status=$?
    case "$status" in
        0|124) status=0; break ;;
        130) exit 130 ;;
        137) if (( $(date +%s) >= train_deadline )); then status=0; break; fi ;;
        125|126|127) die "Eğitim başlatılamadı (kod $status)" ;;
    esac
    if (( retry >= 5 )); then
        echo "gece2: yeniden deneme sınırına ulaşıldı (kod $status)" >&2
        break
    fi
    retry=$((retry + 1))
    if (( status == 75 )); then nan_count=$((nan_count + 1)); fi
    printf '%s %s\n' "$retry" "$nan_count" > "$out/gece2.retry.tmp"
    mv -- "$out/gece2.retry.tmp" "$out/gece2.retry"
    echo "gece2: eğitim kesildi (kod $status), deneme $retry, LR ×${lr_scales[nan_count]}"
    remaining=$((train_deadline - $(date +%s)))
    if (( remaining <= 0 )); then break; fi
    sleep "$((remaining < 10 ? remaining : 10))"
done

[[ -s "$checkpoint" ]] || die "Dışa aktarılacak başarılı eğitim adımı yok."
remaining=$((hard_deadline - $(date +%s)))
(( remaining > 0 )) || die "Süre doldu. Son checkpoint: $checkpoint"
timeout --signal=KILL "${remaining}s" "$python_bin" -m heimdall.export --ckpt "$checkpoint" \
    --tokenizer "$tokenizer" --out "$out/.export.tmp"
printf '%s\n' "$status" > "$out/.export.tmp/gece2.status"
mv -T -- "$out/.export.tmp" "$out/export"
echo "Servis paketi: $out/export"
exit "$status"
