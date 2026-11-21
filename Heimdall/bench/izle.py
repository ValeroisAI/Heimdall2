#!/usr/bin/env python3
"""Eğitim günlüğünü izler: tok/s, ETA, kayıp eğilimi, NaN/takılma, GPU ısı ve VRAM.

    python bench/izle.py gece2.log --out runs/gece2
    python bench/izle.py --out runs/gece2 --stall-min 20 --ntfy https://ntfy.sh/<konu>
    NTFY_URL=... WEBHOOK_URL=... python bench/izle.py gece2.log --out runs/gece2

gece2 süreli koşar (adım satırı `süreli`). ETA, runs/gece2/gece2.window ikinci satırındaki
sert teslimden 300 sn öncesidir (gece2.sh ile aynı pay). Bildirim yalnız izlerken
görülen takılma, NaN, OOM, aşırı ısı ve bitiş için gider; eski satırlar tekrar basılmaz.
Ek paket yok. Çıkış: --selftest başarısızsa 1, aksi halde 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

STEP_RE = re.compile(
    r"adım\s+(\d+)/(\d[\d,]*|süreli)\s+\|\s+loss\s+(\S+)\s+\|\s+([\d,]+)\s+tok/s",
    re.IGNORECASE,
)
HEAD_RE = re.compile(r"adım başına\s+([\d,]+)\s+token,\s+(\d[\d,]*|süreli)\s+adım")
TEMP_RE = re.compile(r"Temperature \(Sensor (edge|junction|memory)\) \(C\):\s*([0-9.]+)", re.I)
USED_RE = re.compile(r"VRAM Total Used Memory \(B\):\s*([0-9]+)", re.I)
TOTAL_RE = re.compile(r"VRAM Total Memory \(B\):\s*([0-9]+)", re.I)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _finite(value: float) -> bool:
    return math.isfinite(value)


class State:
    def __init__(self) -> None:
        self.step: int | None = None
        self.total: int | None = None
        self.timed = False
        self.loss: float | None = None
        self.tok_s: float | None = None
        self.tokens_per_step: int | None = None
        self.losses: list[float] = []
        self.last_progress = time.monotonic()
        self.saw_step = False
        self.nan = False
        self.oom = False
        self.finished = False
        self.deadline: float | None = None
        self.note = ""

    def mark(self) -> None:
        self.last_progress = time.monotonic()
        self.saw_step = True


def load_deadline(out: Path | None) -> float | None:
    """gece2.window: satır1 başlangıç, satır2 sert teslim (unix). Eğitim 300 sn önce durur."""
    if out is None:
        return None
    path = out / "gece2.window"
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return None
    try:
        return int(lines[1]) - 300
    except ValueError:
        return None


def ingest(state: State, line: str) -> str | None:
    """Bir satırı işle. Yeni bir adım görüldüyse 'step', olay varsa 'nan'/'oom'/'finish', yoksa None."""
    text = line.strip()
    if not text:
        return None
    head = HEAD_RE.search(text)
    if head:
        state.tokens_per_step = int(_num(head.group(1)))
        if head.group(2) == "süreli":
            state.timed = True
        else:
            state.total = int(_num(head.group(2)))
        return None
    step = STEP_RE.search(text)
    if step:
        new_step = int(step.group(1))
        if state.saw_step and state.step == new_step:
            return None
        state.step = new_step
        if step.group(2) == "süreli":
            state.timed = True
        else:
            state.total = int(_num(step.group(2)))
        try:
            loss = float(step.group(3))
        except ValueError:
            loss = math.nan
        state.loss = loss
        state.tok_s = _num(step.group(4))
        state.mark()
        if not _finite(loss):
            state.nan = True
            state.note = text
            return "nan"
        state.losses.append(loss)
        del state.losses[:-8]
        if state.total is not None and state.step >= state.total:
            state.finished = True
            return "finish"
        return "step"
    if text.startswith("{") and '"step"' in text:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "step" in obj:
            new_step = int(obj["step"])
            if "loss" in obj and state.saw_step and state.step == new_step:
                return None
            state.step = new_step
            if "tok_s" in obj:
                state.tok_s = float(obj["tok_s"])
            if "loss" in obj:
                state.loss = float(obj["loss"])
                if not _finite(state.loss):
                    state.nan = True
                    state.note = text
                    state.mark()
                    return "nan"
                state.losses.append(state.loss)
                del state.losses[:-8]
            state.mark()
            if state.total is not None and state.step >= state.total and "loss" in obj:
                state.finished = True
                return "finish"
            return "step"
    low = text.lower()
    if "sonlu değil" in text or "kaydedilmedi" in low:
        state.nan = True
        state.note = text
        state.mark()
        return "nan"
    if "out of memory" in low or "outofmemory" in low:
        state.oom = True
        state.note = text
        return "oom"
    if text.startswith("Servis paketi:") or "EĞİTİM BİTTİ" in text or "=== bitti" in text or "Eğitim süresi doldu" in text:
        state.finished = True
        state.note = text
        state.mark()
        return "finish"
    return None


def trend(losses: list[float]) -> str:
    if len(losses) < 4:
        return "·"
    first, last = losses[0], losses[-1]
    if last > first + 0.15 and losses[-1] > losses[-2] > losses[-3]:
        return "↑"
    if last < first - 0.02:
        return "↓"
    return "→"


def rising_hard(state: State) -> bool:
    losses = state.losses
    if state.step is None or state.step < 100 or len(losses) < 6:
        return False
    return all(b > a for a, b in zip(losses, losses[1:])) and losses[-1] > min(losses) + 0.5


def eta_seconds(state: State, now: float | None = None) -> float | None:
    now = time.time() if now is None else now
    if state.deadline is not None:
        return max(0.0, state.deadline - now)
    if (state.total is None or state.step is None or not state.tok_s
            or not state.tokens_per_step or state.total <= state.step):
        return None
    return (state.total - state.step) * state.tokens_per_step / state.tok_s


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}sa{minutes:02d}dk"
    return f"{minutes}dk"


def fmt_status(state: State, gpu: str) -> str:
    step = "?" if state.step is None else str(state.step)
    total = "süreli" if state.timed and state.total is None else (str(state.total) if state.total else "?")
    loss = "nan" if state.loss is None or not _finite(state.loss) else f"{state.loss:.4f}"
    tok = "?" if not state.tok_s else f"{state.tok_s:,.0f}"
    clock = time.strftime("%H:%M:%S")
    return (f"{clock}  adım {step}/{total}  loss {loss} {trend(state.losses)}  "
            f"{tok} tok/s  ETA {fmt_eta(eta_seconds(state))}  {gpu}")


def stalled(state: State, started: float, stall_s: float, grace_s: float, now: float | None = None) -> bool:
    now = time.monotonic() if now is None else now
    if state.finished:
        return False
    if not state.saw_step:
        return now - started >= grace_s
    return now - state.last_progress >= stall_s


def read_gpu() -> tuple[float | None, float | None, float | None]:
    """junction/edge °C, kullanılan GB, toplam GB. rocm-smi yoksa (None, None, None)."""
    exe = shutil.which("rocm-smi") or ("/opt/rocm/bin/rocm-smi" if os.path.isfile("/opt/rocm/bin/rocm-smi") else "")
    if not exe:
        return None, None, None
    try:
        out = subprocess.run([exe, "--showtemp", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None, None, None
    text = out.stdout + "\n" + out.stderr
    temps = {name.lower(): float(val) for name, val in TEMP_RE.findall(text)}
    temp = temps.get("junction", temps.get("edge"))
    used = USED_RE.search(text)
    total = TOTAL_RE.search(text)
    used_gb = int(used.group(1)) / 2**30 if used else None
    total_gb = int(total.group(1)) / 2**30 if total else None
    return temp, used_gb, total_gb


def gpu_text(temp: float | None, used: float | None, total: float | None) -> str:
    parts = []
    parts.append("GPU ?" if temp is None else f"GPU {temp:.0f}°C")
    if used is not None and total is not None:
        parts.append(f"VRAM {used:.1f}/{total:.1f}GB")
    elif used is not None:
        parts.append(f"VRAM {used:.1f}GB")
    return " ".join(parts)


def push(kind: str, text: str, ntfy: str | None, webhook: str | None, payload: dict) -> None:
    print(f"BİLDİRİM {kind}: {text}", flush=True)
    if ntfy:
        priority = "high" if kind in {"stall", "nan", "oom", "hot"} else "default"
        req = urllib.request.Request(
            ntfy, data=text.encode(),
            headers={"Title": f"Heimdall {kind}", "Priority": priority, "Tags": kind},
        )
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"ntfy iletilemedi: {exc}", file=sys.stderr, flush=True)
    if webhook:
        body = json.dumps({"event": kind, "text": text, **payload}).encode()
        req = urllib.request.Request(webhook, data=body, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"webhook iletilemedi: {exc}", file=sys.stderr, flush=True)


class Tail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.buf = ""

    def read_new(self) -> list[str]:
        if not self.path.is_file():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.buf = ""
        if size == self.offset:
            return []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        self.buf += chunk
        lines = self.buf.split("\n")
        self.buf = lines.pop() if lines else ""
        return lines


def snapshot(paths: list[Path], state: State) -> None:
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            ingest(state, line)


def file_age(paths: list[Path]) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path.is_file()]
    if not mtimes:
        return None
    return max(0.0, time.time() - max(mtimes))


def follow(paths: list[Path], state: State, args: argparse.Namespace, started: float) -> None:
    tails = [Tail(path) for path in paths]
    for tail in tails:
        if tail.path.is_file():
            tail.offset = tail.path.stat().st_size
    announced = False
    sent: dict[str, float] = {}
    last_gpu = 0.0
    gpu = "GPU ?"
    last_print = 0.0
    stall_s = args.stall_min * 60
    grace_s = args.grace_min * 60
    remind_s = max(stall_s, 30 * 60)

    def notify(kind: str, text: str, repeat: bool = False) -> None:
        now = time.monotonic()
        if kind in sent and (not repeat or now - sent[kind] < remind_s):
            return
        sent[kind] = now
        push(kind, text, args.ntfy, args.webhook, {
            "step": state.step, "loss": state.loss, "tok_s": state.tok_s,
            "eta_s": eta_seconds(state), "note": state.note,
        })

    while True:
        got = False
        for tail in tails:
            for line in tail.read_new():
                got = True
                kind = ingest(state, line)
                if kind == "step":
                    print(fmt_status(state, gpu), flush=True)
                    last_print = time.monotonic()
                    if rising_hard(state):
                        notify("loss", fmt_status(state, gpu) + " kayıp 6 adımdır yükseliyor")
                elif kind == "nan":
                    notify("nan", state.note or "kayıp sonlu değil")
                elif kind == "oom":
                    notify("oom", state.note or "GPU belleği doldu")
                elif kind == "finish":
                    print(fmt_status(state, gpu) + "  BİTTİ", flush=True)
                    notify("finish", state.note or fmt_status(state, gpu))
                    return
        now_m = time.monotonic()
        if now_m - last_gpu >= 60:
            temp, used, total = read_gpu()
            gpu = gpu_text(temp, used, total)
            last_gpu = now_m
            if temp is not None and temp >= args.hot_c:
                notify("hot", f"GPU {temp:.0f}°C  {fmt_status(state, gpu)}", repeat=True)
            elif temp is not None and temp < args.hot_c - 10:
                sent.pop("hot", None)
            if used is not None and used >= args.vram_gb:
                notify("vram", f"VRAM {used:.1f} GB  {fmt_status(state, gpu)}")
        if not announced and not any(p.is_file() for p in paths):
            print("günlük bekleniyor: " + ", ".join(str(p) for p in paths), flush=True)
            announced = True
        if stalled(state, started, stall_s, grace_s):
            notify("stall", "takıldı " + fmt_status(state, gpu), repeat=True)
        if got is False and now_m - last_print >= 60 and state.saw_step and not state.finished:
            print(fmt_status(state, gpu), flush=True)
            last_print = now_m
        if args.once:
            print(fmt_status(state, gpu), flush=True)
            return
        time.sleep(args.poll)


def selftest() -> None:
    state = State()
    header = "T=2048 mikro=8 birikim=16 → adım başına 262,144 token, süreli adım"
    assert ingest(state, header) is None
    assert state.tokens_per_step == 262144 and state.timed
    line = "adım     20/süreli | loss 3.1234 | 47,000 tok/s | ~24.0 TFLOPS | grad 1.02 | 13.3 GB | 5M tok"
    assert ingest(state, line) == "step"
    assert ingest(state, line) is None
    assert state.losses == [3.1234]
    assert state.step == 20 and abs(state.loss - 3.1234) < 1e-9 and abs(state.tok_s - 47000) < 1
    state.deadline = 1_700_000_000
    assert eta_seconds(state, now=1_700_000_000 - 90) == 90
    numbered = State()
    ingest(numbered, "T=2048 mikro=8 birikim=16 → adım başına 262,144 token, 3,051 adım")
    ingest(numbered, "adım     20/3051 | loss 3.0000 | 47,000 tok/s | ~1 TFLOPS | grad 1 | 13 GB | 5M tok")
    expect = (3051 - 20) * 262144 / 47000
    assert abs((eta_seconds(numbered, now=0) or 0) - expect) < 1
    assert ingest(numbered, "adım 21: kayıp veya gradyan sonlu değil; son iyi checkpoint korunuyor.") == "nan"
    assert ingest(State(), '{"step": 4, "tokens": 8, "loss": NaN, "tok_s": 10}') == "nan"
    oom = State()
    assert ingest(oom, "torch.OutOfMemoryError: HIP out of memory") == "oom"
    done = State()
    assert ingest(done, "Servis paketi: runs/gece2/export") == "finish"
    assert trend([3.0, 2.9, 2.8, 2.7]) == "↓"
    assert trend([2.0, 2.2, 2.4, 2.6]) == "↑"
    quiet = State()
    assert stalled(quiet, started=0, stall_s=10, grace_s=25, now=25)
    assert not stalled(quiet, started=0, stall_s=10, grace_s=25, now=10)
    quiet.saw_step = True
    quiet.last_progress = 100
    assert stalled(quiet, started=0, stall_s=20, grace_s=25, now=120)
    quiet.finished = True
    assert not stalled(quiet, started=0, stall_s=20, grace_s=25, now=1000)
    hard = State()
    hard.step = 120
    hard.losses = [2.0, 2.2, 2.4, 2.6, 2.8, 3.1]
    assert rising_hard(hard)
    print("SELFTEST OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heimdall gece koşusu izleyici")
    p.add_argument("log", nargs="?", help="stdout günlüğü (gece2.log) veya log.jsonl")
    p.add_argument("--out", help="eğitim dizini; log.jsonl ve gece2.window buradan okunur")
    p.add_argument("--stall-min", type=float, default=20, help="yeni adım gelmezse takılma (dk)")
    p.add_argument("--grace-min", type=float, default=25, help="ilk adımdan önce compile payı (dk)")
    p.add_argument("--hot-c", type=float, default=95.0)
    p.add_argument("--vram-gb", type=float, default=15.5, help="bunun üstünde VRAM uyarısı")
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--ntfy", default=os.environ.get("NTFY_URL"))
    p.add_argument("--webhook", default=os.environ.get("WEBHOOK_URL"))
    p.add_argument("--once", action="store_true", help="var olan günlüğü özetle ve çık")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    paths: list[Path] = []
    if args.log:
        paths.append(Path(args.log))
    out = Path(args.out) if args.out else None
    if out is not None:
        paths.append(out / "log.jsonl")
    if not paths:
        for guess in (Path("gece2.log"), Path("runs/gece2/log.jsonl")):
            paths.append(guess)
        out = out or Path("runs/gece2")
    # Aynı dosyayı iki kez izleme.
    uniq: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    state = State()
    state.deadline = load_deadline(out)
    snapshot(uniq, state)
    age = file_age(uniq)
    if state.saw_step and age is not None:
        state.last_progress = time.monotonic() - age
    temp, used, total = read_gpu()
    print(fmt_status(state, gpu_text(temp, used, total)), flush=True)
    if state.finished:
        print("BİTTİ " + (state.note or ""), flush=True)
        return 0
    if args.once:
        return 0
    print("izliyor: " + ", ".join(str(p) for p in uniq), flush=True)
    if state.deadline:
        print("ETA teslimi: " + time.strftime("%F %T", time.localtime(state.deadline)), flush=True)
    started = time.monotonic() - (age if age is not None and not state.saw_step else 0.0)
    try:
        follow(uniq, state, args, started)
    except KeyboardInterrupt:
        print("\nizleme durdu", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
