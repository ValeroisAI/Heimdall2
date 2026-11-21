# Gece koşusu (Ubuntu, RX 9070 XT)

11 saatlik tek koşu. Kart: **RX 9070 XT 16 GB, gfx1201**. Yığın: **PyTorch 2.14 + ROCm 10.1**, `torch.compile` açık. Model: `bench/gece2.sh` içindeki `gece` ön ayarı (ölçülmüş 97.7M, ikili FFN, T=2048, mikro-batch 8, adım başı 262144 token). Eğitim `runs/gece2/` altına yazılır.

Komutların hepsi iç depoda çalışır: `Heimdall/Heimdall` (yanında `bench/` ve `heimdall/` olan dizin).

Uyku bu betiğin içinde kapanmaz. Başlatma satırı `systemd-inhibit` ile sarılı olmadan bırakılırsa makine gece askıya alır ve koşu ölür.

## 1. Kurulum

```bash
cd Heimdall/Heimdall
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.14.0+rocm10.1.0"
pip install numpy tokenizers safetensors fastapi "uvicorn[standard]" pydantic huggingface_hub pyarrow httpx
pip install "flash-linear-attention>=0.2"    # olmazsa eğitim PyTorch çekirdeğine düşer, koşu yine yürür
```

`HSA_OVERRIDE_GFX_VERSION` yazma. gfx1201 tekerleği yerli ISA ister. Override ancak `rocminfo` içinde `gfx1201` hiç yoksa ve son çare olarak `12.0.1` olur.

Ortam (eğitim bunları kendi de kurar; kabukta durmaları iyi olur):

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

## 2. Ön kontrol

```bash
bash bench/onkontrol_rocm.sh
```

Her satır `PASS`, `WARN` veya `FAIL`. `FAIL` altındaki `düzelt:` komutunu uygula, betiği yeniden çalıştır. `SONUÇ` içinde FAIL varken başlama. GPU dumanı (bf16 matmul, `torch.compile`, Triton, varsa flash-linear-attention) birkaç dakika sürer. `heimdall.train` zaten çalışıyorsa duman testi atlanır; çalışan koşuyu düşürmesin.

Eşikler: boş disk ≥ 30 GB (`NEED_GB`), boş RAM hedefi 16 GB ve alt sınır 8 GB (`NEED_RAM_GB`).

## 3. Veri

Tokenizer varsayılanı `../04_TOKENIZERLAR/valerois_tokenizer_8k.json` (8192). Başka tokenizer ile üretilmiş `.bin` gece2'de "8K sözlük dışında token" diye düşer.

11 saat × ~47 bin tok/s ≈ 1.9 milyar görülen token. Dosyalar bundan küçük olabilir; yükleyici karışımı tekrarlar. Yine de birkaç yüz milyon token hazırla, tek parquet'i bütün gece ezberlemesin.

```bash
python -m heimdall.hazirla --repo HuggingFaceFW/fineweb-edu \
  --dosya sample/10BT/000_00000.parquet --kolon text \
  --out veri/fineweb.bin --max-token 500e6
python -m heimdall.hazirla --repo HuggingFaceTB/cosmopedia \
  --dosya data/web_samples_v2/train-00000-of-00118.parquet --kolon text \
  --out veri/cosmo.bin --max-token 200e6
```

Daha geniş karışım (indirmesi uzun, gece için şart değil): `bash bench/hafta_veri.sh`. `bash bench/veri_gece.sh` beş dosya (~2.4B token, en az 35 GB) yazar; çıktı adı `veri/cosmo.bin` değil `veri/cosmopedia.bin`. İndirme geceyi yerse yukarıdaki iki dosyayla başla. gece2'ye betiğin bastığı yolları ver, `cosmo.bin` ile karıştırma.

`bench/gece.sh` bu gece için kullanılmaz. O betik `last.pt` varken aynı çıktı dizinine sıfırdan başlar.

## 4. Eğitim

Fişte olsun. Oturum kapanmasın diye tmux. Inhibit, gece2'yi sarmalamak zorunda; betik bunu kendi yapmaz.

```bash
tmux new -s gece
source .venv/bin/activate
systemd-inhibit --what=idle:sleep --why="heimdall gece2" --mode=block \
  bash bench/gece2.sh --hours 11 veri/fineweb.bin:0.7 veri/cosmo.bin:0.3 \
  2>&1 | tee -a gece2.log
```

Ayrıl: `Ctrl-b` sonra `d`. Geri dön: `tmux attach -t gece`.

tmux yoksa (ön kontrol uyarır):

```bash
nohup systemd-inhibit --what=idle:sleep --why="heimdall gece2" --mode=block \
  bash bench/gece2.sh --hours 11 veri/fineweb.bin:0.7 veri/cosmo.bin:0.3 \
  > gece2.log 2>&1 &
echo $! > gece2.pid
```

Ne olur: pencere `runs/gece2/gece2.window` içine yazılır (başlangıç, sert teslim, veri listesi). Eğitim sert teslimden 5 dakika önce durur; kalan pay dışa aktarmadır. Kayıt en geç 5 dakikada bir `runs/gece2/last.pt`. Bitişte paket `runs/gece2/export`.

Aynı `OUT` ile ikinci kez veri listesini değiştirme. Başka veri için `OUT=runs/gece2b`.

## 5. İzleme

İkinci bir terminal (tmux penceresi de olur):

```bash
source .venv/bin/activate
python bench/izle.py gece2.log --out runs/gece2
```

Telefona düşsün (takılma, NaN, OOM, ≥95°C, bitiş):

```bash
python bench/izle.py gece2.log --out runs/gece2 --ntfy https://ntfy.sh/<gizli-konu>
# veya: --webhook https://ornek/hook   ya da NTFY_URL / WEBHOOK_URL
```

Satır: adım, loss ve eğilim (`↓` `→` `↑`), tok/s, ETA, GPU °C, VRAM. ETA, pencere dosyasındaki teslimden gelir. Varsayılan takılma 20 dk, ilk adımdan önce compile payı 25 dk (`--stall-min`, `--grace-min`). Tek bakış: `--once`.

`bench/dur.sh` ve `bench/devam.sh` hafta koşusuna (`hafta.sh`) aittir. gece2'yi onlar durdurmaz ve sürdürmez.

## 6. Sürdürme

Aynı komut. Pencere ve `last.pt` duruyorsa gece2 sıfırdan başlamaz, kayıttan devam eder. Saat argümanı ilk penceredeki teslimi değiştirmez.

```bash
systemd-inhibit --what=idle:sleep --why="heimdall gece2" --mode=block \
  bash bench/gece2.sh --hours 11 veri/fineweb.bin:0.7 veri/cosmo.bin:0.3 \
  2>&1 | tee -a gece2.log
```

Elle durdurma: tmux'ta `Ctrl-C`. Checkpoint yazılır, paket yazılmaz, çıkış 130. Devam için aynı komut. Paketi erken almak:

```bash
python -m heimdall.export --ckpt runs/gece2/last.pt \
  --tokenizer ../04_TOKENIZERLAR/valerois_tokenizer_8k.json \
  --out runs/gece2/export
```

`runs/gece2/export` varsa gece2 yeniden eğitim açmaz, paket yolunu basıp çıkar. Yeni koşu istiyorsan başka `OUT`.

## 7. SFT

Gece2 paketi yazıldıktan sonra, kart boşken. `heimdall.train` JSONL okumaz; SFT ayrı betiktir.

```bash
bash bench/sft_gece.sh
```

Veri yoksa `sft_veri.py` `veri/sft/train_metin.jsonl` üretir, sonra `bench/sft_egit.py` `runs/gece2/last.pt` üstünden ~25 dk çalışır. Çıktı: `runs/gece_sft/last.pt` ve `runs/gece_sft/export`. Başka taban: `CKPT=runs/gece2/last.pt MINUTES=20 bash bench/sft_gece.sh`. Kartı gece2 ile paylaşma.

Bu betik adım arasında checkpoint yazmaz ve NaN'de durmaz. Süreç ölürse baştan başlar. Bittiğinde `runs/gece_sft/sft_log.jsonl` içindeki loss sonlu değilse paketi kullanma.

## 8. Değerlendirme

Gece bittikten sonra. Eval gece2 ile aynı anda kartı doldurur.

```bash
CKPT=runs/gece2/last.pt bash bench/degerlendir_gece.sh
python bench/uret.py --ckpt runs/gece2/last.pt
python bench/lmeval_heimdall.py --ckpt runs/gece2/last.pt --tasks arc_easy,piqa --limit 200 --out runs/gece2/lmeval.json
```

`degerlendir_gece.sh` hem `--ckpt` hem `CKPT=` okur. SFT sonrası: `bash bench/degerlendir_gece.sh --ckpt runs/gece_sft/last.pt`.

Tam liste uzun sürer (`arc_easy,arc_challenge,hellaswag,piqa,winogrande,lambada_openai,wikitext`). 100M / ~2B token civarı bir taban modelde çoğu görev şansa yakındır; `lmeval_heimdall.py` şans düzeyini de basar.

## 9. Bir şey olursa

| Belirti | Ne yap |
|---|---|
| Ön kontrol FAIL | Satırdaki `düzelt:` komutu, sonra `bash bench/onkontrol_rocm.sh` tekrar. FAIL varken başlama. |
| `no kernel image` | `unset HSA_OVERRIDE_GFX_VERSION`. Tekerlek `torch[device-gfx1201]==2.14.0+rocm10.1.0` olsun. `python bench/rocm_teshis.py`. BLAS şüphesi: `TORCH_BLAS_PREFER_HIPBLASLT=0 python bench/rocm_teshis.py`. |
| Grup / izin | `sudo usermod -aG render,video "$USER"` ve oturumu kapatıp aç. `ls -l /dev/kfd`. |
| OOM | Mikro-batch'i büyütme. Ölçülen sığan ayar mikro 8, ~13 GB. Başka süreci kapat (`rocm-smi --showpids`). gece2 aynı ayarı birkaç kez dener, sonra durur; `last.pt` kaldıysa bölüm 6. |
| NaN (`sonlu değil`, çıkış 75) | gece2 öğrenme hızını yarıya indirip `last.pt`'den dener. Günlük bunu yazıyorsa dokunma. Beş denemede de düşerse `runs/gece2/gece2.retry` içindeki ikinci sayı 5'tir; daha küçük bir veri dumanı (`--hours 0.2`, yeni `OUT`) ile ayrı bak. |
| Takılma (izle "takıldı") | `tmux attach -t gece` ve `rocm-smi --showtemp`. Isı ≥95°C ise hava. Günlük ilerlemiyor ve süreç yoksa bölüm 6. `dmesg` içinde GPU reset varsa makineyi yeniden başlat, ön kontrol, sonra sürdürme. |
| Makine uyumuş | Inhibit'siz kalmıştır. Uyandır, bölüm 6. Bir sonraki sefer başlatma satırını olduğu gibi kullan. |
| Disk doldu | `df -h .`. Silinebilir olan `runs/boyut_*`. `last.pt` ve `gece2.window` silinmez. |
| Süre doldu, paket yok | `runs/gece2/last.pt` varsa bölüm 6'daki `heimdall.export`. Pencere teslimi geçtiyse gece2 "Süre doldu" der; export komutu teslimden bağımsız çalışır. |
| Geç sürdürme, veri taraması düştü | `gece2.sh` taramayı `timeout --signal=KILL` ile sınırlar ve çıkış kodunu yakalar. `last.pt` varsa eğitimi atlayıp paketi yazar. Checkpoint yoksa "Veri ön kontrolü başarısız" der; veri yolunu düzelt. Paket yine yoksa bölüm 6'daki `heimdall.export`. |
| Yanlış veri / sözlük | Yeni dizin: `OUT=runs/gece2b bash bench/gece2.sh ...`. Eski `gece2.window` veri listesini kilitler. |
| `bench/devam.sh` | Hafta koşusunu açar, gece2'yi değil. Kullanma. |
