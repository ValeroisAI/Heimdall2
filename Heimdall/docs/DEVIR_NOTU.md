# Devir notu (2026-11-24): C katmanı, döngü deneyi, gece koşusu

Başka bir oturumun kaldığı yerden sürdürebilmesi için durum özeti. Ayrıntı: `docs/GECE_RUNBOOK.md`, `docs/OTURUM_GECE_GJALLAR.md`.

## Depo
- Depo: `ValeroisAI/Heimdall2` (özel). Eski depo (`ValeroisAI/Bifrost`) silindi; geçmiş yeniden yazıldı (commit numaraları değişti; eski numaralar geçersiz).
- Dal: `claude/lucid-pasteur-h2nrvj` bütün işi taşır. `main` yalnızca gece koşusu + Gjallar birleşmesine kadar gelir; C katmanı, döngü ve test düzeltmeleri henüz `main`'de yok.
- Ev makinesi: RX 9070 XT 16 GB, Ubuntu, `~/Desktop/bifrost/Heimdall`, `source .venv101/bin/activate` şart. Bu bulut oturumunda GPU yok.

## Yapılanlar (ölçülmüş sayılarla)
- **Hız:** fla autotune genişletme + BitLinear ölçek katlama + BitFlip temizliği → 46.8K → **58.0K tok/s** (37 TFLOPS), VRAM 13.3 → 10.0 GB.
- **C katmanı** (LFM2 tarzı kapılı kısa evrişim, `--arch kisa` = `CCCA`, dikkat NoPE): kod ve testler hazır.
  - Düello (10 dk, tek tohum, token tabanlı, RoPE'lu sürüm): kisa **0.2496**, tf 0.2731, hibrit 0.3390 bit/bayt (düşük iyi). NoPE'lu `kisa` bu düelloda **henüz koşulmadı**.
  - MQAR (uzun bağlam hatırlama, boşluk 64 / 1K / 16K): hibrit DA 1.00/1.00/1.00; C+RoPE dikkat 1.00/0.06/0.00; C+NoPE dikkat 0.87/0.90/0.88; saf dikkat 0.15/0.02/0.02. Sonuç: RoPE'lu dikkat uzunluğa genellenmiyor; `C` varsa dikkat otomatik NoPE.
- **Gece koşusu** (`runs/gece2`, ev makinesinde): 97.7M param, `DDDA`, ikili FFN, 8625 adım, 2.26B token, ~11 saat. Doğrulama: fineweb 2.979 (ppl 19.7), cosmo 1.986 (ppl 7.3). Karışım `fineweb.bin:0.5 cosmo.bin:0.5` (**kod yok**, bu yüzden kod/JSON istemleri çöp çıkıyor). Örnek üretim: cümle düzeyinde akıcı, tutarlılık ve talimat izleme zayıf (SFT yok). Eski 4.7 saatlik koşu (%32 kodlu karışım): fineweb 3.13.
- **Döngü** (GLM-5.4 tarzı, `--dongu BAŞ:SON`, `--baslangic ckpt`): hazır checkpoint'in katman aralığını iki kez çalıştırır, token başına kapı (kapalı başlar). Kod + test + CPU uçtan uca duman testi tamam; **GPU'da koşulmadı**.
- **Testler:** 92 geçti, 1 atlandı (`lm_eval` yok). Triton testleri ve transformers referans testi ayrı süreçte koşar (fla/Triton yüklüyken yorumlayıcı ayarı geç kalıyordu).
- **Ön kontrol ve gece betikleri** (`bench/onkontrol_rocm.sh`, `gece2.sh`): etkin venv'i seçer; Triton duman testi dosyadan koşar.

## Bekleyen işler (sırayla)
1. `bash bench/dongu_deney.sh 2>&1 | tee dongu.log` (~2.5-3 saat): `runs/gece2/last.pt`'den kontrol (döngüsüz) ve döngü kolu, aynı veri/token/LR; sonda tablo. Ardından aynı betik gece modelini SFT eder (`SFT=0` ile atlanır). OOM olursa `LOOP_MICRO=4`. Fark ~0.01 nat'tan küçükse gürültü sayın (tek tohum); döngü ~%33 fazla hesap harcar.
2. NoPE düellosu: `python bench/yama_duello.py --dakika 10 --kollar kisa kisa_rope --seed 1 --out runs/duello_nope` (~25 dk). Hangi mimariyle ikinci gece koşusu yapılacağına bu karar verir.
3. SFT sonrası: `bash bench/degerlendir_gece.sh` (passkey, ARC/PIQA, 10 istem). `sft_gece.sh` ve `degerlendir_gece.sh` GPU'da hiç koşulmadı.
4. Sonraki gece koşusunda karışıma kod ekle (`bench/veri_gece.sh`, ~2.4B token, en az 35 GB).
5. Gjallar: skor/top-k/gather hâlâ CPU'da, hız kazancı yok (bkz. `OTURUM_GECE_GJALLAR.md`).

## Çalıştırma kuralları
- GPU'da aynı anda iki iş koşturma (OOM görülmüştü). Başlamadan: `/opt/rocm/bin/rocm-smi --showmeminfo vram` boş mu bak. `/opt/rocm/bin`'i PATH'e ekleme.
- `ulimit -n 65536` (memlock uyarısı engel olmadı).
- `gece2.sh --hours` en az 0.1; son 5 dakika export'a ayrılır (0.1 saatte eğitim penceresi 60 sn, tok/s ölçülmez; deneme için 0.3).
- Disk: yeterli boş yer bırak (gece koşusu için ≥ 20 GB).

## Güvenlik
- `valerois.com` yönetim anahtarı ve yedek parolası eski depo geçmişinde açıktaydı; geçmişten temizlendi. **Sunucuda değiştirilip değiştirilmediği doğrulanmadı**: değiştirilmediyse önce bu. Değerleri bu depoya yazma.
- Commit yazarı `360678560+berkayshnnz@users.noreply.github.com` (e-posta yeniden yazıldı); Claude commit'leri `noreply@anthropic.com`.
