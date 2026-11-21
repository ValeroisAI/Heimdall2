# Oturum notu: gece koşusu + Gjallar

Başka bir oturumun devam edebilmesi için durum özeti. Ayrıntı: `docs/GECE_RUNBOOK.md`, `docs/GJALLAR_PLAN.md`, `docs/GJALLAR_SPEC.md`.

## Ne yapıldı
**1. Gece koşusu (Ubuntu / ROCm, tek komut)**
- `heimdall/config.py`: `gece` preset'i. 97,7M parametre, d_model 768, 12 katman, ikili (1-bit) FFN, T=2048, 8K tokenizer, AdamW lr 2e-3, WSD takvimi. Daha önce 4,7 saatte eğitilen tarifle aynı; yeni mimari yok.
- `bench/gece2.sh`: bitiş saatine göre çalışır, WSD düşüşü o saate denk gelir, 5 dakikada bir kayıt, çökme/NaN'de son kayıttan devam, sonda `OUT/export`.
- `bench/veri_gece.sh`: yaklaşık 2,4B token (FineWeb-Edu %50, Cosmopedia %20, kod %20, OpenWebMath %6, WikiHow %4). Sonda `--data` dizesini yazdırır.
- `bench/sft_gece.sh`, `bench/sft_egit.py`: 20-30 dk ChatML SFT, kayıp yalnız asistan tokenlarında.
- `bench/degerlendir_gece.sh`: doğrulama kaybı, passkey, ARC/PIQA, 10 sabit istem.
- `bench/onkontrol_rocm.sh` (ROCm ön kontrolü), `bench/izle.py` (hız/ETA/NaN izleme), `.gitattributes` (`*.sh`, `*.py` LF kalır).
- Kod incelemesinde üç hata bulunup düzeltildi: veri taraması ölünce export atlanması; `sft_egit.py`'de eksik `nullcontext` importu, NaN koruması ve periyodik kayıt; `degerlendir_gece.sh`'in `--ckpt` okumaması.

**2. Gjallar (KV okuma seçici/geçit)**
- Çalışan: 4-bit blok merkezi + shortlist + tam rerank, oracle dikkat kütlesinin ~%89-99'unu yakalar. Geçit adımların %25'inde açıkken gecikmeli pass-key 3/3.
- Çalışmayan: rezervuar tek başına (pass-key 0/3), sınırlı birleşik geçit (0/3), yüksek LSE yüzdelikleri (0/3), ham GDN çıktı normu (AUROC 0,50), okuma sınırı bittikten sonra gelen geç iğne.
- Hız kazancı yok. Skor/top-k/gather hâlâ CPU'da. Tek 0.8B model, dört pass-key; 128K üstü ve eğitilmiş model denenmedi.

## Doğrulanan / doğrulanmayan
- Doğrulandı: betiklerin `bash -n` sözdizimi, derleme, veri/SFT testleri 4/4, `gece` preset'i yükleniyor, `gece2.sh` için sahte çökme/NaN senaryosu (GPT).
- **Doğrulanmadı:** eğitim → export → SFT → değerlendirme zinciri uçtan uca; ROCm'da gerçek tok/s ve VRAM (bu makinede ROCm yok); `torch.compile`/Triton GPU'da. "11 saatte 1,6-1,85B token" yalnızca README'deki 47,4K tok/s'ten hesaplanmış tahmin.
- Beklenti: akıcı ama sık yanlış bilgi veren, GPT-2 small düzeyinde taban model; SFT sonrası küçük sohbet modeli. Tokenizer 8K olduğu için kayıp değerleri halka açık modellerle karşılaştırılamaz.

## Bu oturumda yapılmayanlar
Katman, quant, optimizer ve tokenizer üzerinde yeni deney yapılmadı; bunlarla ilgili sonuçlar README'deki eski ölçümlerdir.

## Sıradaki adım (Ubuntu)
1. `bash bench/onkontrol_rocm.sh`
2. `gece2.sh` ile 5-10 dakikalık deneme; gerçek tok/s buradan çıkar, `--hours` ve token bütçesi ona göre ayarlanır.
3. `bash bench/veri_gece.sh`, sonra 11 saatlik koşu (`systemd-inhibit --what=idle:sleep bash bench/gece2.sh --hours 11 <veri_gece.sh'nin yazdırdığı --data>`).
4. Sonra SFT ve değerlendirme. Tahmin tutmazsa bütçeyi yeniden ayarla.
5. Gjallar için: skor/top-k/gather'ı GPU'ya taşımak, geç iğne için bütçe stratejisi, daha büyük model.
