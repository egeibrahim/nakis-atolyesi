# Nakış Atölyesi

Görselden nakış makinesi dosyası (PES / DST) üreten açık kaynak araç.

- **Site:** https://egeibrahim.github.io/nakis-atolyesi/
- **Hızlı motor:** Tarayıcıda çalışır, sunucu gerekmez (`index.html`).
- **Profesyonel motor:** Dikişleri [Ink/Stitch](https://inkstitch.org) üretir; `server/` klasöründeki servis gerekir.

## Özellikler
- Renk ayırma, arka plan temizleme (sadece dış zemin / tüm zemin rengi)
- Otomatik saten (ince bölgeler) ve tatami dolgu (geniş bölgeler)
- Bölgeye tıklayıp dikiş tipi, sıklık, açı, çekme telafisi ve alt dikiş seçme
- PES (iplik renkleriyle), DST ve Ink/Stitch SVG çıktısı

## Profesyonel motoru yayınlama (Render)
1. https://render.com adresinde GitHub hesabınla giriş yap.
2. **New → Blueprint** seç, bu repoyu bağla, **Apply** de. Ayarlar `render.yaml` dosyasından gelir.
3. Yayın bitince servis adresini (ör. `https://nakis-atolyesi-api.onrender.com`) kopyala.
4. Sitede **Motor → Profesyonel (Ink/Stitch)** seç ve adresi **Sunucu adresi** alanına yapıştır.

Yerelde çalıştırmak için: `docker build -t nakis-api server && docker run -p 8000:8000 nakis-api`

## API
`POST /api/digitize` — multipart: `image` (dosya), `params` (JSON: `width`, `ncol`, `removeBg`, `satin`, `minDet`, `row`, `maxst`, `angle`, `underlay`, `overrides`, `colorOverrides`).
Yanıt: `pes`, `dst` (base64), `svg`, `stitches`, `threads`, `count`, `trims`.

## Lisans
GPL-3.0. Profesyonel motor, GPL-3.0 lisanslı Ink/Stitch v3.3.0'ı kullanır.
