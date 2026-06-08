# Manuel test senaryoları — Sprint 0–7 (recreate sonrası)

Bu doküman, Docker stack **recreate + warm-up** sonrası dashboard ve API’yi doğrulamak içindir. Otomatik kısım için: `scripts/smoke_sprints_0_7.sh`.

## 0. Hazırlık

```bash
cd /home/batuhan/unified-ai-security/infra
docker compose build gateway dashboard
docker compose up -d --force-recreate gateway dashboard
```

Warm-up (~1–2 dk):

```bash
curl -s http://localhost:8000/health | head -c 400
# status: HEALTHY beklenir
```

Otomatik smoke (vault + mock + CRUD + auth_sources):

```bash
cd /home/batuhan/unified-ai-security
chmod +x scripts/smoke_sprints_0_7.sh
./scripts/smoke_sprints_0_7.sh
```

Gemini vault’un dolu olduğunu da dene:

```bash
RUN_GEMINI=1 ./scripts/smoke_sprints_0_7.sh
```

Image içi sprint testleri (rebuild sonrası tam suite):

```bash
RUN_PYTEST=1 ./scripts/smoke_sprints_0_7.sh
# veya:
docker exec uais-gateway python -m pytest tests/ -q
```

> **Not:** Yeni test dosyaları (`test_target_presets.py`, vb.) image’da yoksa `docker compose build gateway` şart.

---

## Sprint 0 — Vault

| # | Adım | Beklenen |
|---|------|----------|
| 0.1 | `curl -s http://localhost:8000/secrets` | JSON: `names`, `items`; **değer yok** |
| 0.2 | Dashboard → Targets → query auth → Vault’a key yapıştır → **Save** | “Stored `GEMINI_API_KEY` in local vault” |
| 0.3 | `curl -s http://localhost:8000/secrets/GEMINI_API_KEY` | `"source":"vault"` (env’de aynı isim yoksa) |
| 0.4 | `infra/.env` ve kök `.env` | `GEMINI_API_KEY` **yok** (env gölgelemesi testi) |

---

## Sprint 1 — Header `${VAR}`

| # | Adım | Beklenen |
|---|------|----------|
| 1.1 | Preset: **Google Gemini (x-goog-api-key header)** → Apply | Form: header auth, `x-goog-api-key: ${MY_GEMINI_KEY}` |
| 1.2 | `auth.token_env` yerine header’daki isim → Vault: `MY_GEMINI_KEY` | Badge: 🔒 stored |
| 1.3 | **Test connection** | 200, `auth_sources` içinde header `${MY_GEMINI_KEY}` → vault |
| 1.4 | (Alternatif) Query preset + vault | Senin mevcut `gemini_flash_test` ile aynı sonuç |

---

## Sprint 2 — Form UX

| # | Adım | Beklenen |
|---|------|----------|
| 2.1 | type = **web** | `auth.type` dropdown **yok**; “Session auth” + “Extra headers” |
| 2.2 | type = **api** | `auth.type` dropdown var (none/bearer/header/query/basic) |
| 2.3 | api + bearer | Inline token alanı **expander içinde**, kapalı |
| 2.4 | Vault boş + Test (bilerek) | 401/403 → uyarı: “paste into 🔐 Vault value” + secret adları listesi |

---

## Sprint 3 — Test connection diagnostics

| # | Adım | Beklenen |
|---|------|----------|
| 3.1 | Başarılı Test connection | Yeşil kutu + “Save … Run test” `st.info` |
| 3.2 | Expander: **Auth sources** | Tablo: `auth.token_env: …` → env / vault / missing |
| 3.3 | `curl` ile mock probe | `auth_sources` anahtarı JSON’da (değer yok) — smoke script yapar |

---

## Sprint 4 — Web cookie / storage_state

| # | Adım | Beklenen |
|---|------|----------|
| 4.1 | type = web, cookies JSON: `[{"name":"s","value":"${MY_SESSION}"}]` | Kayıt sonrası yaml `auth.type: cookie` |
| 4.2 | Vault: `MY_SESSION` + değer → Save | `/secrets` listesinde isim |
| 4.3 | `storage_state_path` = `/app/runs/state.json` | Path yaml’da; dosya container’da erişilebilir olmalı |
| 4.4 | Test connection (Playwright kuruluysa) | Başarı veya net “install playwright” mesajı |

Playwright container’da yoksa: `docker exec -it uais-gateway bash -lc "playwright install chromium"` (image’a bağlı).

---

## Sprint 5 — Form boşlukları

| # | Adım | Beklenen |
|---|------|----------|
| 5.1 | web → expander **Fallback selectors** | `fallback_input` / `fallback_response` yaml’da liste |
| 5.2 | type = **tools_local** | Açıklama bloğu; `has_tools` işaretleme hatırlatması |
| 5.3 | Formda `rate_limit` alanı | **Yok** (bilinçli; runner okumuyor) |

---

## Sprint 6 — Provider presets

| # | Adım | Beklenen |
|---|------|----------|
| 6.1 | Form üstü preset dropdown | OpenAI, Anthropic, Gemini (query/header), … |
| 6.2 | **(custom)** → Apply | Form boşalmaz / sentinel |
| 6.3 | Gemini (query) → Apply → id/name doldur → vault → Save | `targets.yaml` doğru endpoint/template |

---

## Sprint 7 — Run test

| # | Adım | Beklenen |
|---|------|----------|
| 7.1 | Run test sayfası | `enabled: false` target’lar `(disabled)` suffix |
| 7.2 | Hiç target yoksa | `st.error` + dur |
| 7.3 | `gemini_flash_test` + küçük suite | Run başlar, `runs/` altında artefakt |

---

## Hızlı regresyon matrisi (tek sayfa)

| Alan | Kontrol |
|------|---------|
| Vault yaz/oku/sil | smoke § Sprint 0 |
| DELETE target | smoke § CRUD |
| Mock test | smoke § Sprint 3 |
| auth_sources | smoke + dashboard expander |
| Gemini canlı | `RUN_GEMINI=1` veya dashboard Test |
| UI preset | manuel 6.x |
| Web cookie | manuel 4.x (Playwright gerekir) |

---

## Sorun giderme

| Belirti | Olası neden |
|---------|-------------|
| DELETE /targets → 500 errno 30 | `external_eval` `:ro` mount — override’ı kontrol et |
| DELETE → EBUSY | Çift mount (dir + targets.yaml) — `target_loader` fix + recreate |
| `/targets/test` ImportError `iter_var_names` | Eski container — `docker compose up -d --force-recreate gateway` |
| pytest 490, host’ta 564 | Image rebuild; veya `tests/` mount ekle |
| Test 401, vault dolu | `.env`’de aynı isimle boş env gölgelemesi |
