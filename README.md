# 🎮 Game Price ETL Pipeline
### Advanced Analytics Pipeline — Rekomendasi Harga Top-Up Game Mobile

> Pipeline ETL end-to-end untuk menganalisis pasar top-up game mobile dan memberikan rekomendasi harga berbasis Machine Learning. Data diambil dari 28 sumber publik nyata, diproses, dimuat ke database, dan divisualisasikan melalui BI dashboard interaktif.

---

## 📋 Deskripsi Proyek

Proyek ini membangun **pipeline ETL otomatis** yang:

1. **Mengekstrak** data harga top-up, statistik pemain, kompetisi esports, dan konteks pasar dari 28 sumber publik (GamsGo, SEAGM, Business of Apps, Liquipedia, dll.)
2. **Mentransformasi** data mentah menjadi dataset bersih dengan feature engineering untuk analitik dan ML
3. **Memuat** hasil ke database (SQLite/Aiven PostgreSQL) + Parquet warehouse
4. **Menghasilkan** Business Intelligence (heatmap, chart korelasi) dan model ML rekomendasi harga

### 🎯 Tujuan & Manfaat

| Use Case | Deskripsi |
|---|---|
| **Serving Analisis** | BI dashboard + heatmap korelasi untuk insight harga kompetitif |
| **Serving ML** | Dataset 90 kombinasi (game × region × segmen) untuk model prediksi harga |
| **Keputusan Bisnis** | Kalkulator harga interaktif — input kondisi pasar → output harga optimal |

---

## 🏗️ Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIRFLOW DAG ORCHESTRATION                     │
│                                                                  │
│  start ──► extract ──► transform ──► load ────► train_ml ──► notify
│                                  └──► heatmap ─┘              │
└─────────────────────────────────────────────────────────────────┘

EXTRACT (Phase 1)                    TRANSFORM (Phase 2)
├── TopUpExtractor                   ├── transform_topup()      → fact_pricing.csv
│   └── GamsGo, SEAGM, GameBar,     ├── transform_game_stats() → dim_game.csv
│       LDShop, Codashop, BuffBuff  ├── transform_age()        → dim_age.csv
├── GameStatsExtractor               ├── transform_competition()→ fact_competition.csv
│   └── BusinessOfApps, Quantumrun, ├── transform_monthly_revenue() → fact_revenue.csv
│       rec0ded, BitTopup, Udonis    ├── build_ml_dataset()    → ml_features.csv
├── EsportsExtractor                 └── generate_heatmaps()   → 2× PNG
│   └── Liquipedia, EsportsCharts
└── MarketContextExtractor           LOAD (Phase 3)
    └── SQMagazine, TechRT, Accio   ├── SQLite / Aiven PostgreSQL
                                     ├── 6 tabel + audit_log
MACHINE LEARNING (Phase 4)          └── 6× Parquet warehouse
├── GradientBoosting ⭐ (best)
├── XGBoost                          OUTPUT
├── RandomForest                     ├── outputs/charts/ (5 PNG)
└── Ridge (baseline)                 ├── outputs/ml_model.joblib
                                     └── outputs/reports/
```

---

## 📁 Struktur Direktori

```
GamePriceETL/
├── config/
│   └── settings.py          # Semua konfigurasi via env variable
├── dags/
│   └── game_price_etl_dag.py # Apache Airflow DAG
├── src/
│   ├── extract/
│   │   └── extractor.py     # Phase 1: 4 extractor class
│   ├── transform/
│   │   └── transformer.py   # Phase 2: 6 transform + heatmap
│   ├── load/
│   │   └── loader.py        # Phase 3: DB load + Parquet + audit
│   ├── ml/
│   │   └── model.py         # Phase 4: 4 model ML + inference
│   └── bi/                  # BI dashboard (HTML interaktif)
├── data/
│   ├── raw/                 # JSON mentah dari extractor
│   ├── processed/           # CSV siap analisis (6 file)
│   └── warehouse/           # Parquet + SQLite DB
├── outputs/
│   ├── charts/              # Heatmap + ML chart PNG
│   └── reports/
├── tests/
│   └── test_pipeline.py     # Unit test semua phase
├── run_pipeline.py          # Runner standalone (tanpa Airflow)
├── requirements.txt
├── .env.example             # Template environment variable
└── .gitignore
```

---

## ⚙️ Cara Instalasi & Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/<username>/GamePriceETL.git
cd GamePriceETL
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment

```bash
cp .env.example .env
# Edit .env — isi DATABASE_URL jika pakai Aiven PostgreSQL
# Jika tidak diisi, pipeline otomatis pakai SQLite lokal
```

### 5. Jalankan Pipeline

```bash
# Full pipeline (Extract → Transform → Load → ML)
python run_pipeline.py

# Jalankan per phase
python run_pipeline.py --phase extract
python run_pipeline.py --phase transform
python run_pipeline.py --phase load
python run_pipeline.py --phase ml
```

### 6. (Opsional) Jalankan dengan Airflow

```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow users create --username admin --password admin --role Admin \
    --firstname Game --lastname ETL --email admin@example.com
cp dags/game_price_etl_dag.py $AIRFLOW_HOME/dags/

# Terminal 1
airflow scheduler

# Terminal 2
airflow webserver --port 8080

# Trigger manual
airflow dags trigger game_price_etl_pipeline
```

### 7. Jalankan Unit Test

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### 8. Jalankan Dashboard Streamlit (Lokal)

```bash
# Pastikan pipeline sudah dijalankan minimal sekali
bash run_app.sh

# Atau jalankan manual:
streamlit run streamlit_app/app.py
```

### 9. Deploy ke Streamlit Cloud ☁️

Dashboard dapat di-deploy gratis ke [Streamlit Cloud](https://share.streamlit.io/).

#### Langkah-langkah:

1. **Push ke GitHub**
   ```bash
   git add -A
   git commit -m "Siap deploy ke Streamlit Cloud"
   git push origin main
   ```

2. **Buka [share.streamlit.io](https://share.streamlit.io/)** → Login dengan GitHub

3. **Klik "New app"** dan isi:
   - **Repository**: pilih repo `Game-Pricing-ETL_BI_ML`
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. **Konfigurasi Secrets** (opsional, untuk koneksi Aiven PostgreSQL):
   - Klik **Advanced settings** → **Secrets**
   - Paste isi berikut (ganti dengan credential asli):
     ```toml
     DATABASE_URL = "postgresql://avnadmin:PASSWORD@host:port/defaultdb?sslmode=require"
     IDR_PER_USD = "16250"
     ```

5. **Custom Requirements** (penting!):
   - Jika deploy gagal karena `apache-airflow`, rename file requirements:
     ```bash
     # Backup requirements original
     cp requirements.txt requirements_full.txt
     # Gunakan versi slim untuk deploy
     cp requirements_streamlit.txt requirements.txt
     git add -A && git commit -m "Use slim requirements for Streamlit Cloud" && git push
     ```

6. **Klik "Deploy!"** — Streamlit Cloud akan:
   - Install dependencies dari `requirements.txt`
   - Jalankan `app.py`
   - Auto-generate dataset dari verified data (tanpa perlu menjalankan pipeline)

> **📝 Catatan:** Dashboard akan otomatis membuat dataset dari verified data jika CSV belum ada. Jika ingin data dari Aiven PostgreSQL, pastikan secrets sudah dikonfigurasi.

---

## 🗃️ Sumber Data (28 URL)

### Platform Top-Up
| Platform | URL |
|---|---|
| GamsGo | https://www.gamsgo.com/top-up/mobile-legends |
| SEAGM | https://www.seagm.com/en-us/mobile-legends-diamonds-top-up |
| GameBar | https://www.gamebar.gg/top-up/mobile-legends |
| LDShop | https://www.ldshop.gg/top-up/mobile-legends-bang-bang.html |
| Codashop | https://www.codashop.com/id-en/mobile-legends/ |
| TOPUPlive | https://www.topuplive.com/article/best-discounts-and-offers... |
| BuffBuff | https://buffbuff.com/ |
| Joytify | https://www.joytify.com/en-us |

### Analytics & Revenue
| Sumber | URL |
|---|---|
| Business of Apps | https://www.businessofapps.com/data/free-fire-statistics/ |
| Quantumrun – MLBB | https://www.quantumrun.com/consulting/mobile-legends-bang-bang/ |
| rec0ded – MLBB | https://rec0ded88.com/statistics/mobile-legends-bang-bang/ |
| rec0ded – Free Fire | https://rec0ded88.com/statistics/free-fire/ |
| CoopBoardGames | https://coopboardgames.com/online-gaming/mobile-legends-bang-bang/ |
| BitTopup Genshin | https://news.bittopup.com/news/genshin-impact-2025-15.2m-players-0.8b-revenue |
| IconEra – MLBB | https://icon-era.com/blog/mobile-legends-bang-bang-live-player-count-and-statistics.443/ |
| Udonis – Free Fire | https://www.blog.udonis.co/mobile-marketing/mobile-games/free-fire-player-count |
| SQ Magazine | https://sqmagazine.co.uk/mobile-games-statistics/ |
| TechRT | https://techrt.com/mobile-game-spending-statistics/ |
| Accio | https://www.accio.com/business/most-profitable-mobile-games-2025-trend |
| Market.us | https://scoop.market.us/gaming-monetization-statistics/ |
| Quantumrun Mobile | https://www.quantumrun.com/consulting/mobile-game-statistics/ |

### Esports & Kompetisi
| Sumber | URL |
|---|---|
| Liquipedia MLBB | https://liquipedia.net/mobilelegends/Portal:Statistics |
| Liquipedia FF | https://liquipedia.net/freefire/Portal:Statistics |
| Esports Charts | https://escharts.com/games/free-fire |
| Wikipedia MSC 2025 | https://en.wikipedia.org/wiki/MSC_2025 |
| IconEra Stats | https://icon-era.com/statistics/video-game-statistics/ |

### Komunitas & Lainnya
| Sumber | URL |
|---|---|
| Statista Genshin | https://www.statista.com/statistics/1295196/genshin-impact-arpu-country/ |
| iGitems | https://igitems.com/freefire/charts |
| EpicNPC | https://www.epicnpc.com/forums/mobile-legends-bang-bang-mlbb-top-up.3467/ |

---

## 🗄️ Schema Database

```
dim_game              fact_pricing          dim_age
─────────────         ─────────────         ────────────
game_key (PK)         id (PK)               id (PK)
game                  game_key              game_key
publisher             tier_label            age_band
genre                 price_official_usd    pct
mau_millions          price_3rdparty_usd    game_median_age
revenue_2024_usd      value_score           age_price_sensitivity
arpdau_usd            spend_tier
competition_score     price_elasticity      fact_competition
community_health_...                        ───────────────────
                      fact_revenue_monthly  game_key (PK)
ml_features           ────────────────────  competition_score
───────────           game_key              competition_tier
game_key              year                  prize_pool_2025_usd
region                month_num             peak_viewers
spending_segment      revenue_usd           tournaments_total
...14 features...     seasonal_multiplier
optimal_price_per_... 
                      etl_audit_log
                      ─────────────
                      table_name
                      rows_loaded
                      checksum (MD5)
                      status
                      loaded_at
```

---

## 🤖 Hasil Model ML

| Model | R² | MAE (USD) | CV R² (5-fold) |
|---|---|---|---|
| **GradientBoosting** ⭐ | 0.9339 | $0.448 | **0.9768** |
| XGBoost | 0.9272 | $0.436 | 0.9677 |
| RandomForest | ~0.90 | ~$0.55 | ~0.95 |
| Ridge (baseline) | ~0.65 | ~$0.90 | ~0.64 |

### Inference Contoh

```python
import joblib
from src.ml.model import predict_price

model = joblib.load("outputs/ml_model.joblib")
result = predict_price(
    model=model,
    genre="MOBA",
    competition_score=80,
    mau_millions=50,
    arpdau_usd=0.025,
    age_median=22,
    region="SEA",
    spending_segment="dolphin"
)
print(f"Harga optimal: ${result['predicted_price_per_100_usd']}")
print(f"Range: ${result['range_low_usd']} – ${result['range_high_usd']}")
print(f"IDR: Rp{result['predicted_price_per_100_idr']:,}")
```

---

## 🛠️ Tools & Libraries

| Kategori | Tools |
|---|---|
| **Orchestration** | Apache Airflow 2.x |
| **Web Scraping** | requests, BeautifulSoup4, lxml, fake-useragent, tenacity |
| **Data Processing** | pandas, numpy |
| **Database** | SQLite (dev), Aiven PostgreSQL (prod), SQLAlchemy |
| **Warehouse** | Apache Parquet (via pyarrow) |
| **ML** | scikit-learn, XGBoost, joblib |
| **Visualisasi** | matplotlib, seaborn, plotly |
| **Logging** | loguru |
| **Testing** | pytest, pytest-cov |
| **Config** | python-dotenv |

---

## 📊 Output yang Dihasilkan

```
data/processed/
  ├── dim_game.csv              (6 rows × 22 cols)
  ├── fact_pricing.csv          (51 rows × 19 cols)
  ├── dim_age.csv               (24 rows × 7 cols)
  ├── fact_competition.csv      (6 rows × 9 cols)
  ├── fact_revenue_monthly.csv  (36 rows × 9 cols)
  └── ml_features.csv           (90 rows × 22 cols)

data/warehouse/
  ├── *.parquet (6 file Parquet)
  └── game_price_etl.db (SQLite)

outputs/charts/
  ├── heatmap_feature_correlation.png  ← Korelasi antar fitur
  ├── heatmap_price_by_game_region.png ← Harga per game × region
  ├── ml_feature_importance.png
  ├── ml_actual_vs_predicted.png
  └── ml_model_comparison.png
```

---

## 👥 Kontributor

| Nama | Role |
|---|---|
| [Nama Mahasiswa] | Lead Developer — ETL Pipeline |
| [Nama Anggota 2] | Data Engineer — Transform & BI |
| [Nama Anggota 3] | ML Engineer — Model Training |

---

## 📄 Lisensi

MIT License — bebas digunakan untuk keperluan akademik dan non-komersial.

---

*Game Price ETL Pipeline v2.0 | Data real dari 28 sumber publik*
