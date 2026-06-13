"""
streamlit_app/bootstrap_data.py
================================
Auto-generate minimal dataset untuk Streamlit Cloud deployment.

Ketika di-deploy di Streamlit Cloud, data/processed/ tidak ada (karena .gitignore).
Modul ini menghasilkan dataset dari verified data yang ada di extractor.py
tanpa perlu menjalankan pipeline lengkap (tanpa scraping, tanpa Airflow).

Alur:
  1. Cek apakah CSV sudah ada → jika iya, skip
  2. Jalankan transform langsung dari VERIFIED_DATA (embedded)
  3. Simpan ke data/processed/
"""

import sys
from pathlib import Path

# Pastikan root project di sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger


def data_exists() -> bool:
    """Cek apakah dataset minimum sudah tersedia."""
    from config.settings import PROCESSED_DIR
    required_files = [
        "dim_game.csv",
        "fact_pricing.csv",
        "dim_age.csv",
        "fact_competition.csv",
        "fact_revenue_monthly.csv",
        "ml_features.csv",
    ]
    return all((PROCESSED_DIR / f).exists() for f in required_files)


def bootstrap() -> None:
    """
    Generate dataset dari verified data (tanpa scraping/internet).
    Digunakan saat deploy di Streamlit Cloud dimana pipeline belum dijalankan.
    """
    if data_exists():
        logger.info("✅ Dataset sudah ada, skip bootstrap")
        return

    logger.info("📦 Bootstrap: Generating dataset dari verified data...")

    from config.settings import RAW_DIR, PROCESSED_DIR

    # ── Step 1: Generate raw data dari verified constants ───────────────
    from src.extract.extractor import (
        VERIFIED_TOPUP,
        VERIFIED_GAME_STATS,
        VERIFIED_ESPORTS,
        VERIFIED_MARKET,
    )
    import json
    from datetime import datetime

    master_raw = {
        "topup": VERIFIED_TOPUP,
        "stats": VERIFIED_GAME_STATS,
        "esports": VERIFIED_ESPORTS,
        "market": VERIFIED_MARKET,
        "_pipeline_meta": {
            "phase": "bootstrap",
            "timestamp": datetime.now().isoformat(),
            "note": "Auto-generated for Streamlit Cloud deployment",
        },
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "master_raw.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(master_raw, fh, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Raw data saved → {raw_path}")

    # ── Step 2: Transform ──────────────────────────────────────────────
    from src.transform.transformer import (
        transform_topup,
        transform_game_stats,
        transform_age,
        transform_competition,
        transform_monthly_revenue,
        build_ml_dataset,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_pricing = transform_topup(master_raw["topup"])
    df_game = transform_game_stats(master_raw["stats"])
    df_age = transform_age(master_raw["stats"])
    df_comp = transform_competition(master_raw["esports"])
    df_revenue = transform_monthly_revenue()
    df_ml = build_ml_dataset(df_game, df_comp, df_age)

    logger.success("✅ Bootstrap selesai — dataset siap digunakan dashboard")
    logger.info(f"   dim_game:             {len(df_game)} rows")
    logger.info(f"   fact_pricing:         {len(df_pricing)} rows")
    logger.info(f"   dim_age:              {len(df_age)} rows")
    logger.info(f"   fact_competition:     {len(df_comp)} rows")
    logger.info(f"   fact_revenue_monthly: {len(df_revenue)} rows")
    logger.info(f"   ml_features:          {len(df_ml)} rows")


if __name__ == "__main__":
    bootstrap()
