"""
run_pipeline.py
===============
Master runner — jalankan full ETL pipeline tanpa Airflow.
Cocok untuk development, testing, atau demo lokal.

Usage:
    python run_pipeline.py                   # Full pipeline
    python run_pipeline.py --phase extract
    python run_pipeline.py --phase transform
    python run_pipeline.py --phase load
    python run_pipeline.py --phase ml
    python run_pipeline.py --phase all
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Pastikan root project ada di sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from config.settings import LOGS_DIR

# ── Setup logging ke file + console ───────────────────────────────────────
log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, level="INFO", rotation="10 MB")


def phase_extract() -> dict:
    from src.extract.extractor import run_extract
    return run_extract()


def phase_transform() -> dict:
    from src.transform.transformer import run_transform
    return run_transform()


def phase_load(dataframes: dict | None = None) -> dict:
    import pandas as pd
    from src.load.loader import run_load
    from config.settings import PROCESSED_DIR

    if dataframes is None:
        # Baca dari CSV yang sudah ada
        csv_map = {
            "game":    "dim_game",
            "pricing": "fact_pricing",
            "age":     "dim_age",
            "comp":    "fact_competition",
            "revenue": "fact_revenue_monthly",
            "ml":      "ml_features",
        }
        dataframes = {
            alias: pd.read_csv(PROCESSED_DIR / f"{name}.csv")
            for alias, name in csv_map.items()
            if (PROCESSED_DIR / f"{name}.csv").exists()
        }

    return run_load(dataframes)


def phase_ml():
    from src.ml.model import run_ml
    return run_ml()


def run_full_pipeline() -> None:
    """Jalankan semua phase secara berurutan dan catat durasi tiap fase."""
    logger.info("=" * 70)
    logger.info("  GAME PRICE ETL PIPELINE — Full Run")
    logger.info(f"  Mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    timings: dict[str, float] = {}

    # ── Phase 1: Extract ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n📥 PHASE 1 — EXTRACT")
    phase_extract()
    timings["extract"] = round(time.perf_counter() - t0, 2)

    # ── Phase 2: Transform ────────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n⚙️  PHASE 2 — TRANSFORM")
    dataframes = phase_transform()
    timings["transform"] = round(time.perf_counter() - t0, 2)

    # ── Phase 3: Load ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n💾 PHASE 3 — LOAD")
    phase_load(dataframes)
    timings["load"] = round(time.perf_counter() - t0, 2)

    # ── Phase 4: ML ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n🤖 PHASE 4 — MACHINE LEARNING")
    phase_ml()
    timings["ml"] = round(time.perf_counter() - t0, 2)

    total = sum(timings.values())
    logger.info("\n" + "=" * 70)
    logger.success("✅ PIPELINE SELESAI")
    logger.info("=" * 70)
    for phase, secs in timings.items():
        logger.info(f"   {phase.upper():<12}: {secs:.2f} detik")
    logger.info(f"   {'TOTAL':<12}: {total:.2f} detik")
    logger.info("=" * 70)
    logger.info("\nOutput files:")
    logger.info("  📁 data/raw/            → JSON mentah")
    logger.info("  📁 data/processed/      → CSV siap analisis")
    logger.info("  📁 data/warehouse/      → Parquet + SQLite DB")
    logger.info("  📊 outputs/charts/      → Chart PNG (heatmap, ML)")
    logger.info("  🤖 outputs/ml_model.joblib → Trained model")
    logger.info(f"  📋 Log: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Game Price ETL Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["extract", "transform", "load", "ml", "all"],
        default="all",
        help="Phase pipeline yang akan dijalankan (default: all)",
    )
    args = parser.parse_args()

    phase_dispatch = {
        "extract":   phase_extract,
        "transform": phase_transform,
        "load":      phase_load,
        "ml":        phase_ml,
        "all":       run_full_pipeline,
    }

    phase_dispatch[args.phase]()
