"""
config/settings.py
==================
Konfigurasi terpusat — semua nilai dapat di-override via environment variable.
Tidak ada hardcoding di source code; nilai default hanya sebagai fallback.

Mendukung 3 sumber konfigurasi (prioritas tertinggi di atas):
  1. Streamlit secrets  (.streamlit/secrets.toml) — untuk Streamlit Cloud
  2. Environment variable (dari .env / shell)
  3. Default hardcoded (fallback)

Cara penggunaan:
    from config.settings import DB_URL, IDR_PER_USD, RAW_DIR

Environment variables yang didukung:
    DATABASE_URL    : PostgreSQL connection string (Aiven/lokal)
    IDR_PER_USD     : Kurs rupiah terhadap USD
    LOG_LEVEL       : DEBUG | INFO | WARNING | ERROR
    RAW_DATA_DIR    : Path direktori data mentah
    PROCESSED_DIR   : Path direktori data terproses
    WAREHOUSE_DIR   : Path direktori warehouse / Parquet
    OUTPUTS_DIR     : Path direktori output (chart, report)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Root project -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ── Helper: ambil config dari st.secrets → env → default -------------------
def _get_config(key: str, default: str = "") -> str:
    """
    Ambil nilai konfigurasi dengan prioritas:
      1. Streamlit secrets (jika tersedia)
      2. Environment variable
      3. Default value
    """
    # Coba Streamlit secrets terlebih dahulu
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    # Fallback ke environment variable
    return os.getenv(key, default)


# ── Database (Kriteria 4: Load ke target DB) --------------------------------
# Gunakan Aiven PostgreSQL bila tersedia, fallback ke SQLite lokal
DB_URL: str = _get_config(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'data' / 'warehouse' / 'game_price_etl.db'}"
)

# ── Kurs & Konstanta Bisnis -------------------------------------------------
IDR_PER_USD: float = float(_get_config("IDR_PER_USD", "16250"))
EXTRACT_TIMEOUT: int = int(_get_config("EXTRACT_TIMEOUT", "15"))  # detik
MAX_RETRIES: int = int(_get_config("MAX_RETRIES", "3"))

# ── Direktori Data ----------------------------------------------------------
RAW_DIR       = Path(_get_config("RAW_DATA_DIR",  str(BASE_DIR / "data" / "raw")))
PROCESSED_DIR = Path(_get_config("PROCESSED_DIR", str(BASE_DIR / "data" / "processed")))
WAREHOUSE_DIR = Path(_get_config("WAREHOUSE_DIR", str(BASE_DIR / "data" / "warehouse")))
OUTPUTS_DIR   = Path(_get_config("OUTPUTS_DIR",   str(BASE_DIR / "outputs")))
CHARTS_DIR    = OUTPUTS_DIR / "charts"
REPORTS_DIR   = OUTPUTS_DIR / "reports"
LOGS_DIR      = BASE_DIR / "logs"

# ── Pastikan direktori ada --------------------------------------------------
for _dir in [RAW_DIR, PROCESSED_DIR, WAREHOUSE_DIR, OUTPUTS_DIR, CHARTS_DIR, REPORTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Logging ----------------------------------------------------------------
LOG_LEVEL: str = _get_config("LOG_LEVEL", "INFO")

# ── Daftar game yang diproses (mudah diperluas tanpa ubah kode lain) ---------
GAME_KEYS: list[str] = [
    "mlbb",
    "free_fire",
    "pubg_mobile",
    "genshin_impact",
    "honkai_star_rail",
    "cod_mobile",
]

# ── Mapping region ke indeks harga -----------------------------------------
REGION_PRICE_INDEX: dict[str, float] = {
    "SEA":         0.70,
    "LATAM":       0.75,
    "Global_West": 1.00,
    "East_Asia":   1.20,
    "MENA":        0.85,
}

# ── Spending segment multiplier --------------------------------------------
SEGMENT_MULTIPLIER: dict[str, float] = {
    "whale":   2.8,
    "dolphin": 1.4,
    "minnow":  0.75,
}

# ── Currency key per game (untuk flatten tier data) ------------------------
GAME_CURRENCY_KEY: dict[str, str] = {
    "mlbb":             "diamonds",
    "free_fire":        "diamonds",
    "pubg_mobile":      "uc",
    "genshin_impact":   "crystals",
    "honkai_star_rail": "shards",
    "cod_mobile":       "cp",
}
