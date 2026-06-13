"""
src/load/loader.py
==================
Phase 3 — LOAD
Memuat data ke:
  - Database SQLite lokal (fallback) / PostgreSQL Aiven via DATABASE_URL
  - Parquet warehouse (kolumnar analytics)
  - CSV sudah dibuat di Transform phase

Menggunakan SQLAlchemy engine untuk PostgreSQL (diperlukan oleh pandas to_sql)
dan sqlite3 connection untuk SQLite lokal.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column, DateTime, Float, Integer, MetaData,
    String, Table, Text, create_engine, text,
)

from config.settings import DB_URL, PROCESSED_DIR, WAREHOUSE_DIR


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════

def _get_connection():
    """
    Buat koneksi database sesuai DATABASE_URL.
    SQLite  → sqlite3 langsung (kompatibel pandas)
    Postgres → SQLAlchemy engine (diperlukan pandas to_sql)
    """
    parsed = urlparse(DB_URL)
    if parsed.scheme.startswith("sqlite"):
        # Ekstrak path dari URI sqlite:///path/to/db
        db_path = DB_URL.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"DB: SQLite → {db_path}")
        return sqlite3.connect(db_path), "sqlite"
    else:
        try:
            engine = create_engine(DB_URL)
            # Test koneksi
            with engine.connect() as test_conn:
                test_conn.execute(text("SELECT 1"))
            logger.info(f"DB: PostgreSQL → {parsed.hostname}")
            return engine, "postgres"
        except Exception as exc:
            logger.warning(f"PostgreSQL gagal ({exc}), fallback SQLite")
            db_path = str(WAREHOUSE_DIR / "game_price_etl.db")
            return sqlite3.connect(db_path), "sqlite"


def _create_schema(conn, db_type: str) -> None:
    """Buat tabel-tabel database jika belum ada."""

    ddl_statements = [
        """CREATE TABLE IF NOT EXISTS dim_game (
            game_key TEXT PRIMARY KEY,
            game TEXT, publisher TEXT, genre TEXT, release_year INTEGER,
            platforms TEXT, mau_millions REAL, dau_millions REAL,
            dau_to_mau_ratio REAL, total_downloads_billions REAL,
            revenue_2024_usd INTEGER, arpdau_usd REAL, arpu_monthly_usd REAL,
            revenue_per_mau_usd REAL, session_minutes INTEGER,
            competition_score INTEGER, android_share_pct INTEGER,
            community_health_score REAL, revenue_tier TEXT,
            age_median INTEGER, top_countries TEXT, loaded_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS fact_pricing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_key TEXT, game TEXT, publisher TEXT, currency_type TEXT,
            tier_label TEXT, base_currency INTEGER, bonus_currency INTEGER,
            total_currency INTEGER, price_official_usd REAL, price_official_idr INTEGER,
            price_3rdparty_usd REAL, price_3rdparty_idr INTEGER, discount_avg_pct REAL,
            price_per_unit_usd REAL, price_per_100_usd REAL, price_per_100_idr INTEGER,
            value_score REAL, spend_tier TEXT, price_elasticity REAL, loaded_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS dim_age (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_key TEXT, game TEXT, age_band TEXT, pct INTEGER,
            game_median_age INTEGER, age_price_sensitivity TEXT, source_url TEXT, loaded_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS fact_competition (
            game_key TEXT PRIMARY KEY,
            competition_score INTEGER, competition_tier TEXT, prize_pool_2025_usd INTEGER,
            prize_pool_log REAL, tournaments_total INTEGER, peak_viewers INTEGER,
            peak_viewers_log REAL, regional_leagues INTEGER, loaded_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS fact_revenue_monthly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_key TEXT, year INTEGER, month_num INTEGER, month_name TEXT,
            revenue_usd_millions REAL, revenue_usd INTEGER, revenue_idr INTEGER,
            seasonal_multiplier REAL, loaded_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS ml_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_key TEXT, game TEXT, genre TEXT, region TEXT, spending_segment TEXT,
            mau_millions REAL, dau_millions REAL, arpdau_usd REAL,
            revenue_2024_usd INTEGER, competition_score REAL, community_health_score REAL,
            game_median_age REAL, dau_to_mau_ratio REAL, region_price_index REAL,
            region_arpu_mult REAL, segment_spend_mult REAL, segment_player_pct INTEGER,
            genre_encoded INTEGER, region_encoded INTEGER, segment_encoded INTEGER,
            optimal_price_per_100_usd REAL, optimal_price_per_100_idr INTEGER, loaded_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS etl_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT, rows_loaded INTEGER, checksum TEXT,
            status TEXT, loaded_at TEXT, note TEXT
        )""",
    ]

    if db_type == "postgres":
        # Gunakan SQLAlchemy engine untuk PostgreSQL
        with conn.connect() as pg_conn:
            for ddl in ddl_statements:
                ddl = ddl.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
                ddl = ddl.replace("TEXT PRIMARY KEY", "VARCHAR(50) PRIMARY KEY")
                pg_conn.execute(text(ddl))
            pg_conn.commit()
    else:
        cur = conn.cursor()
        for ddl in ddl_statements:
            cur.execute(ddl)
        conn.commit()

    logger.success("Schema database OK")


# ═══════════════════════════════════════════════════════════════════════════
# LOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_dataframe(df: pd.DataFrame, table_name: str, conn, db_type: str) -> int:
    """
    Muat DataFrame ke tabel — hapus data lama (idempotent/replace).
    Catat audit log dan checksum integritas.
    """
    if df is None or len(df) == 0:
        logger.warning(f"  [{table_name}] DataFrame kosong — skip")
        return 0

    df = df.copy()
    df["loaded_at"] = datetime.utcnow().isoformat()

    # Checksum untuk validasi integritas
    checksum = hashlib.md5(
        pd.util.hash_pandas_object(df.drop(columns=["loaded_at"]), index=True)
        .values.tobytes()
    ).hexdigest()

    try:
        # if_exists="replace" otomatis DROP+CREATE+INSERT (idempotent)
        # conn sudah berupa SQLAlchemy engine (postgres) atau sqlite3 connection
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        row_count = len(df)
        logger.success(f"  [{table_name}] {row_count} baris dimuat ke DB ✓")
        _write_audit(conn, db_type, table_name, row_count, checksum, "SUCCESS")
        return row_count

    except Exception as exc:
        logger.error(f"  [{table_name}] Load gagal: {exc}")
        _write_audit(conn, db_type, table_name, 0, "", "FAILED", str(exc))
        return 0


def _write_audit(conn, db_type: str, table_name: str, rows: int, checksum: str,
                 status: str, note: str = "") -> None:
    """Tulis audit log."""
    try:
        if db_type == "postgres":
            with conn.connect() as pg_conn:
                pg_conn.execute(
                    text("INSERT INTO etl_audit_log (table_name, rows_loaded, checksum, status, loaded_at, note) "
                         "VALUES (:tbl, :rows, :chk, :st, :la, :nt)"),
                    {"tbl": table_name, "rows": rows, "chk": checksum,
                     "st": status, "la": datetime.utcnow().isoformat(), "nt": note}
                )
                pg_conn.commit()
        else:
            conn.cursor().execute(
                "INSERT INTO etl_audit_log (table_name, rows_loaded, checksum, status, loaded_at, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (table_name, rows, checksum, status, datetime.utcnow().isoformat(), note)
            )
            conn.commit()
    except Exception:
        pass


def _validate_integrity(conn, db_type: str) -> dict[str, int]:
    """Verifikasi jumlah baris setiap tabel dan referential integrity."""
    results: dict[str, int] = {}
    tables = ["dim_game", "fact_pricing", "dim_age",
              "fact_competition", "fact_revenue_monthly", "ml_features"]

    if db_type == "postgres":
        with conn.connect() as pg_conn:
            for tbl in tables:
                try:
                    result = pg_conn.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
                    count = result.fetchone()[0]
                    results[tbl] = count
                    status = "✓" if count > 0 else "⚠ KOSONG"
                    logger.info(f"  Integrity [{tbl}]: {count} rows {status}")
                except Exception as exc:
                    logger.error(f"  Integrity [{tbl}]: {exc}")
                    results[tbl] = -1

            # Referential integrity check
            try:
                result = pg_conn.execute(text("""
                    SELECT COUNT(*) FROM fact_pricing
                    WHERE game_key NOT IN (SELECT game_key FROM dim_game)
                """))
                orphans = result.fetchone()[0]
                if orphans:
                    logger.warning(f"  ⚠ {orphans} orphan rows di fact_pricing")
                else:
                    logger.success("  ✓ Referential integrity OK")
            except Exception:
                pass
    else:
        cur = conn.cursor()
        for tbl in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                count = cur.fetchone()[0]
                results[tbl] = count
                status = "✓" if count > 0 else "⚠ KOSONG"
                logger.info(f"  Integrity [{tbl}]: {count} rows {status}")
            except Exception as exc:
                logger.error(f"  Integrity [{tbl}]: {exc}")
                results[tbl] = -1

        # Referential integrity check
        try:
            cur.execute("""
                SELECT COUNT(*) FROM fact_pricing
                WHERE game_key NOT IN (SELECT game_key FROM dim_game)
            """)
            orphans = cur.fetchone()[0]
            if orphans:
                logger.warning(f"  ⚠ {orphans} orphan rows di fact_pricing")
            else:
                logger.success("  ✓ Referential integrity OK")
        except Exception:
            pass

    return results


def _save_parquet(df: pd.DataFrame, name: str) -> None:
    """Simpan ke format Parquet untuk analytics warehouse."""
    out = WAREHOUSE_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    size_kb = out.stat().st_size // 1024
    logger.success(f"  Parquet → {out.name} ({size_kb} KB)")


# ═══════════════════════════════════════════════════════════════════════════
# MASTER LOAD RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_load(dataframes: dict[str, pd.DataFrame]) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 3 — LOAD")
    logger.info("=" * 60)

    conn, db_type = _get_connection()
    _create_schema(conn, db_type)

    load_map = {
        "dim_game":             dataframes.get("game"),
        "fact_pricing":         dataframes.get("pricing"),
        "dim_age":              dataframes.get("age"),
        "fact_competition":     dataframes.get("comp"),
        "fact_revenue_monthly": dataframes.get("revenue"),
        "ml_features":          dataframes.get("ml"),
    }

    total_rows = 0
    for table_name, df in load_map.items():
        if df is not None:
            rows = _load_dataframe(df, table_name, conn, db_type)
            total_rows += rows
            _save_parquet(df, table_name)

    integrity = _validate_integrity(conn, db_type)

    # Close connection / dispose engine
    if db_type == "postgres":
        conn.dispose()
    else:
        conn.close()

    summary = {
        "loaded_at":  datetime.utcnow().isoformat(),
        "db_type":    db_type,
        "db_url":     DB_URL[:40] + "...",
        "total_rows": total_rows,
        "tables":     integrity,
        "parquet_dir":str(WAREHOUSE_DIR),
    }

    out = WAREHOUSE_DIR / "load_summary.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.success(f"Load selesai — {total_rows} total baris dimuat ke {db_type.upper()}")
    return summary


if __name__ == "__main__":
    dfs = {
        alias: pd.read_csv(PROCESSED_DIR / f"{csv}.csv")
        for alias, csv in [
            ("game","dim_game"), ("pricing","fact_pricing"), ("age","dim_age"),
            ("comp","fact_competition"), ("revenue","fact_revenue_monthly"), ("ml","ml_features"),
        ]
        if (PROCESSED_DIR / f"{csv}.csv").exists()
    }
    run_load(dfs)
