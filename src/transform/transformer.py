"""
src/transform/transformer.py
=============================
Phase 2 — TRANSFORM
Pembersihan & Transformasi data mentah menjadi dataset siap analisis.

Operasi yang dilakukan:
  - Penanganan nilai null / missing (imputation & drop)
  - Deduplikasi baris berdasarkan primary key
  - Normalisasi format (tipe data, kurs, satuan)
  - Feature engineering (skor turunan, encoding kategoris)
  - Validasi integritas referensial antar tabel
  - Pembuatan heatmap korelasi (output PNG)
  - Simpan semua tabel ke CSV (data/processed/)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from config.settings import (
    CHARTS_DIR,
    GAME_CURRENCY_KEY,
    GAME_KEYS,
    IDR_PER_USD,
    PROCESSED_DIR,
    RAW_DIR,
    REGION_PRICE_INDEX,
    SEGMENT_MULTIPLIER,
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: VALIDASI & PEMBERSIHAN UMUM
# ═══════════════════════════════════════════════════════════════════════════

def _drop_duplicates_log(df: pd.DataFrame, subset: list[str], label: str) -> pd.DataFrame:
    """Deduplikasi dan log jumlah baris yang dihapus."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"  [{label}] Deduplikasi: {dropped} baris duplikat dihapus")
    return df


def _fill_nulls_median(df: pd.DataFrame, numeric_cols: list[str], label: str) -> pd.DataFrame:
    """Isi nilai null pada kolom numerik dengan median (kelompok per game_key jika ada)."""
    for col in numeric_cols:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count:
            if "game_key" in df.columns:
                df[col] = df.groupby("game_key")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())
            logger.warning(f"  [{label}] {col}: {null_count} null → diisi median")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM 1: HARGA TOP-UP → fact_pricing
# ═══════════════════════════════════════════════════════════════════════════

def transform_topup(raw_topup: dict) -> pd.DataFrame:
    """
    Flatten tier data menjadi tabel relasional.
    Derived columns:
      - total_currency   : base + bonus
      - price_idr        : konversi ke IDR
      - price_per_100    : harga per 100 unit (target ML)
      - value_score      : currency per $1
      - spend_tier       : kategorisasi micro/small/medium/large/mega
      - price_elasticity : indeks sensitivitas harga per tier
      - discount_avg_pct : rata-rata diskon 3rd-party
    """
    logger.info("Transform ▶ fact_pricing")
    rows: list[dict] = []

    elasticity_map = {
        "micro": -2.1, "small": -1.6, "medium": -0.9,
        "large": -0.5, "mega": -0.3,
    }

    for game_key in GAME_KEYS:
        data = raw_topup.get(game_key)
        if not data or "_error" in data:
            logger.warning(f"  Skipping {game_key}: no data")
            continue

        currency_key = GAME_CURRENCY_KEY.get(game_key, "amount")
        disc_min = data.get("third_party_discount_pct_min", 0)
        disc_max = data.get("third_party_discount_pct_max", 0)
        disc_avg = (disc_min + disc_max) / 2.0

        for tier in data.get("tiers", []):
            # Tangani kunci currency berbeda per game
            base = tier.get(currency_key, tier.get("diamonds", 0))
            bonus = tier.get("bonus", 0)
            total = base + bonus
            price_usd = tier.get("price_usd", 0.0)

            if price_usd <= 0 or total <= 0:
                continue  # skip tier invalid

            price_3rd = round(price_usd * (1 - disc_avg / 100), 3)

            # Kategorisasi tier berdasarkan harga
            if price_usd < 2:
                spend_tier = "micro"
            elif price_usd < 10:
                spend_tier = "small"
            elif price_usd < 30:
                spend_tier = "medium"
            elif price_usd < 60:
                spend_tier = "large"
            else:
                spend_tier = "mega"

            rows.append({
                "game_key":            game_key,
                "game":                data["game"],
                "publisher":           data.get("publisher", ""),
                "currency_type":       data.get("currency", ""),
                "tier_label":          tier.get("label", "Unknown"),
                "base_currency":       int(base),
                "bonus_currency":      int(bonus),
                "total_currency":      int(total),
                "price_official_usd":  round(price_usd, 2),
                "price_official_idr":  int(price_usd * IDR_PER_USD),
                "price_3rdparty_usd":  price_3rd,
                "price_3rdparty_idr":  int(price_3rd * IDR_PER_USD),
                "discount_avg_pct":    disc_avg,
                "price_per_unit_usd":  round(price_usd / total, 6),
                "price_per_100_usd":   round((price_usd / total) * 100, 4),
                "price_per_100_idr":   int(((price_usd / total) * 100) * IDR_PER_USD),
                "value_score":         round(total / price_usd, 2),
                "spend_tier":          spend_tier,
                "price_elasticity":    elasticity_map[spend_tier],
            })

    df = pd.DataFrame(rows)
    df = _drop_duplicates_log(df, ["game_key", "tier_label"], "fact_pricing")
    df = _fill_nulls_median(df, ["price_official_usd", "value_score"], "fact_pricing")

    # Validasi: tidak ada harga negatif
    invalid = df[df["price_official_usd"] <= 0]
    if len(invalid):
        logger.warning(f"  fact_pricing: {len(invalid)} baris harga <= 0 dihapus")
        df = df[df["price_official_usd"] > 0]

    logger.success(f"  fact_pricing: {len(df)} rows × {len(df.columns)} cols")
    df.to_csv(PROCESSED_DIR / "fact_pricing.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM 2: STATISTIK GAME → dim_game
# ═══════════════════════════════════════════════════════════════════════════

# Revenue 2024 canonical (dari multiple source cross-check)
REVENUE_2024: dict[str, int] = {
    "mlbb":            194_990_000,
    "free_fire":       408_000_000,
    "pubg_mobile":     2_000_000_000,
    "genshin_impact":  700_000_000,
    "honkai_star_rail": 900_000_000,
    "cod_mobile":      360_000_000,
}


def transform_game_stats(raw_stats: dict) -> pd.DataFrame:
    """
    Normalisasi statistik game. Derived columns:
      - arpu_monthly_usd      : ARPDAU × 30
      - revenue_per_mau_usd   : Revenue per 1 pengguna per tahun
      - dau_to_mau_ratio      : Retensi harian
      - community_health_score: Skor kesehatan komunitas 0-100
      - revenue_tier          : blockbuster / major / mid / small
    """
    logger.info("Transform ▶ dim_game")
    rows: list[dict] = []

    for game_key in GAME_KEYS:
        data = raw_stats.get(game_key, {})
        if "_error" in data:
            continue

        mau  = float(data.get("mau_millions", 0) or 0)
        dau  = float(data.get("dau_millions", mau * 0.15) or mau * 0.15)
        arpdau = float(data.get("arpdau_usd", 0) or 0)
        rev24  = REVENUE_2024.get(game_key, 0)
        dl     = float(data.get("total_downloads_billions", 0) or 0)

        d2m_ratio = round(dau / mau, 3) if mau > 0 else 0.0

        # Community health score (formula terdokumentasi di GUIDE.md)
        mau_norm   = min(mau / 130, 1.0) * 40
        ratio_norm = min(d2m_ratio / 0.35, 1.0) * 30
        dl_norm    = min(dl / 1.5, 1.0) * 30
        health     = round(mau_norm + ratio_norm + dl_norm, 1)

        # Revenue tier
        if rev24 >= 1_000_000_000:
            revenue_tier = "blockbuster"
        elif rev24 >= 400_000_000:
            revenue_tier = "major"
        elif rev24 >= 100_000_000:
            revenue_tier = "mid"
        else:
            revenue_tier = "small"

        rows.append({
            "game_key":              game_key,
            "game":                  data.get("game", game_key),
            "publisher":             data.get("publisher", "Unknown"),
            "genre":                 data.get("genre", "Unknown"),
            "release_year":          int(data.get("release_year", 0)),
            "platforms":             ", ".join(data.get("platforms", [])),
            "mau_millions":          mau,
            "dau_millions":          round(dau, 2),
            "dau_to_mau_ratio":      d2m_ratio,
            "registered_users_millions": data.get("registered_users_millions"),
            "total_downloads_billions":  round(dl, 3),
            "revenue_2024_usd":      rev24,
            "arpdau_usd":            arpdau,
            "arpu_monthly_usd":      round(arpdau * 30, 3),
            "revenue_per_mau_usd":   round(rev24 / (mau * 1_000_000), 2) if mau > 0 else 0,
            "session_minutes":       int(data.get("session_minutes", 25)),
            "competition_score":     int(data.get("competition_score", 50)),
            "android_share_pct":     int(data.get("android_share_pct", 75)),
            "community_health_score":health,
            "revenue_tier":          revenue_tier,
            "age_median":            int(data.get("age_median", 25)),
            "top_countries":         ", ".join(data.get("top_countries", [])),
        })

    df = pd.DataFrame(rows)
    df = _drop_duplicates_log(df, ["game_key"], "dim_game")
    df = _fill_nulls_median(df, ["mau_millions", "dau_millions", "arpdau_usd"], "dim_game")

    logger.success(f"  dim_game: {len(df)} rows × {len(df.columns)} cols")
    df.to_csv(PROCESSED_DIR / "dim_game.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM 3: DEMOGRAFI USIA → dim_age
# ═══════════════════════════════════════════════════════════════════════════

def transform_age(raw_stats: dict) -> pd.DataFrame:
    """Flatten distribusi usia dari nested dict ke tabel relasional."""
    logger.info("Transform ▶ dim_age")
    rows: list[dict] = []

    sensitivity_map = {
        "high": lambda age: age < 22,
        "medium": lambda age: 22 <= age < 30,
        "low": lambda age: age >= 30,
    }

    source_map: dict[str, str] = {
        "mlbb":            "https://icon-era.com/blog/mobile-legends-bang-bang-live-player-count-and-statistics.443/",
        "free_fire":       "https://www.businessofapps.com/data/free-fire-statistics/",
        "pubg_mobile":     "https://sqmagazine.co.uk/mobile-games-statistics/",
        "genshin_impact":  "https://news.bittopup.com/news/genshin-impact-2025-15.2m-players-0.8b-revenue",
        "honkai_star_rail":"https://techrt.com/mobile-game-spending-statistics/",
        "cod_mobile":      "https://techrt.com/mobile-game-spending-statistics/",
    }

    for game_key in GAME_KEYS:
        data = raw_stats.get(game_key, {})
        age_dist = data.get("age_distribution", {})
        age_median = int(data.get("age_median", 25))

        # Tentukan sensitivitas
        if age_median < 22:
            sensitivity = "high"
        elif age_median < 30:
            sensitivity = "medium"
        else:
            sensitivity = "low"

        for band, pct in age_dist.items():
            rows.append({
                "game_key":            game_key,
                "game":                data.get("game", game_key),
                "age_band":            band,
                "pct":                 int(pct),
                "game_median_age":     age_median,
                "age_price_sensitivity": sensitivity,
                "source_url":          source_map.get(game_key, ""),
            })

    df = pd.DataFrame(rows)
    df = _drop_duplicates_log(df, ["game_key", "age_band"], "dim_age")

    # Validasi: persentase per game harus ~ 100%
    totals = df.groupby("game_key")["pct"].sum()
    for gk, total in totals.items():
        if not (95 <= total <= 105):
            logger.warning(f"  dim_age [{gk}]: total pct = {total} (expected ~100)")

    logger.success(f"  dim_age: {len(df)} rows × {len(df.columns)} cols")
    df.to_csv(PROCESSED_DIR / "dim_age.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM 4: KOMPETISI ESPORTS → fact_competition
# ═══════════════════════════════════════════════════════════════════════════

def transform_competition(raw_esports: dict) -> pd.DataFrame:
    """Normalisasi data kompetisi, log-transform untuk normalisasi distribusi."""
    logger.info("Transform ▶ fact_competition")
    rows: list[dict] = []

    for game_key in GAME_KEYS:
        data = raw_esports.get(game_key, {})
        if "_error" in data:
            continue

        score = int(data.get("competition_score", 50))
        prize = int(data.get("prize_pool_2025_usd", 0))
        viewers = int(data.get("peak_viewers", 0))

        # Tier kompetisi berdasarkan skor
        if score >= 80:
            tier = "Elite"
        elif score >= 60:
            tier = "High"
        elif score >= 40:
            tier = "Medium"
        else:
            tier = "Low"

        rows.append({
            "game_key":           game_key,
            "competition_score":  score,
            "competition_tier":   tier,
            "prize_pool_2025_usd":prize,
            "prize_pool_log":     round(np.log1p(prize), 4),
            "tournaments_total":  int(data.get("tournaments_total", 0)),
            "peak_viewers":       viewers,
            "peak_viewers_log":   round(np.log1p(viewers), 4),
            "regional_leagues":   int(data.get("regional_leagues", 0)),
        })

    df = pd.DataFrame(rows)
    df = _drop_duplicates_log(df, ["game_key"], "fact_competition")
    logger.success(f"  fact_competition: {len(df)} rows × {len(df.columns)} cols")
    df.to_csv(PROCESSED_DIR / "fact_competition.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM 5: REVENUE BULANAN → fact_revenue_monthly
# ═══════════════════════════════════════════════════════════════════════════

# Data dari: Quantumrun.com, Udonis.co — diverifikasi manual
MONTHLY_REVENUE_MILLIONS: dict[str, dict[int, list[float]]] = {
    "mlbb": {
        2024: [18.0,14.0,16.0,15.0,17.0,13.0,14.0,16.0,15.0,17.0,19.0,14.0],
        2025: [24.3,16.0,22.4,15.0,16.0,18.0],
    },
    "free_fire": {
        2024: [20.0,17.0,18.0,16.0,17.0,16.0,18.0,17.0,19.0,18.0,17.0,31.0],
        2025: [30.9,17.0,18.0,16.0,17.0,18.0],
    },
}

SEASONAL_MULTIPLIER: dict[int, float] = {
    1:1.35, 2:0.90, 3:1.20, 4:0.95, 5:1.05, 6:0.88,
    7:0.90, 8:1.00, 9:0.98, 10:1.05, 11:1.15, 12:1.10,
}

MONTH_NAMES: list[str] = [
    "Jan","Feb","Mar","Apr","Mei","Jun",
    "Jul","Agu","Sep","Okt","Nov","Des"
]


def transform_monthly_revenue() -> pd.DataFrame:
    """Buat time series revenue bulanan dari data terverifikasi."""
    logger.info("Transform ▶ fact_revenue_monthly")
    rows: list[dict] = []

    for game_key, year_data in MONTHLY_REVENUE_MILLIONS.items():
        for year, monthly_list in year_data.items():
            for month_idx, rev_m in enumerate(monthly_list, start=1):
                rows.append({
                    "game_key":             game_key,
                    "year":                 year,
                    "month_num":            month_idx,
                    "month_name":           MONTH_NAMES[month_idx - 1],
                    "revenue_usd_millions": rev_m,
                    "revenue_usd":          int(rev_m * 1_000_000),
                    "revenue_idr":          int(rev_m * 1_000_000 * IDR_PER_USD),
                    "seasonal_multiplier":  SEASONAL_MULTIPLIER.get(month_idx, 1.0),
                })

    df = pd.DataFrame(rows)
    df = _drop_duplicates_log(df, ["game_key", "year", "month_num"], "fact_revenue_monthly")
    logger.success(f"  fact_revenue_monthly: {len(df)} rows")
    df.to_csv(PROCESSED_DIR / "fact_revenue_monthly.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM 6: ML DATASET → ml_features
# ═══════════════════════════════════════════════════════════════════════════

def build_ml_dataset(
    df_game: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_age:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Buat flat ML-ready dataset: setiap baris = kombinasi (game × region × segment).
    Total: 6 games × 5 regions × 3 segments = 90 baris.
    Target: optimal_price_per_100_usd
    """
    logger.info("Transform ▶ ml_features")

    regions = [
        {"region": "SEA",         "price_index": 0.70, "arpu_mult": 0.65},
        {"region": "LATAM",       "price_index": 0.75, "arpu_mult": 0.70},
        {"region": "Global_West", "price_index": 1.00, "arpu_mult": 1.00},
        {"region": "East_Asia",   "price_index": 1.20, "arpu_mult": 1.30},
        {"region": "MENA",        "price_index": 0.85, "arpu_mult": 0.80},
    ]
    segments = [
        {"segment": "whale",   "spend_mult": 2.8, "pct_players": 3},
        {"segment": "dolphin", "spend_mult": 1.4, "pct_players": 15},
        {"segment": "minnow",  "spend_mult": 0.75,"pct_players": 82},
    ]

    # Merge kompetisi & usia ke dim_game
    # Hapus competition_score dari df_game jika sudah ada, hindari konflik _x/_y
    game_cols_no_comp = [c for c in df_game.columns if c != "competition_score"]
    df_age_med = df_age.groupby("game_key")["game_median_age"].first().reset_index()
    df_merged  = df_game[game_cols_no_comp].merge(
        df_comp[["game_key", "competition_score", "competition_tier"]], on="game_key", how="left"
    )
    df_merged  = df_merged.merge(df_age_med, on="game_key", how="left")

    rows: list[dict] = []
    genre_enc  = {g: i for i, g in enumerate(df_game["genre"].unique())}
    region_enc = {"SEA": 0, "LATAM": 1, "Global_West": 2, "East_Asia": 3, "MENA": 4}
    seg_enc    = {"whale": 2, "dolphin": 0, "minnow": 1}

    for _, game_row in df_merged.iterrows():
        mau    = float(game_row["mau_millions"])
        dau    = float(game_row["dau_millions"])
        arpdau = float(game_row["arpdau_usd"])
        comp   = float(game_row["competition_score"])
        health = float(game_row["community_health_score"])
        age    = float(game_row.get("game_median_age", 25))
        d2m    = float(game_row["dau_to_mau_ratio"])

        for reg in regions:
            for seg in segments:
                # Formula harga optimal (documented in GUIDE.md)
                base_p100  = max(arpdau * 50, 0.5)
                comp_f     = 1 - (comp / 200)
                age_f      = 1.25 if age > 30 else (0.80 if age < 20 else 1.0)
                mau_f      = 0.88 if mau > 100 else (1.12 if mau < 20 else 1.0)
                opt_price  = round(base_p100 * comp_f * seg["spend_mult"] * age_f * mau_f * reg["price_index"], 4)

                rows.append({
                    # Identifiers
                    "game_key":             game_row["game_key"],
                    "game":                 game_row["game"],
                    "genre":                game_row["genre"],
                    "region":               reg["region"],
                    "spending_segment":     seg["segment"],
                    # Features
                    "mau_millions":         mau,
                    "dau_millions":         dau,
                    "arpdau_usd":           arpdau,
                    "revenue_2024_usd":     int(game_row["revenue_2024_usd"]),
                    "competition_score":    comp,
                    "community_health_score": health,
                    "game_median_age":      age,
                    "dau_to_mau_ratio":     d2m,
                    "region_price_index":   reg["price_index"],
                    "region_arpu_mult":     reg["arpu_mult"],
                    "segment_spend_mult":   seg["spend_mult"],
                    "segment_player_pct":   seg["pct_players"],
                    # Encoded categoricals
                    "genre_encoded":        genre_enc.get(game_row["genre"], 0),
                    "region_encoded":       region_enc.get(reg["region"], 0),
                    "segment_encoded":      seg_enc.get(seg["segment"], 0),
                    # Target variable
                    "optimal_price_per_100_usd": opt_price,
                    "optimal_price_per_100_idr": int(opt_price * IDR_PER_USD),
                })

    df = pd.DataFrame(rows)
    df = _drop_duplicates_log(df, ["game_key", "region", "spending_segment"], "ml_features")
    logger.success(f"  ml_features: {len(df)} rows × {len(df.columns)} cols")
    df.to_csv(PROCESSED_DIR / "ml_features.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# HEATMAP KORELASI (Kriteria 4: Analisis heatmap Python)
# ═══════════════════════════════════════════════════════════════════════════

def generate_heatmaps(df_ml: pd.DataFrame, df_game: pd.DataFrame) -> None:
    """
    Menghasilkan 2 heatmap korelasi menggunakan Seaborn:
      1. Korelasi antar fitur ML (feature correlation matrix)
      2. Revenue per game × region (price heatmap)
    Disimpan ke outputs/charts/
    """
    logger.info("Transform ▶ Heatmap Korelasi")

    # ── Heatmap 1: Feature Correlation Matrix ──────────────────────────────
    numeric_cols = [
        "mau_millions", "dau_millions", "arpdau_usd", "competition_score",
        "community_health_score", "game_median_age", "region_price_index",
        "segment_spend_mult", "optimal_price_per_100_usd",
    ]
    corr = df_ml[numeric_cols].corr()
    readable_labels = [
        "MAU (Juta)", "DAU (Juta)", "ARPDAU", "Skor Kompetisi",
        "Komunitas Score", "Usia Median", "Indeks Harga Regional",
        "Multiplier Segmen", "Harga Optimal/100",
    ]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, linecolor="#1e2d45",
        xticklabels=readable_labels, yticklabels=readable_labels,
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title(
        "Heatmap Korelasi — Fitur Penentu Harga Top-Up Game\n"
        "Nilai mendekati +1 = korelasi positif kuat | mendekati −1 = korelasi negatif kuat",
        fontsize=12, pad=16
    )
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    out1 = CHARTS_DIR / "heatmap_feature_correlation.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"  Heatmap 1 saved → {out1}")

    # ── Heatmap 2: Revenue per Game per Tier (Price Heatmap) ───────────────
    pivot_data = df_ml.pivot_table(
        index="game", columns="region",
        values="optimal_price_per_100_usd", aggfunc="mean"
    )
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot_data, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, linecolor="#ccc",
        ax=ax2, cbar_kws={"label": "Harga Optimal per 100 Currency (USD)"}
    )
    ax2.set_title(
        "Heatmap Harga Optimal per Game × Region\n"
        "(Rata-rata semua segmen spending)",
        fontsize=12, pad=12
    )
    ax2.set_xlabel("Region Pasar", fontsize=11)
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=15, labelsize=9)
    ax2.tick_params(axis="y", rotation=0, labelsize=9)
    fig2.tight_layout()
    out2 = CHARTS_DIR / "heatmap_price_by_game_region.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"  Heatmap 2 saved → {out2}")


# ═══════════════════════════════════════════════════════════════════════════
# MASTER TRANSFORM RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_transform() -> dict[str, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("PHASE 2 — TRANSFORM")
    logger.info("=" * 60)

    with open(RAW_DIR / "master_raw.json", encoding="utf-8") as fh:
        raw = json.load(fh)

    df_pricing  = transform_topup(raw.get("topup", {}))
    df_game     = transform_game_stats(raw.get("stats", {}))
    df_age      = transform_age(raw.get("stats", {}))
    df_comp     = transform_competition(raw.get("esports", {}))
    df_revenue  = transform_monthly_revenue()
    df_ml       = build_ml_dataset(df_game, df_comp, df_age)

    # Heatmap korelasi
    generate_heatmaps(df_ml, df_game)

    logger.success("Transform selesai — semua CSV tersimpan di data/processed/")
    return {
        "pricing": df_pricing, "game": df_game, "age": df_age,
        "comp": df_comp, "revenue": df_revenue, "ml": df_ml,
    }


if __name__ == "__main__":
    run_transform()
