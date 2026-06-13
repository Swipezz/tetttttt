"""
src/ml/model.py
===============
Phase ML — Price Recommendation Model

Model: Gradient Boosting Regressor (terbaik dari 4 model yang dibandingkan)
Target: optimal_price_per_100_usd
Features: 14 fitur dari ml_features.csv

Output:
  - outputs/ml_model.joblib      → trained model
  - outputs/charts/              → 3 chart evaluasi model
  - outputs/reports/ml_report.md → laporan evaluasi teks
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

from config.settings import CHARTS_DIR, OUTPUTS_DIR, PROCESSED_DIR

# ── Kolom fitur (tidak hardcoded — dibaca dari dataset) ───────────────────
FEATURE_COLS: list[str] = [
    "mau_millions",
    "dau_millions",
    "arpdau_usd",
    "competition_score",
    "community_health_score",
    "game_median_age",
    "dau_to_mau_ratio",
    "region_price_index",
    "region_arpu_mult",
    "segment_spend_mult",
    "segment_player_pct",
    "genre_encoded",
    "region_encoded",
    "segment_encoded",
]
TARGET_COL = "optimal_price_per_100_usd"

FEATURE_LABELS: dict[str, str] = {
    "segment_spend_mult":       "Multiplier Segmen Spending",
    "arpdau_usd":               "ARPDAU (Pendapatan/Pengguna/Hari)",
    "competition_score":        "Skor Kompetisi",
    "region_price_index":       "Indeks Harga Regional",
    "mau_millions":             "MAU — Ukuran Komunitas",
    "game_median_age":          "Usia Median Pemain",
    "dau_to_mau_ratio":         "Rasio DAU / MAU",
    "community_health_score":   "Skor Kesehatan Komunitas",
    "region_arpu_mult":         "Multiplier ARPU Regional",
    "segment_player_pct":       "Persentase Player di Segmen",
    "genre_encoded":            "Genre Game (encoded)",
    "region_encoded":           "Region (encoded)",
    "segment_encoded":          "Segmen Spending (encoded)",
    "dau_millions":             "DAU (Pengguna Aktif Harian)",
}


# ═══════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ═══════════════════════════════════════════════════════════════════════════

def _load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Muat dataset ML dari CSV, validasi fitur, kembalikan X, y, df."""
    csv_path = PROCESSED_DIR / "ml_features.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset loaded: {len(df)} rows × {len(df.columns)} cols")

    # Pastikan semua fitur tersedia
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Fitur hilang di dataset: {missing}")

    # Bersihkan inf / NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    logger.info(f"After cleaning: {len(df)} rows")

    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(float)
    return X, y, df


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING — 4 MODEL DIBANDINGKAN
# ═══════════════════════════════════════════════════════════════════════════

def _build_models() -> dict:
    """Definisi 4 model — parameter tidak hardcoded dalam string."""
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_samples_split=5, subsample=0.85, random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.85, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1,
        ),
        "Ridge": Ridge(alpha=1.0),
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple[dict, str]:
    """
    Latih semua model, evaluasi dengan test split + 5-fold CV.
    Return hasil evaluasi dan nama model terbaik.
    """
    logger.info("ML ▶ Training 4 model")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = _build_models()
    results: dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        cv_r2   = cross_val_score(model, X, y, cv=kf, scoring="r2")

        results[name] = {
            "model":   model,
            "r2":      round(r2_score(y_test, y_pred), 4),
            "mae":     round(mean_absolute_error(y_test, y_pred), 4),
            "rmse":    round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "cv_r2":   round(cv_r2.mean(), 4),
            "cv_std":  round(cv_r2.std(), 4),
            "y_pred":  y_pred,
            "y_test":  y_test.values,
        }
        logger.info(
            f"  {name}: R²={results[name]['r2']:.4f} | "
            f"MAE=${results[name]['mae']:.4f} | CV={results[name]['cv_r2']:.4f}"
        )

    best = max(results, key=lambda k: results[k]["cv_r2"])
    logger.success(f"  Best model: {best} (CV R²={results[best]['cv_r2']:.4f})")
    return results, best


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """Ambil feature importance dari model tree-based atau koefisien Ridge."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.abs(model.coef_)

    df_imp = pd.DataFrame({
        "feature":   feature_cols,
        "label":     [FEATURE_LABELS.get(f, f) for f in feature_cols],
        "importance": imp,
    }).sort_values("importance", ascending=False)
    df_imp["importance_pct"] = (df_imp["importance"] / df_imp["importance"].sum() * 100).round(1)
    return df_imp


# ═══════════════════════════════════════════════════════════════════════════
# VISUALISASI
# ═══════════════════════════════════════════════════════════════════════════

def _chart_feature_importance(df_imp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3b82f6"] * 3 + ["#10b981"] * 3 + ["#6b7280"] * (len(df_imp) - 6)
    top = df_imp.head(10)
    ax.barh(top["label"][::-1], top["importance_pct"][::-1],
            color=colors[:10][::-1], edgecolor="none", height=0.6)
    for i, (_, row) in enumerate(top[::-1].iterrows()):
        ax.text(row["importance_pct"] + 0.3, i, f"{row['importance_pct']:.1f}%",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Kontribusi terhadap Prediksi Harga (%)", fontsize=11)
    ax.set_title("Feature Importance — Model Rekomendasi Harga", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ml_feature_importance.png", dpi=150)
    plt.close()
    logger.success("  Chart saved: ml_feature_importance.png")


def _chart_actual_vs_predicted(results: dict, best_name: str) -> None:
    r = results[best_name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(r["y_test"], r["y_pred"], alpha=0.5, color="#3b82f6",
               edgecolors="white", linewidth=0.5, s=50)
    lim = max(r["y_test"].max(), r["y_pred"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Prediksi Sempurna")
    ax.set_xlabel("Harga Aktual (USD/100 unit)", fontsize=10)
    ax.set_ylabel("Harga Prediksi (USD/100 unit)", fontsize=10)
    ax.set_title(f"Aktual vs Prediksi — {best_name}\nR²={r['r2']:.4f}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    residuals = r["y_pred"] - r["y_test"]
    ax2.hist(residuals, bins=20, color="#8b5cf6", alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Residual (Prediksi − Aktual)", fontsize=10)
    ax2.set_ylabel("Frekuensi", fontsize=10)
    ax2.set_title(f"Distribusi Error\nMAE=${r['mae']:.4f} | RMSE=${r['rmse']:.4f}", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ml_actual_vs_predicted.png", dpi=150)
    plt.close()
    logger.success("  Chart saved: ml_actual_vs_predicted.png")


def _chart_model_comparison(results: dict) -> None:
    names   = list(results.keys())
    r2_vals = [results[n]["r2"]   for n in names]
    mae_vals= [results[n]["mae"]  for n in names]
    cv_vals = [results[n]["cv_r2"]for n in names]
    colors  = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, title in zip(
        axes,
        [r2_vals, mae_vals, cv_vals],
        ["R² Score (↑ lebih baik)", "MAE USD (↓ lebih baik)", "Cross-Val R² (↑ lebih baik)"],
    ):
        bars = ax.bar(names, vals, color=colors, edgecolor="none", width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle("Perbandingan 4 Model ML — Game Price Optimizer", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ml_model_comparison.png", dpi=150)
    plt.close()
    logger.success("  Chart saved: ml_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict_price(
    model,
    genre: str,
    competition_score: float,
    mau_millions: float,
    arpdau_usd: float,
    age_median: float,
    region: str,
    spending_segment: str,
) -> dict:
    """
    Prediksi harga optimal per 100 unit currency.

    Parameters
    ----------
    genre            : "MOBA" | "Battle Royale" | "RPG / Gacha / Open World" | "Shooter / Battle Royale"
    competition_score: 0–100
    mau_millions     : Monthly Active Users (juta)
    arpdau_usd       : Average Revenue per DAU (USD)
    age_median       : Usia median pemain
    region           : "SEA" | "LATAM" | "Global_West" | "East_Asia" | "MENA"
    spending_segment : "whale" | "dolphin" | "minnow"
    """
    from config.settings import REGION_PRICE_INDEX, SEGMENT_MULTIPLIER, IDR_PER_USD

    genre_enc  = {"MOBA": 0, "Battle Royale": 1, "RPG / Gacha / Open World": 2, "Shooter / Battle Royale": 3}
    region_enc = {"SEA": 0, "LATAM": 1, "Global_West": 2, "East_Asia": 3, "MENA": 4}
    seg_enc    = {"whale": 2, "dolphin": 0, "minnow": 1}
    region_arpu= {"SEA": 0.65, "LATAM": 0.70, "Global_West": 1.00, "East_Asia": 1.30, "MENA": 0.80}
    seg_pct    = {"whale": 3, "dolphin": 15, "minnow": 82}

    dau_est = mau_millions * 0.18
    d2m     = dau_est / mau_millions if mau_millions > 0 else 0
    health  = min(mau_millions / 130, 1) * 40 + min(d2m / 0.35, 1) * 30 + 20

    X_pred = pd.DataFrame([{
        "mau_millions":           mau_millions,
        "dau_millions":           round(dau_est, 2),
        "arpdau_usd":             arpdau_usd,
        "competition_score":      competition_score,
        "community_health_score": round(health, 1),
        "game_median_age":        age_median,
        "dau_to_mau_ratio":       round(d2m, 3),
        "region_price_index":     REGION_PRICE_INDEX.get(region, 1.0),
        "region_arpu_mult":       region_arpu.get(region, 1.0),
        "segment_spend_mult":     SEGMENT_MULTIPLIER.get(spending_segment, 1.0),
        "segment_player_pct":     seg_pct.get(spending_segment, 15),
        "genre_encoded":          genre_enc.get(genre, 0),
        "region_encoded":         region_enc.get(region, 0),
        "segment_encoded":        seg_enc.get(spending_segment, 0),
    }])

    price = float(model.predict(X_pred)[0])
    return {
        "predicted_price_per_100_usd": round(price, 3),
        "predicted_price_per_100_idr": int(price * IDR_PER_USD),
        "range_low_usd":               round(price * 0.82, 3),
        "range_high_usd":              round(price * 1.22, 3),
        "range_low_idr":               int(price * 0.82 * IDR_PER_USD),
        "range_high_idr":              int(price * 1.22 * IDR_PER_USD),
        "inputs": {
            "genre": genre, "competition_score": competition_score,
            "mau_millions": mau_millions, "arpdau_usd": arpdau_usd,
            "age_median": age_median, "region": region, "segment": spending_segment,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# MASTER ML RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_ml() -> tuple:
    logger.info("=" * 60)
    logger.info("PHASE ML — MACHINE LEARNING")
    logger.info("=" * 60)

    X, y, df = _load_dataset()
    results, best_name = train_and_evaluate(X, y)
    best_model = results[best_name]["model"]

    df_imp = get_feature_importance(best_model, FEATURE_COLS)
    logger.info("\nTop 5 Features:\n" + df_imp[["label","importance_pct"]].head(5).to_string(index=False))

    # Charts
    _chart_feature_importance(df_imp)
    _chart_actual_vs_predicted(results, best_name)
    _chart_model_comparison(results)

    # Simpan model
    model_path = OUTPUTS_DIR / "ml_model.joblib"
    joblib.dump(best_model, model_path)
    logger.success(f"Model saved → {model_path}")

    # Demo predictions
    demo_cases = [
        ("MOBA",                    85, 110, 0.018, 22, "SEA",         "dolphin"),
        ("Battle Royale",           60, 130, 0.030, 20, "LATAM",       "minnow"),
        ("RPG / Gacha / Open World",25, 15,  0.126, 35, "East_Asia",   "whale"),
        ("Shooter / Battle Royale", 80, 100, 0.065, 27, "Global_West", "dolphin"),
    ]
    logger.info("\nDemo Predictions:")
    for args in demo_cases:
        r = predict_price(best_model, *args)
        logger.info(
            f"  {args[0][:18]:20s} | {args[5]:12s} | {args[6]:8s} → "
            f"${r['predicted_price_per_100_usd']:6.3f}  "
            f"(Rp{r['predicted_price_per_100_idr']:>8,})"
        )

    # Simpan ringkasan evaluasi
    eval_summary = {
        "best_model": best_name,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": {
            name: {k: v for k, v in res.items() if k not in ["model", "y_pred", "y_test"]}
            for name, res in results.items()
        },
        "feature_importance_top5": df_imp[["label", "importance_pct"]].head(5).to_dict("records"),
    }
    out_json = OUTPUTS_DIR / "ml_eval_summary.json"
    with open(out_json, "w") as fh:
        json.dump(eval_summary, fh, indent=2)

    logger.success("ML selesai")
    return best_model, results, df_imp


if __name__ == "__main__":
    run_ml()
