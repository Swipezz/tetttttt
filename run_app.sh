#!/bin/bash
# ============================================================
# run_app.sh — Launcher Streamlit App
# Game Price Intelligence Dashboard
#
# Usage:
#   bash run_app.sh          # jalankan di localhost:8501
#   bash run_app.sh --port 8080
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$PROJECT_DIR/streamlit_app/app.py"
DATA_DIR="$PROJECT_DIR/data/processed"
PORT="${1:-8501}"

echo "======================================"
echo " Game Price Intelligence — Streamlit"
echo "======================================"

# Cek data sudah ada
if [ ! -f "$DATA_DIR/ml_features.csv" ]; then
    echo ""
    echo "⚠  Data belum ada. Jalankan ETL pipeline terlebih dahulu:"
    echo "   python run_pipeline.py"
    echo ""
    read -p "Jalankan pipeline sekarang? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        cd "$PROJECT_DIR"
        python run_pipeline.py
    else
        echo "Dibatalkan."
        exit 1
    fi
fi

echo ""
echo "✓ Data ditemukan di $DATA_DIR"
echo "✓ Membuka aplikasi di http://localhost:$PORT"
echo ""
echo "Tekan Ctrl+C untuk menghentikan aplikasi."
echo ""

cd "$PROJECT_DIR"
streamlit run "$APP_FILE" \
    --server.port "$PORT" \
    --server.headless true \
    --theme.base dark \
    --theme.primaryColor "#3b82f6" \
    --theme.backgroundColor "#0f1629" \
    --theme.secondaryBackgroundColor "#1e293b" \
    --theme.textColor "#f1f5f9"
