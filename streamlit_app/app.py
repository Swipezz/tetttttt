"""
streamlit_app/app.py
=====================
Game Price Intelligence — Streamlit App
Business Intelligence + Machine Learning yang bisa dilatih sendiri

Jalankan:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor

# Add src to path untuk import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DB_URL

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── Streamlit page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Game Price Intelligence",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1629; }
    .main .block-container { padding-top: 1.5rem; }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f2744);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 12px; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px; font-weight: 700; color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .sub {
        font-size: 11px; color: #475569; margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1e40af20, transparent);
        border-left: 3px solid #3b82f6;
        padding: 8px 16px; border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    }

    /* Insight box */
    .insight-box {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 8px; padding: 12px 16px;
        font-size: 13px; color: #cbd5e1;
        line-height: 1.6; margin-top: 12px;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #052e16, #0f2744);
        border: 1px solid #10b981;
        border-radius: 12px; padding: 24px;
        text-align: center; margin-top: 16px;
    }
    .result-price { font-size: 48px; font-weight: 800; color: #10b981; }
    .result-sub   { font-size: 14px; color: #64748b; margin-top: 4px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f1629; border-right: 1px solid #1e293b; }

    /* Hide only the footer branding. Keep Streamlit's header/menu visible
       because the Deploy action lives in the app chrome. */
    footer { visibility: hidden; }

    /* Tab style */
    .stTabs [data-baseweb="tab"] { color: #64748b; }
    .stTabs [aria-selected="true"] { color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# AUTO-BOOTSTRAP DATA (untuk Streamlit Cloud deployment)
# ══════════════════════════════════════════════════════════════════════════
from streamlit_app.bootstrap_data import bootstrap
bootstrap()

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """
    Muat dataset dari Aiven PostgreSQL (primary) atau fallback ke CSV lokal.
    Di-cache oleh Streamlit agar tidak reload setiap interact.
    """
    try:
        logger.info("Attempting to connect to configured database")
        engine = create_engine(DB_URL)
        
        # Test koneksi
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info("✅ Database connection successful — loading from Aiven PostgreSQL")
        st.info("📊 Data dimuat dari **Aiven PostgreSQL**", icon="✅")
        
        return {
            "game":    pd.read_sql("SELECT * FROM dim_game", engine),
            "pricing": pd.read_sql("SELECT * FROM fact_pricing", engine),
            "age":     pd.read_sql("SELECT * FROM dim_age", engine),
            "comp":    pd.read_sql("SELECT * FROM fact_competition", engine),
            "revenue": pd.read_sql("SELECT * FROM fact_revenue_monthly", engine),
            "ml":      pd.read_sql("SELECT * FROM ml_features", engine),
        }
    
    except Exception as exc:
        logger.warning("Database unavailable ({}), falling back to local CSV", type(exc).__name__)
        st.warning(f"⚠️  Aiven DB gagal — menggunakan **CSV lokal** (mungkin data tidak ter-sync)", icon="⚠️")
        
        try:
            return {
                "game":    pd.read_csv(DATA / "dim_game.csv"),
                "pricing": pd.read_csv(DATA / "fact_pricing.csv"),
                "age":     pd.read_csv(DATA / "dim_age.csv"),
                "comp":    pd.read_csv(DATA / "fact_competition.csv"),
                "revenue": pd.read_csv(DATA / "fact_revenue_monthly.csv"),
                "ml":      pd.read_csv(DATA / "ml_features.csv"),
            }
        except FileNotFoundError as e:
            st.error(f"❌ Data tidak ditemukan: {e}\n\nJalankan `python run_pipeline.py` terlebih dahulu.")
            st.stop()


# Warna per game
GAME_COLORS = {
    "mlbb":            "#e879a0",
    "free_fire":       "#34d399",
    "pubg_mobile":     "#60a5fa",
    "genshin_impact":  "#fb923c",
    "honkai_star_rail":"#a78bfa",
    "cod_mobile":      "#facc15",
}
GAME_SHORT = {
    "mlbb": "MLBB", "free_fire": "Free Fire", "pubg_mobile": "PUBG Mobile",
    "genshin_impact": "Genshin Impact", "honkai_star_rail": "Honkai: SR", "cod_mobile": "CoD Mobile",
}
IDR_RATE = 16_250


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎮 Game Price Intelligence")
        st.markdown("---")
        page = st.radio(
            "Navigasi",
            ["📊 Business Intelligence", "🤖 Machine Learning"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Data Sources**")
        st.markdown("📡 28 sumber real-time")
        st.markdown("🎯 6 game mobile terbesar")
        st.markdown("🌍 5 region pasar global")
        st.markdown("---")
        st.caption("Game Price ETL Pipeline v2.0")
        st.caption("Data: Mei 2026 | 28 sumber publik")
    return page


# ══════════════════════════════════════════════════════════════════════════
# BI PAGE
# ══════════════════════════════════════════════════════════════════════════

def page_bi(data):
    df_game  = data["game"]
    df_price = data["pricing"]
    df_age   = data["age"]
    df_comp  = data["comp"]
    df_rev   = data["revenue"]
    df_ml    = data["ml"]

    st.markdown("# 📊 Business Intelligence Dashboard")
    st.markdown("Analisis pasar top-up game mobile berdasarkan data real dari 28 sumber publik.")

    # ── Filter sidebar ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔧 Filter BI")
        selected_games = st.multiselect(
            "Pilih Game",
            options=df_game["game_key"].tolist(),
            default=df_game["game_key"].tolist(),
            format_func=lambda x: GAME_SHORT.get(x, x),
        )
        idr_toggle = st.toggle("Tampilkan dalam IDR", value=False)

    if not selected_games:
        st.warning("Pilih minimal 1 game.")
        return

    df_game_f  = df_game[df_game["game_key"].isin(selected_games)]
    df_price_f = df_price[df_price["game_key"].isin(selected_games)]
    df_age_f   = df_age[df_age["game_key"].isin(selected_games)]
    df_comp_f  = df_comp[df_comp["game_key"].isin(selected_games)]
    df_rev_f   = df_rev[df_rev["game_key"].isin(selected_games)]

    # ── KPI Row ───────────────────────────────────────────────────────────
    total_rev = df_game_f["revenue_2024_usd"].sum()
    avg_arpdau= df_game_f["arpdau_usd"].mean()
    total_mau = df_game_f["mau_millions"].sum()
    avg_disc  = df_price_f["discount_avg_pct"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total Revenue 2024</div>
            <div class="value">${total_rev/1e9:.1f}M</div>
            <div class="sub">{len(selected_games)} game dipilih</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total MAU</div>
            <div class="value">{total_mau:.0f}Jt</div>
            <div class="sub">Pengguna aktif bulanan</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Rata-rata ARPDAU</div>
            <div class="value">${avg_arpdau:.4f}</div>
            <div class="sub">Revenue per DAU per hari</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Avg Diskon 3rd-Party</div>
            <div class="value">{avg_disc:.1f}%</div>
            <div class="sub">vs harga official</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tab BI ────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "💰 Revenue & Pasar",
        "🎯 Harga Top-Up",
        "👥 Demografi Pemain",
        "🏆 Kompetisi",
        "🔥 Heatmap Korelasi",
    ])

    # ── TAB 1: Revenue ────────────────────────────────────────────────────
    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Revenue 2024 per Game")
            rev_col = "revenue_2024_usd"
            df_sorted = df_game_f.sort_values(rev_col, ascending=True)
            fig = px.bar(
                df_sorted, x=rev_col, y="game_key",
                orientation="h",
                color="game_key",
                color_discrete_map=GAME_COLORS,
                labels={"revenue_2024_usd": "Revenue USD", "game_key": ""},
                custom_data=["game", rev_col],
            )
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Revenue: $%{customdata[1]:,.0f}<extra></extra>",
                texttemplate="$%{x:,.0f}",
                textposition="outside",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False,
                xaxis=dict(gridcolor="#1e293b", tickprefix="$"),
                yaxis=dict(tickvals=df_sorted["game_key"].tolist(),
                           ticktext=[GAME_SHORT.get(k,k) for k in df_sorted["game_key"].tolist()]),
                height=300, margin=dict(l=0,r=40,t=0,b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class="insight-box">
                💡 <b>Insight:</b> PUBG Mobile mendominasi revenue ($2M) karena basis pengguna di China (64% revenue) dan spending whale yang tinggi. Game gacha seperti Genshin/HSR menghasilkan revenue besar meski MAU lebih kecil.
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("#### 🌐 Kompetisi vs ARPDAU (Bubble = Revenue)")
            # dim_game sudah punya competition_score — gunakan langsung tanpa merge
            # untuk menghindari konflik kolom competition_score_x / competition_score_y
            df_merge = df_game_f.copy()
            if "competition_score" not in df_merge.columns:
                comp_cols = df_comp_f[["game_key", "competition_score"]]
                df_merge = df_merge.merge(comp_cols, on="game_key", how="left")
            df_merge["game_short"] = df_merge["game_key"].map(GAME_SHORT)
            df_merge["size_"] = df_merge["revenue_2024_usd"] / 5e7

            fig2 = px.scatter(
                df_merge,
                x="competition_score", y="arpdau_usd",
                size="size_", color="game_key",
                color_discrete_map=GAME_COLORS,
                text="game_short",
                hover_data={"game_short": True, "competition_score": True,
                            "arpdau_usd": ":.3f", "size_": False, "game_key": False},
                labels={"competition_score": "Skor Kompetisi (0-100)",
                        "arpdau_usd": "ARPDAU (USD)"},
            )
            fig2.update_traces(textposition="top center", textfont_size=9)
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False,
                xaxis=dict(gridcolor="#1e293b", range=[0, 105]),
                yaxis=dict(gridcolor="#1e293b"),
                height=300, margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("""<div class="insight-box">
                💡 Game gacha (kiri-atas) = kompetisi rendah + ARPDAU tinggi → ruang premium pricing lebih luas. Game MOBA/BR (kanan-bawah) = kompetisi tinggi → harga harus lebih kompetitif.
            </div>""", unsafe_allow_html=True)

        # Monthly revenue trend
        st.markdown("#### 📈 Tren Revenue Bulanan 2024")
        df_rev_2024 = df_rev_f[df_rev_f["year"] == 2024].copy()
        df_rev_2024["game_short"] = df_rev_2024["game_key"].map(GAME_SHORT)

        fig3 = px.line(
            df_rev_2024, x="month_name", y="revenue_usd_millions",
            color="game_key", color_discrete_map=GAME_COLORS,
            markers=True, line_dash_sequence=["solid","dash"],
            labels={"revenue_usd_millions": "Revenue (USD Juta)", "month_name": "Bulan", "game_key": "Game"},
            category_orders={"month_name": ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]},
            custom_data=["game_key"],
        )
        # Seasonal annotations
        fig3.add_vrect(x0="Jan", x1="Feb", fillcolor="#10b981", opacity=0.05,
                       annotation_text="Peak Season", annotation_position="top left")
        fig3.add_vrect(x0="Jun", x1="Agu", fillcolor="#ef4444", opacity=0.05,
                       annotation_text="Low Season", annotation_position="top left")
        fig3.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>%{x}: $%{y}M<extra></extra>"
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", legend_title="Game",
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", ticksuffix="M"),
            height=340, margin=dict(l=0,r=0,t=20,b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 2: Harga Top-Up ───────────────────────────────────────────────
    with tabs[1]:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 💰 Tabel Harga Top-Up Lengkap")

            # Filter tier
            tier_filter = st.multiselect(
                "Filter Tier",
                options=["micro","small","medium","large","mega"],
                default=["micro","small","medium","large","mega"],
                key="tier_filter_bi",
            )
            df_pf = df_price_f[df_price_f["spend_tier"].isin(tier_filter)].copy()

            price_col = "price_official_idr" if idr_toggle else "price_official_usd"
            price_3rd = "price_3rdparty_idr" if idr_toggle else "price_3rdparty_usd"
            prefix    = "Rp" if idr_toggle else "$"

            show_df = df_pf[[
                "game","tier_label","spend_tier",
                "base_currency","bonus_currency","total_currency",
                price_col, price_3rd, "discount_avg_pct","value_score","price_elasticity"
            ]].copy()
            show_df.columns = [
                "Game","Tier","Kategori",
                "Base Curr.","Bonus","Total Curr.",
                f"Harga Official ({prefix})", f"Harga 3rd-Party ({prefix})",
                "Diskon (%)","Value Score","Elastisitas"
            ]

            # Color-code by tier
            tier_colors = {"micro":"#3b82f6","small":"#10b981","medium":"#f59e0b","large":"#ef4444","mega":"#a78bfa"}

            st.dataframe(
                show_df,
                use_container_width=True,
                height=380,
                column_config={
                    "Diskon (%)": st.column_config.ProgressColumn(
                        min_value=0, max_value=40, format="%.1f%%"
                    ),
                    "Value Score": st.column_config.NumberColumn(format="%.1f"),
                    "Elastisitas": st.column_config.NumberColumn(format="%.1f"),
                },
                hide_index=True,
            )

        with col2:
            st.markdown("#### 📊 Diskon 3rd-Party per Game")
            disc_df = df_price_f.groupby("game_key")["discount_avg_pct"].mean().reset_index()
            disc_df["game_short"] = disc_df["game_key"].map(GAME_SHORT)

            fig_disc = px.bar(
                disc_df.sort_values("discount_avg_pct"),
                x="discount_avg_pct", y="game_short",
                orientation="h",
                color="game_key", color_discrete_map=GAME_COLORS,
                labels={"discount_avg_pct":"Diskon (%)","game_short":""},
                text="discount_avg_pct",
            )
            fig_disc.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig_disc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False,
                xaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
                height=250, margin=dict(l=0,r=40,t=0,b=0),
            )
            st.plotly_chart(fig_disc, use_container_width=True)

            st.markdown("#### ⚡ Elastisitas Harga per Tier")
            elast_data = pd.DataFrame({
                "Tier": ["Micro","Small","Medium","Large","Mega"],
                "Elastisitas": [-2.1, -1.6, -0.9, -0.5, -0.3],
            })
            fig_el = px.bar(
                elast_data, x="Tier", y="Elastisitas",
                color="Elastisitas",
                color_continuous_scale=["#ef4444","#f59e0b","#3b82f6","#10b981","#a78bfa"],
                text="Elastisitas",
            )
            fig_el.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_el.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False,
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
                height=250, margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig_el, use_container_width=True)

    # ── TAB 3: Demografi ──────────────────────────────────────────────────
    with tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎂 Distribusi Usia Player")
            age_pivot = df_age_f.pivot_table(
                index="game_key", columns="age_band", values="pct", fill_value=0
            ).reset_index()
            age_pivot["game_short"] = age_pivot["game_key"].map(GAME_SHORT)

            fig_age = go.Figure()
            age_bands = ["13-17","18-24","25-34","35+"]
            age_colors_plot = ["#60a5fa","#a78bfa","#34d399","#f59e0b"]

            for band, color in zip(age_bands, age_colors_plot):
                if band in age_pivot.columns:
                    fig_age.add_trace(go.Bar(
                        name=f"{band} tahun",
                        x=age_pivot["game_short"],
                        y=age_pivot[band],
                        marker_color=color,
                        text=age_pivot[band].apply(lambda v: f"{v}%"),
                        textposition="inside",
                        textfont=dict(size=9, color="white"),
                    ))

            fig_age.update_layout(
                barmode="stack",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", legend=dict(orientation="h", y=-0.2),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
                height=320, margin=dict(l=0,r=0,t=0,b=60),
            )
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Sensitivitas Harga per Game")
            df_sens = df_game_f.copy()
            df_sens["game_short"] = df_sens["game_key"].map(GAME_SHORT)
            df_sens["sensitivity"] = df_sens["age_median"].apply(
                lambda a: "🔴 Sangat Sensitif" if a < 22 else ("🟡 Sedang" if a < 30 else "🟢 Kurang Sensitif")
            )
            df_sens["sensitivity_val"] = df_sens["age_median"].apply(
                lambda a: 3 if a < 22 else (2 if a < 30 else 1)
            )

            for _, row in df_sens.iterrows():
                color = GAME_COLORS.get(row["game_key"], "#888")
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.5);border:1px solid #1e293b;
                    border-left:4px solid {color};border-radius:8px;
                    padding:12px 16px;margin-bottom:10px;display:flex;
                    justify-content:space-between;align-items:center">
                    <div>
                        <div style="font-weight:600;color:#f1f5f9;font-size:13px">{row['game_short']}</div>
                        <div style="color:#64748b;font-size:11px">Usia median: {row['age_median']} tahun | MAU: {row['mau_millions']}Jt</div>
                    </div>
                    <div style="font-size:13px;font-weight:600">{row['sensitivity']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### 🐋 Segmentasi Spending Player")
        seg_data = pd.DataFrame({
            "Segmen":  ["🐋 Whale", "🐬 Dolphin", "🐟 Minnow"],
            "Batas Bulanan": [">$100 (>Rp1,6Jt)", "$10-100 (Rp160rb-1,6Jt)", "<$10 (<Rp160rb)"],
            "% Player":  [3, 15, 82],
            "% Revenue": [60, 30, 10],
        })
        col1, col2 = st.columns(2)
        with col1:
            fig_s1 = px.pie(seg_data, names="Segmen", values="% Player",
                            color_discrete_sequence=["#ef4444","#3b82f6","#10b981"],
                            title="Komposisi Player (%)")
            fig_s1.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#94a3b8",
                                 height=260,margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig_s1, use_container_width=True)
        with col2:
            fig_s2 = px.pie(seg_data, names="Segmen", values="% Revenue",
                            color_discrete_sequence=["#ef4444","#3b82f6","#10b981"],
                            title="Kontribusi Revenue (%)")
            fig_s2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#94a3b8",
                                 height=260,margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig_s2, use_container_width=True)

    # ── TAB 4: Kompetisi ──────────────────────────────────────────────────
    with tabs[3]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏆 Skor Kompetisi per Game")
            df_c = df_comp_f.copy()
            df_c["game_short"] = df_c["game_key"].map(GAME_SHORT)
            df_c = df_c.sort_values("competition_score", ascending=True)

            fig_comp = px.bar(
                df_c, x="competition_score", y="game_short",
                orientation="h", color="game_key", color_discrete_map=GAME_COLORS,
                text="competition_score",
                labels={"competition_score":"Skor Kompetisi","game_short":""},
            )
            fig_comp.update_traces(texttemplate="%{text}/100", textposition="outside")
            fig_comp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False,
                xaxis=dict(gridcolor="#1e293b", range=[0,115]),
                height=280, margin=dict(l=0,r=60,t=0,b=0),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col2:
            st.markdown("#### 💰 Prize Pool Esports 2025")
            df_c2 = df_comp_f[df_comp_f["prize_pool_2025_usd"] > 0].copy()
            df_c2["game_short"] = df_c2["game_key"].map(GAME_SHORT)

            fig_pp = px.bar(
                df_c2.sort_values("prize_pool_2025_usd"),
                x="prize_pool_2025_usd", y="game_short",
                orientation="h", color="game_key", color_discrete_map=GAME_COLORS,
                text="prize_pool_2025_usd",
                labels={"prize_pool_2025_usd":"Prize Pool (USD)","game_short":""},
            )
            fig_pp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_pp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False,
                xaxis=dict(gridcolor="#1e293b", tickprefix="$"),
                height=280, margin=dict(l=0,r=100,t=0,b=0),
            )
            st.plotly_chart(fig_pp, use_container_width=True)

        st.markdown("#### 📊 Detail Kompetisi")
        df_c3 = df_comp_f.merge(df_game_f[["game_key","game"]], on="game_key")
        df_c3["game_short"] = df_c3["game_key"].map(GAME_SHORT)
        st.dataframe(
            df_c3[["game_short","competition_tier","competition_score",
                   "prize_pool_2025_usd","tournaments_total","peak_viewers","regional_leagues"]],
            column_config={
                "game_short":          st.column_config.TextColumn("Game"),
                "competition_tier":    st.column_config.TextColumn("Tier"),
                "competition_score":   st.column_config.ProgressColumn("Skor",min_value=0,max_value=100,format="%d/100"),
                "prize_pool_2025_usd": st.column_config.NumberColumn("Prize Pool",format="$%,.0f"),
                "tournaments_total":   st.column_config.NumberColumn("Turnamen"),
                "peak_viewers":        st.column_config.NumberColumn("Peak Viewers",format="%,.0f"),
                "regional_leagues":    st.column_config.NumberColumn("Liga Regional"),
            },
            hide_index=True, use_container_width=True,
        )

    # ── TAB 5: Heatmap ────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### 🔥 Heatmap Korelasi Antar Fitur")
        st.caption("Nilai mendekati +1 = korelasi positif kuat | mendekati -1 = korelasi negatif kuat | 0 = tidak berkorelasi")

        numeric_cols = [
            "mau_millions","dau_millions","arpdau_usd","competition_score",
            "community_health_score","game_median_age","region_price_index",
            "segment_spend_mult","optimal_price_per_100_usd",
        ]
        col_labels = {
            "mau_millions":"MAU (Juta)","dau_millions":"DAU (Juta)",
            "arpdau_usd":"ARPDAU","competition_score":"Skor Kompetisi",
            "community_health_score":"Health Score","game_median_age":"Usia Median",
            "region_price_index":"Indeks Harga Regional",
            "segment_spend_mult":"Multiplier Segmen","optimal_price_per_100_usd":"Harga Optimal",
        }

        df_corr = df_ml[numeric_cols].rename(columns=col_labels).corr()
        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        df_corr_masked = df_corr.where(~mask)

        fig_hm = px.imshow(
            df_corr_masked,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_hm.update_traces(textfont_size=11)
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            coloraxis_colorbar=dict(title="Korelasi", tickformat=".1f"),
            height=520, margin=dict(l=0,r=0,t=20,b=0),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Price heatmap
        st.markdown("#### 🗺️ Heatmap Harga Optimal: Game × Region")
        price_pivot = df_ml.pivot_table(
            index="game", columns="region",
            values="optimal_price_per_100_usd", aggfunc="mean"
        )
        fig_ph = px.imshow(
            price_pivot,
            color_continuous_scale="YlOrRd",
            text_auto=".3f",
            labels={"color":"Harga Optimal (USD/100 unit)"},
            aspect="auto",
        )
        fig_ph.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            xaxis_title="Region Pasar",
            height=340, margin=dict(l=0,r=0,t=20,b=0),
        )
        st.plotly_chart(fig_ph, use_container_width=True)
        st.markdown("""<div class="insight-box">
            💡 <b>Cara membaca heatmap ini:</b> Warna merah = harga optimal lebih tinggi. 
            Genshin Impact + East Asia memiliki harga tertinggi karena ARPDAU tinggi (0.126) dan 
            kompetisi rendah (25/100) — ideal untuk premium pricing. 
            Free Fire + SEA paling murah karena pemain muda dan harga sensitif.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ML PAGE
# ══════════════════════════════════════════════════════════════════════════

def page_ml(data):
    df_ml = data["ml"].copy()

    st.markdown("# 🤖 Machine Learning — Price Optimizer")
    st.markdown("Latih model sendiri dengan konfigurasi yang bisa disesuaikan, split 80% training / 20% testing.")

    # ── Tabs ML ───────────────────────────────────────────────────────────
    ml_tabs = st.tabs([
        "⚙️ Konfigurasi & Training",
        "📊 Evaluasi Model",
        "🔮 Prediksi Harga",
        "📖 Panduan Fitur",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: KONFIGURASI & TRAINING
    # ══════════════════════════════════════════════════════════════════════
    with ml_tabs[0]:
        col_cfg, col_data = st.columns([1, 1])

        with col_cfg:
            st.markdown("### ⚙️ Konfigurasi Training")

            # Model selection
            model_choice = st.selectbox(
                "Pilih Algoritma Model",
                ["Gradient Boosting (Recommended)", "XGBoost", "Random Forest", "Ridge Regression"],
                help="Gradient Boosting biasanya memberikan akurasi terbaik untuk dataset ini."
            )

            # Train-test split
            st.markdown("**Rasio Train / Test Split**")
            test_size = st.slider(
                "Ukuran Data Test (%)",
                min_value=10, max_value=40, value=20, step=5,
                help="Default 20% sesuai standar 80/20. Semakin besar test set = evaluasi lebih ketat.",
                format="%d%%"
            )
            train_pct = 100 - test_size
            st.info(f"📊 **{train_pct}% Training** ({int(len(df_ml)*train_pct/100)} baris) | **{test_size}% Testing** ({int(len(df_ml)*test_size/100)} baris)")

            # Hyperparameter by model
            st.markdown("**Hyperparameter**")
            hp = {}
            if "Gradient Boosting" in model_choice or "XGBoost" in model_choice:
                hp["n_estimators"] = st.slider("Jumlah Pohon (n_estimators)", 50, 500, 300, 50)
                hp["learning_rate"] = st.select_slider("Learning Rate",
                    options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
                hp["max_depth"]  = st.slider("Kedalaman Pohon (max_depth)", 2, 8, 4)
                if "XGBoost" in model_choice:
                    hp["subsample"] = st.slider("Subsample Ratio", 0.5, 1.0, 0.85, 0.05)
            elif "Random Forest" in model_choice:
                hp["n_estimators"] = st.slider("Jumlah Pohon", 50, 500, 200, 50)
                hp["max_depth"]  = st.slider("Kedalaman Pohon", 2, 10, 6)
            else:  # Ridge
                hp["alpha"] = st.select_slider("Regularisasi Alpha",
                    options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)

            # Cross-validation
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5,
                                 help="Lebih banyak fold = evaluasi lebih reliable tapi lebih lambat.")

            # Feature selection
            st.markdown("**Pilih Fitur yang Digunakan**")
            all_features = [
                "mau_millions", "dau_millions", "arpdau_usd", "competition_score",
                "community_health_score", "game_median_age", "dau_to_mau_ratio",
                "region_price_index", "region_arpu_mult", "segment_spend_mult",
                "segment_player_pct", "genre_encoded", "region_encoded", "segment_encoded",
            ]
            feature_labels = {
                "mau_millions": "MAU — Ukuran Komunitas",
                "dau_millions": "DAU — Pengguna Harian",
                "arpdau_usd": "ARPDAU — Revenue per DAU/Hari",
                "competition_score": "Skor Kompetisi",
                "community_health_score": "Skor Kesehatan Komunitas",
                "game_median_age": "Usia Median Pemain",
                "dau_to_mau_ratio": "Rasio DAU/MAU (Retensi)",
                "region_price_index": "Indeks Harga Regional",
                "region_arpu_mult": "Multiplier ARPU Regional",
                "segment_spend_mult": "Multiplier Spending Segmen",
                "segment_player_pct": "% Player di Segmen",
                "genre_encoded": "Genre (encoded)",
                "region_encoded": "Region (encoded)",
                "segment_encoded": "Segmen (encoded)",
            }
            selected_features = st.multiselect(
                "Fitur yang diikutsertakan",
                options=all_features,
                default=all_features,
                format_func=lambda x: feature_labels.get(x, x),
            )

        with col_data:
            st.markdown("### 📋 Preview Dataset")
            st.caption(f"Total: {len(df_ml)} baris × {len(df_ml.columns)} kolom")

            show_cols = ["game","region","spending_segment","arpdau_usd",
                         "competition_score","game_median_age","optimal_price_per_100_usd"]
            st.dataframe(
                df_ml[show_cols].rename(columns={
                    "game": "Game", "region": "Region",
                    "spending_segment": "Segmen", "arpdau_usd": "ARPDAU",
                    "competition_score": "Kompetisi", "game_median_age": "Usia Median",
                    "optimal_price_per_100_usd": "Target Harga (USD/100)",
                }),
                use_container_width=True, height=300, hide_index=True,
            )

            st.markdown("**Distribusi Target Variable**")
            fig_dist = px.histogram(
                df_ml, x="optimal_price_per_100_usd",
                color_discrete_sequence=["#3b82f6"],
                nbins=20,
                labels={"optimal_price_per_100_usd":"Harga Optimal (USD/100 unit)"},
            )
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False,
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
                height=200, margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # ── TOMBOL TRAIN ──────────────────────────────────────────────────
        st.markdown("---")
        train_btn = st.button(
            "🚀 Mulai Training Model",
            type="primary", use_container_width=True,
        )

        if train_btn:
            if len(selected_features) < 3:
                st.error("Pilih minimal 3 fitur untuk training.")
                return

            with st.spinner("🔄 Sedang melatih model..."):
                # Prepare data
                X = df_ml[selected_features].astype(float)
                y = df_ml["optimal_price_per_100_usd"].astype(float)

                # 80/20 split (atau sesuai slider)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )

                # Build model sesuai pilihan
                if "Gradient Boosting" in model_choice:
                    model = GradientBoostingRegressor(
                        n_estimators=hp.get("n_estimators",300),
                        learning_rate=hp.get("learning_rate",0.05),
                        max_depth=hp.get("max_depth",4),
                        random_state=42,
                    )
                elif "XGBoost" in model_choice:
                    model = XGBRegressor(
                        n_estimators=hp.get("n_estimators",300),
                        learning_rate=hp.get("learning_rate",0.05),
                        max_depth=hp.get("max_depth",4),
                        subsample=hp.get("subsample",0.85),
                        random_state=42, verbosity=0,
                    )
                elif "Random Forest" in model_choice:
                    model = RandomForestRegressor(
                        n_estimators=hp.get("n_estimators",200),
                        max_depth=hp.get("max_depth",6),
                        random_state=42, n_jobs=-1,
                    )
                else:
                    model = Ridge(alpha=hp.get("alpha",1.0))

                # Train
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                r2   = r2_score(y_test, y_pred)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Cross-validation
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

                # Feature importance
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_)
                else:
                    importances = np.ones(len(selected_features)) / len(selected_features)

                df_imp = pd.DataFrame({
                    "feature": selected_features,
                    "label":   [feature_labels.get(f,f) for f in selected_features],
                    "importance": importances,
                }).sort_values("importance", ascending=False)
                df_imp["importance_pct"] = (df_imp["importance"] / df_imp["importance"].sum() * 100).round(1)

                # Simpan ke session state
                st.session_state["trained_model"]  = model
                st.session_state["train_results"]  = {
                    "model_name":  model_choice,
                    "r2": r2, "mae": mae, "rmse": rmse,
                    "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
                    "cv_scores": cv_scores.tolist(),
                    "X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test, "y_pred": y_pred,
                    "df_imp": df_imp,
                    "train_size": len(X_train), "test_size_": len(X_test),
                    "selected_features": selected_features,
                    "test_pct": test_size,
                }

                # Simpan model ke file
                model_path = OUT / "ml_model_custom.joblib"
                joblib.dump(model, model_path)

            st.success(f"✅ Training selesai! R² = {r2:.4f} | MAE = ${mae:.4f} | CV R² = {cv_scores.mean():.4f}")
            st.info("Buka tab **📊 Evaluasi Model** untuk melihat hasil detail, atau **🔮 Prediksi Harga** untuk inferensi.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: EVALUASI
    # ══════════════════════════════════════════════════════════════════════
    with ml_tabs[1]:
        if "train_results" not in st.session_state:
            st.info("⚠️ Belum ada model yang dilatih. Buka tab **⚙️ Konfigurasi & Training** dan klik **Mulai Training**.")
            return

        res = st.session_state["train_results"]

        st.markdown(f"### Hasil: {res['model_name']}")
        st.caption(f"Split: {100-res['test_pct']}% Train ({res['train_size']} baris) / {res['test_pct']}% Test ({res['test_size_']} baris)")

        # ── Metric cards ──────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("R² Score", f"{res['r2']:.4f}", "Akurasi model (1.0 = sempurna)", "#10b981"),
            ("MAE (USD)", f"${res['mae']:.4f}", "Rata-rata error prediksi", "#f59e0b"),
            ("RMSE (USD)", f"${res['rmse']:.4f}", "Root mean squared error", "#fb923c"),
            (f"CV R² ({len(res['cv_scores'])}-fold)", f"{res['cv_mean']:.4f}±{res['cv_std']:.4f}", "Konsistensi cross-validation", "#3b82f6"),
        ]
        for col, (label, val, sub, color) in zip([m1,m2,m3,m4], metrics):
            with col:
                st.markdown(f"""<div class="metric-card" style="border-top:3px solid {color}">
                    <div class="label">{label}</div>
                    <div class="value" style="font-size:22px;color:{color}">{val}</div>
                    <div class="sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # Actual vs Predicted
        with col1:
            st.markdown("#### 🎯 Aktual vs Prediksi")
            y_test_arr = np.array(res["y_test"])
            y_pred_arr = np.array(res["y_pred"])

            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=y_test_arr, y=y_pred_arr,
                mode="markers",
                marker=dict(color="#3b82f6", size=10, opacity=0.7,
                            line=dict(color="white",width=0.5)),
                name="Prediksi",
                hovertemplate="Aktual: $%{x:.3f}<br>Prediksi: $%{y:.3f}<extra></extra>",
            ))
            lim = max(y_test_arr.max(), y_pred_arr.max()) * 1.1
            fig_avp.add_trace(go.Scatter(
                x=[0,lim], y=[0,lim],
                mode="lines",
                line=dict(color="#ef4444", dash="dash", width=2),
                name="Sempurna",
            ))
            fig_avp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                xaxis=dict(title="Harga Aktual (USD/100)", gridcolor="#1e293b"),
                yaxis=dict(title="Harga Prediksi (USD/100)", gridcolor="#1e293b"),
                legend=dict(x=0.05,y=0.95),
                height=340, margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig_avp, use_container_width=True)

        # Residual distribution
        with col2:
            st.markdown("#### 📊 Distribusi Residual (Error)")
            residuals = y_pred_arr - y_test_arr
            fig_res = px.histogram(
                x=residuals, nbins=15,
                color_discrete_sequence=["#8b5cf6"],
                labels={"x":"Residual (Prediksi − Aktual)"},
            )
            fig_res.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=2)
            fig_res.add_annotation(x=0, y=1, yref="paper",
                text="Ideal: residual = 0", showarrow=False,
                font=dict(color="#ef4444",size=10), xanchor="left", yanchor="top")
            fig_res.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False,
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b", title="Frekuensi"),
                height=340, margin=dict(l=0,r=0,t=0,b=0),
            )
            st.plotly_chart(fig_res, use_container_width=True)

        col3, col4 = st.columns(2)

        # Feature importance
        with col3:
            st.markdown("#### 🔍 Feature Importance")
            df_imp = res["df_imp"]
            fig_fi = px.bar(
                df_imp.head(10).iloc[::-1],
                x="importance_pct", y="label",
                orientation="h",
                color="importance_pct",
                color_continuous_scale=["#1e3a5f","#3b82f6","#60a5fa"],
                text="importance_pct",
                labels={"importance_pct":"Kontribusi (%)","label":""},
            )
            fig_fi.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", showlegend=False, coloraxis_showscale=False,
                xaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
                height=340, margin=dict(l=0,r=60,t=0,b=0),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # CV scores
        with col4:
            st.markdown(f"#### 🔄 Cross-Validation R² ({len(res['cv_scores'])} Folds)")
            cv_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(res["cv_scores"]))],
                "R²":   res["cv_scores"],
            })
            fig_cv = px.bar(
                cv_df, x="Fold", y="R²",
                color="R²",
                color_continuous_scale=["#ef4444","#f59e0b","#10b981"],
                text="R²",
                labels={"R²":"R² Score"},
            )
            fig_cv.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_cv.add_hline(y=res["cv_mean"], line_dash="dash", line_color="#f59e0b",
                             annotation_text=f"Mean={res['cv_mean']:.4f}")
            fig_cv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", coloraxis_showscale=False,
                yaxis=dict(gridcolor="#1e293b", range=[0,1.1]),
                height=340, margin=dict(l=0,r=0,t=30,b=0),
            )
            st.plotly_chart(fig_cv, use_container_width=True)

        # Model interpretation
        st.markdown("---")
        st.markdown("#### 💡 Interpretasi Hasil Model")
        r2_pct = res['r2'] * 100
        cv_pct = res['cv_mean'] * 100
        quality = "🟢 Sangat Baik" if r2_pct >= 90 else ("🟡 Baik" if r2_pct >= 75 else "🔴 Perlu Perbaikan")

        st.markdown(f"""<div class="insight-box">
        <b>{quality}</b> — Model menjelaskan <b>{r2_pct:.1f}%</b> variasi harga pasar. 
        Rata-rata prediksi meleset <b>${res['mae']:.4f}</b> per 100 unit currency 
        (sekitar <b>Rp{int(res['mae']*IDR_RATE):,}</b>).
        Cross-validation R² = <b>{cv_pct:.1f}%</b> — model {"konsisten dan generalisasi baik" if res['cv_std'] < 0.05 else "sedikit overfitting, coba kurangi max_depth"}.
        <br><br>
        <b>Fitur paling berpengaruh:</b> {res['df_imp'].iloc[0]['label']} ({res['df_imp'].iloc[0]['importance_pct']:.1f}%), 
        {res['df_imp'].iloc[1]['label']} ({res['df_imp'].iloc[1]['importance_pct']:.1f}%), 
        {res['df_imp'].iloc[2]['label']} ({res['df_imp'].iloc[2]['importance_pct']:.1f}%).
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: PREDIKSI HARGA
    # ══════════════════════════════════════════════════════════════════════
    with ml_tabs[2]:
        if "trained_model" not in st.session_state:
            st.info("⚠️ Latih model terlebih dahulu di tab **⚙️ Konfigurasi & Training**.")
            return

        model = st.session_state["trained_model"]
        res   = st.session_state["train_results"]
        sel_f = res["selected_features"]

        st.markdown("### 🔮 Kalkulator Harga — Berbasis Model Terlatih")
        st.caption(f"Menggunakan model: **{res['model_name']}** | R²={res['r2']:.4f} | Fitur: {len(sel_f)}")

        col1, col2 = st.columns([1,1])

        with col1:
            st.markdown("#### Parameter Input")

            genre_map   = {"MOBA": 0, "Battle Royale": 1, "RPG / Gacha": 2, "Shooter": 3}
            region_map  = {"SEA": 0, "LATAM": 1, "Global_West": 2, "East_Asia": 3, "MENA": 4}
            seg_map     = {"whale": 2, "dolphin": 0, "minnow": 1}
            region_idx  = {"SEA":0.70,"LATAM":0.75,"Global_West":1.00,"East_Asia":1.20,"MENA":0.85}
            region_arpu = {"SEA":0.65,"LATAM":0.70,"Global_West":1.00,"East_Asia":1.30,"MENA":0.80}
            seg_mult    = {"whale":2.8,"dolphin":1.4,"minnow":0.75}
            seg_pct     = {"whale":3,"dolphin":15,"minnow":82}

            genre   = st.selectbox("🎮 Genre Game",
                ["MOBA","Battle Royale","RPG / Gacha","Shooter"],
                help="Tipe game: MOBA (MLBB), Battle Royale (FF/PUBG), RPG/Gacha (Genshin), Shooter (CoD)")
            region  = st.selectbox("🌍 Region Pasar",
                ["SEA","LATAM","Global_West","East_Asia","MENA"],
                format_func=lambda x: {"SEA":"🌏 Asia Tenggara","LATAM":"🌎 Amerika Latin",
                    "Global_West":"🌍 Global/Barat","East_Asia":"🌏 Asia Timur","MENA":"🌍 Timur Tengah"}[x],
                help="Wilayah utama target pasar")
            segment = st.selectbox("🎯 Segmen Pemain",
                ["whale","dolphin","minnow"],
                format_func=lambda x: {"whale":"🐋 Whale (>$100/bln)","dolphin":"🐬 Dolphin ($10-100/bln)","minnow":"🐟 Minnow (<$10/bln)"}[x],
                help="Target segmen berdasarkan pola belanja pemain")

            st.markdown("---")

            mau    = st.slider("👥 MAU (Juta pengguna aktif)", 5.0, 150.0, 50.0, 5.0)
            arpdau = st.slider("💵 ARPDAU (USD)", 0.005, 0.150, 0.025, 0.005,
                               format="$%.3f",
                               help="Rata-rata pendapatan per pengguna aktif per hari. MLBB=0.018, Genshin=0.126")
            age    = st.slider("🎂 Usia Median Pemain", 15, 40, 22, 1,
                               format="%d tahun")
            comp   = st.slider("🏆 Skor Kompetisi", 0, 100, 60, 5,
                               help="0=sangat niche, 100=sangat kompetitif. MLBB=85, Genshin=25")

        with col2:
            st.markdown("#### Hasil Prediksi")

            # Build input vector with same features used in training
            dau_est  = mau * 0.18
            d2m      = dau_est / mau if mau > 0 else 0
            health   = min(mau/130,1)*40 + min(d2m/0.35,1)*30 + 20

            input_vals = {
                "mau_millions":          mau,
                "dau_millions":          round(dau_est,2),
                "arpdau_usd":            arpdau,
                "competition_score":     float(comp),
                "community_health_score":round(health,1),
                "game_median_age":       float(age),
                "dau_to_mau_ratio":      round(d2m,3),
                "region_price_index":    region_idx[region],
                "region_arpu_mult":      region_arpu[region],
                "segment_spend_mult":    seg_mult[segment],
                "segment_player_pct":    seg_pct[segment],
                "genre_encoded":         genre_map[genre],
                "region_encoded":        region_map[region],
                "segment_encoded":       seg_map[segment],
            }

            # Hanya fitur yang dipilih saat training
            X_pred = pd.DataFrame([[input_vals.get(f,0) for f in sel_f]], columns=sel_f)

            try:
                price = float(model.predict(X_pred)[0])
                price = max(price, 0.001)
                low   = price * 0.82
                high  = price * 1.22

                st.markdown(f"""<div class="result-box">
                    <div style="font-size:12px;color:#64748b;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px">Harga Optimal per 100 unit currency</div>
                    <div class="result-price">${price:.3f}</div>
                    <div class="result-sub">≈ Rp {int(price*IDR_RATE):,}</div>
                    <div style="margin-top:16px;padding-top:16px;border-top:1px solid #10b981">
                        <div style="font-size:12px;color:#64748b;margin-bottom:6px">Range Kompetitif</div>
                        <div style="color:#34d399;font-size:16px;font-weight:600">${low:.3f} — ${high:.3f}</div>
                        <div style="color:#64748b;font-size:12px">Rp {int(low*IDR_RATE):,} — Rp {int(high*IDR_RATE):,}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Estimasi harga per tier
                st.markdown("**💰 Estimasi Harga per Tier Pembelian**")
                tiers = [
                    ("🔵 Micro (<$2)", 0.08),
                    ("🟢 Small ($2–10)", 0.30),
                    ("🟡 Medium ($10–30)", 1.00),
                    ("🔴 Large ($30–60)", 3.50),
                    ("🟣 Mega (>$60)", 12.00),
                ]
                tier_df = pd.DataFrame([
                    {
                        "Tier": label,
                        "Harga (USD)": f"${price*mult*100:.2f}",
                        "Harga (IDR)": f"Rp {int(price*mult*100*IDR_RATE):,}",
                    }
                    for label, mult in tiers
                ])
                st.dataframe(tier_df, hide_index=True, use_container_width=True)

                # Strategi
                st.markdown("**🎯 Strategi yang Disarankan**")
                if price < 0.80:
                    strategy = "**Penetrasi Pasar**: Harga agresif untuk maksimalkan volume. Fokus pada bundle harian/mingguan. Cocok untuk konversi minnow → dolphin."
                elif price < 2.5:
                    strategy = "**Kompetitif**: Harga di tengah pasar. Tambahkan value-added (bonus currency, event akses) untuk diferensiasi tanpa turunkan harga."
                elif price < 8:
                    strategy = "**Premium Terkontrol**: Harga di atas rata-rata, didukung kompetisi rendah/usia dewasa. Pastikan konten eksklusif mendukung harga premium."
                else:
                    strategy = "**Luxury Premium**: Segmen whale di pasar high-income. Jaga eksklusivitas — jangan turunkan harga. Limited bundles dan VIP membership efektif."

                st.info(strategy)

            except Exception as e:
                st.error(f"Prediksi gagal: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4: PANDUAN FITUR
    # ══════════════════════════════════════════════════════════════════════
    with ml_tabs[3]:
        st.markdown("### 📖 Panduan Fitur & Istilah ML")

        glossary = [
            ("MAU (Monthly Active Users)", "Jumlah pengguna unik yang membuka game minimal sekali dalam sebulan. Komunitas lebih besar → persaingan harga lebih ketat.", "MLBB: 110 Juta | Free Fire: 130 Juta"),
            ("DAU (Daily Active Users)", "Pengguna yang membuka game setiap hari. Rasio DAU/MAU tinggi = komunitas lebih loyal dan engaged.", "Free Fire DAU: 33 Juta (25% dari MAU)"),
            ("ARPDAU", "Average Revenue Per Daily Active User — rata-rata USD yang dihasilkan setiap 1 pengguna aktif per hari.", "MLBB: $0.018 | Genshin: $0.126"),
            ("Competition Score", "Skor 0–100 dari: jumlah turnamen aktif + prize pool esports + peak viewers. Tinggi = pasar sangat kompetitif.", "Elite ≥80 | High 60–79 | Medium 40–59 | Low <40"),
            ("Community Health Score", "Skor 0–100 gabungan MAU (40%) + DAU/MAU ratio (30%) + downloads (30%). Komunitas sehat → tahan kenaikan harga.", "Free Fire: 87.8 | MLBB: 66.7"),
            ("Region Price Index", "Faktor pengali daya beli regional vs Global/Barat (1.0). SEA 0.70 = harga pasar SEA 30% lebih murah.", "SEA: 0.70 | East Asia: 1.20 | Global: 1.00"),
            ("Segment Spend Multiplier", "Pengali kemampuan bayar per segmen. Whale 2.8× → rela bayar 2.8× harga rata-rata.", "Whale: 2.8× | Dolphin: 1.4× | Minnow: 0.75×"),
            ("R² Score", "Seberapa baik model menjelaskan variasi harga. 1.0 = sempurna. ≥0.90 = sangat baik untuk bisnis.", "Model kami: 0.977 (97.7% variasi terprediksi)"),
            ("MAE (Mean Absolute Error)", "Rata-rata selisih antara harga prediksi dan aktual. Dalam USD per 100 unit currency.", "MAE $0.45 = prediksi meleset rata-rata Rp7.313 per 100 unit"),
            ("Cross-Validation", "Teknik uji model dengan 5 iterasi data berbeda. CV R² tinggi = model generalisasi baik, tidak hafal data.", "CV R² 0.977 ± 0.02 = sangat konsisten"),
            ("Train/Test Split 80/20", "80% data untuk melatih model, 20% untuk menguji pada data yang belum pernah dilihat. Standar industri.", "90 baris: 72 train + 18 test"),
            ("Optimal Price per 100 USD", "Target variable — harga optimal per 100 unit currency in-game dalam USD. Ini yang diprediksi model.", "MLBB SEA Dolphin: $0.45 | Genshin EA Whale: $24.9"),
        ]

        col1, col2 = st.columns(2)
        for i, (term, defn, example) in enumerate(glossary):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.6);border:1px solid #1e293b;
                    border-radius:10px;padding:14px 16px;margin-bottom:12px">
                    <div style="font-family:monospace;font-size:12px;color:#60a5fa;margin-bottom:6px;font-weight:700">{term}</div>
                    <div style="font-size:12px;color:#94a3b8;line-height:1.5;margin-bottom:6px">{defn}</div>
                    <div style="font-size:11px;color:#475569;font-style:italic">📌 {example}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    data = load_data()
    page = render_sidebar()

    if "Business Intelligence" in page:
        page_bi(data)
    else:
        page_ml(data)


if __name__ == "__main__":
    main()
