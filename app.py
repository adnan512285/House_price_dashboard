# ============================================================
#   ENTERPRISE HOUSE PRICE PREDICTION DASHBOARD
#   Built for: California Housing Dataset + RandomForest Model
#   Design: Dark Glassmorphism + Premium Business UI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import io
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HouseLens AI · Price Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — Dark Glassmorphism Premium
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root variables ── */
:root {
    --bg-deep:      #050810;
    --bg-mid:       #0b0f1e;
    --bg-card:      rgba(16, 22, 45, 0.72);
    --border:       rgba(99, 179, 237, 0.18);
    --border-glow:  rgba(99, 179, 237, 0.45);
    --gold:         #f0b429;
    --gold-dim:     rgba(240, 180, 41, 0.15);
    --cyan:         #63b3ed;
    --cyan-dim:     rgba(99, 179, 237, 0.12);
    --teal:         #4fd1c5;
    --violet:       #9f7aea;
    --rose:         #fc8181;
    --green:        #68d391;
    --text-primary: #e8eaf0;
    --text-muted:   #8892a4;
    --font-head:    'Syne', sans-serif;
    --font-body:    'DM Sans', sans-serif;
    --radius:       16px;
    --radius-sm:    10px;
    --glow-cyan:    0 0 30px rgba(99, 179, 237, 0.25);
    --glow-gold:    0 0 30px rgba(240, 180, 41, 0.25);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,179,237,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(159,122,234,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(79,209,197,0.04) 0%, transparent 60%);
    background-attachment: fixed;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c1a 0%, #0d1228 60%, #060a16 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
}
/* Force all sidebar text bright */
[data-testid="stSidebar"] * {
    color: #e8eaf0 !important;
}
/* Navigation section heading */
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stRadio > label p {
    color: #63b3ed !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
/* Each nav option */
[data-testid="stSidebar"] .stRadio > div > label,
[data-testid="stSidebar"] .stRadio > div > label p,
[data-testid="stSidebar"] .stRadio label div p {
    font-size: 0.97rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: #f0f4ff !important;
    padding: 9px 14px !important;
    border-radius: var(--radius-sm) !important;
    transition: background 0.2s ease, color 0.2s ease !important;
    line-height: 1.5 !important;
}
/* Hover state */
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99,179,237,0.12) !important;
    color: #ffffff !important;
}
/* File uploader labels */
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stFileUploader label p {
    color: #c8d0e0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ── Hide default header ── */
#MainMenu, footer, header {visibility: hidden;}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 10px; }

/* ── Glass card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 30px;
    backdrop-filter: blur(18px) saturate(1.4);
    -webkit-backdrop-filter: blur(18px) saturate(1.4);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--glow-cyan), 0 8px 40px rgba(0,0,0,0.5);
}

/* ── Metric card ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    backdrop-filter: blur(14px);
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.gold::before  { background: linear-gradient(90deg, transparent, var(--gold), transparent); }
.metric-card.cyan::before  { background: linear-gradient(90deg, transparent, var(--cyan), transparent); }
.metric-card.teal::before  { background: linear-gradient(90deg, transparent, var(--teal), transparent); }
.metric-card.violet::before{ background: linear-gradient(90deg, transparent, var(--violet), transparent); }
.metric-card.green::before { background: linear-gradient(90deg, transparent, var(--green), transparent); }
.metric-card.rose::before  { background: linear-gradient(90deg, transparent, var(--rose), transparent); }
.metric-card:hover { transform: translateY(-4px); box-shadow: var(--glow-cyan); }

.metric-label {
    font-family: var(--font-body);
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.metric-value {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e8eaf0 30%, var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-delta {
    font-size: 0.78rem;
    margin-top: 6px;
    color: var(--green);
}
.metric-icon {
    font-size: 1.6rem;
    margin-bottom: 10px;
}

/* ── Section header ── */
.section-header {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.section-sub {
    font-family: var(--font-body);
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 28px;
}
.accent-line {
    width: 48px;
    height: 3px;
    background: linear-gradient(90deg, var(--cyan), var(--teal));
    border-radius: 4px;
    margin-bottom: 10px;
}

/* ── Page hero ── */
.hero-title {
    font-family: var(--font-head);
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.05;
    background: linear-gradient(135deg, #ffffff 0%, var(--cyan) 50%, var(--teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 300;
    margin-top: 8px;
    line-height: 1.7;
}

/* ── Badge / pill ── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-cyan   { background: var(--cyan-dim);  color: var(--cyan);  border: 1px solid rgba(99,179,237,0.3); }
.badge-gold   { background: var(--gold-dim);  color: var(--gold);  border: 1px solid rgba(240,180,41,0.3); }
.badge-green  { background: rgba(104,211,145,0.12); color: var(--green); border: 1px solid rgba(104,211,145,0.3); }
.badge-violet { background: rgba(159,122,234,0.12); color: var(--violet); border: 1px solid rgba(159,122,234,0.3); }

/* ── Global text brightness fix ── */
p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th {
    color: #e8eaf0;
}

/* ── All Streamlit labels (number input, selectbox, slider, file uploader) ── */
.stNumberInput label,
.stNumberInput label p,
.stSelectbox label,
.stSelectbox label p,
.stSlider label,
.stSlider label p,
.stFileUploader label,
.stFileUploader label p,
.stTextInput label,
.stTextInput label p,
.stRadio label,
.stCheckbox label,
.stMarkdown p,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
    color: #dde3f0 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* ── Number input field ── */
.stNumberInput > div > div > input {
    background: rgba(16, 22, 45, 0.9) !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: #f0f4ff !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
.stNumberInput > div > div > input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.2) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(16, 22, 45, 0.9) !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: #f0f4ff !important;
}
.stSelectbox > div > div > div {
    color: #f0f4ff !important;
    font-weight: 500 !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: var(--cyan) !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider [data-testid="stThumbValue"] {
    color: #c8d0e0 !important;
}

/* ── File uploader box ── */
[data-testid="stFileUploader"] {
    background: rgba(16,22,45,0.6) !important;
    border: 1px dashed rgba(99,179,237,0.4) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] section > div {
    color: #c8d0e0 !important;
}
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploaderDropzoneInstructions"] div span,
[data-testid="stFileUploaderDropzoneInstructions"] div small {
    color: #a8b4c8 !important;
    font-size: 0.82rem !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(99,179,237,0.15) !important;
    border: 1px solid rgba(99,179,237,0.4) !important;
    border-radius: 8px !important;
    color: #63b3ed !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ── Form container ── */
[data-testid="stForm"] {
    background: rgba(16,22,45,0.5) !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
}

/* ── Prediction section text ── */
.prediction-result p,
.prediction-result div,
.prediction-result span {
    color: #c8d0e0 !important;
}

/* ── Info box text ── */
.info-box {
    color: #c8d0e0 !important;
}
.info-box b, .info-box strong {
    color: #e8eaf0 !important;
}

/* ── Section sub-heading ── */
.section-sub {
    color: #a8b4c8 !important;
    font-size: 0.92rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a4f7a 0%, #0d6e8a 100%) !important;
    border: 1px solid var(--cyan) !important;
    border-radius: var(--radius-sm) !important;
    color: #ffffff !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 10px 24px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(13,110,138,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,179,237,0.4) !important;
}
.stButton > button p {
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #2d4a22 0%, #276749 100%) !important;
    border: 1px solid var(--green) !important;
    border-radius: var(--radius-sm) !important;
    color: white !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
}
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.stDataFrame { border-radius: var(--radius-sm) !important; }
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px !important;
}
[data-testid="stMetricLabel"] p { color: #a8b4c8 !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"]   { color: #f0f4ff !important; font-weight: 700 !important; }

/* ── Divider ── */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
    margin: 32px 0;
}

/* ── Prediction result ── */
.prediction-result {
    background: linear-gradient(135deg, rgba(99,179,237,0.12) 0%, rgba(79,209,197,0.08) 100%);
    border: 1px solid var(--border-glow);
    border-radius: var(--radius);
    padding: 36px;
    text-align: center;
    box-shadow: var(--glow-cyan);
}
.prediction-price {
    font-family: var(--font-head);
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--gold) 0%, #f6c860 50%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
}

/* ── Info box ── */
.info-box {
    background: rgba(99,179,237,0.08);
    border-left: 3px solid var(--cyan);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: var(--text-muted);
}

/* ── Sidebar brand ── */
.sidebar-brand {
    font-family: var(--font-head);
    font-size: 1.35rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--cyan), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    padding: 4px 0 2px 0;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ── Table styling ── */
.stDataFrame > div { border-radius: var(--radius-sm) !important; background: var(--bg-card) !important; }

/* ── Alert overrides ── */
.stAlert { border-radius: var(--radius-sm) !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: var(--radius-sm); padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: var(--text-muted) !important; font-family: var(--font-body) !important; }
.stTabs [aria-selected="true"] { background: var(--cyan-dim) !important; color: var(--cyan) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER: Plotly dark theme
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,30,0.6)",
        font=dict(family="DM Sans, sans-serif", color="#8892a4", size=12),
        title=dict(font=dict(family="Syne, sans-serif", color="#e8eaf0", size=16)),
        xaxis=dict(gridcolor="rgba(99,179,237,0.1)", zerolinecolor="rgba(99,179,237,0.15)", linecolor="rgba(99,179,237,0.15)"),
        yaxis=dict(gridcolor="rgba(99,179,237,0.1)", zerolinecolor="rgba(99,179,237,0.15)", linecolor="rgba(99,179,237,0.15)"),
        legend=dict(bgcolor="rgba(11,15,30,0.8)", bordercolor="rgba(99,179,237,0.2)", borderwidth=1),
        colorway=["#63b3ed","#4fd1c5","#f0b429","#9f7aea","#fc8181","#68d391","#76e4f7","#fbb6ce"],
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

COLOR_SEQ = px.colors.sequential.Blues_r
DIVERGING  = px.colors.diverging.RdBu

def apply_dark(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,30,0.6)",
        font=dict(family="DM Sans, sans-serif", color="#8892a4", size=12),
        title_font=dict(family="Syne, sans-serif", color="#e8eaf0", size=16),
        legend=dict(bgcolor="rgba(11,15,30,0.8)", bordercolor="rgba(99,179,237,0.2)", borderwidth=1),
        margin=dict(l=40, r=20, t=54, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(99,179,237,0.1)", zerolinecolor="rgba(99,179,237,0.15)", linecolor="rgba(99,179,237,0.2)")
    fig.update_yaxes(gridcolor="rgba(99,179,237,0.1)", zerolinecolor="rgba(99,179,237,0.15)", linecolor="rgba(99,179,237,0.2)")
    return fig


# ─────────────────────────────────────────────
#  SESSION STATE — Data & Model
# ─────────────────────────────────────────────
if "df_raw" not in st.session_state:
    st.session_state.df_raw        = None
if "model"    not in st.session_state:
    st.session_state.model         = None
if "scaler"   not in st.session_state:
    st.session_state.scaler        = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols  = None
if "train_data" not in st.session_state:
    st.session_state.train_data    = None
if "metrics"  not in st.session_state:
    st.session_state.metrics       = {}
if "y_test"   not in st.session_state:
    st.session_state.y_test        = None
if "y_pred"   not in st.session_state:
    st.session_state.y_pred        = None


# ─────────────────────────────────────────────
#  DATA PROCESSING — mirrors training notebook
# ─────────────────────────────────────────────
OCEAN_CATS = [
    "ocean_proximity_<1H OCEAN",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN",
]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.dropna(inplace=True)
    d["total_rooms"]    = np.log(d["total_rooms"]    + 1)
    d["total_bedrooms"] = np.log(d["total_bedrooms"] + 1)
    d["population"]     = np.log(d["population"]     + 1)
    d["households"]     = np.log(d["households"]     + 1)
    d = pd.get_dummies(d, columns=["ocean_proximity"], dtype=int)
    for col in OCEAN_CATS:
        if col not in d.columns:
            d[col] = 0
    d["bedroom_ratio"]    = d["total_bedrooms"] / d["total_rooms"]
    d["household_rooms"]  = d["total_rooms"]    / d["households"]
    return d

@st.cache_data(show_spinner=False)
def train_model(csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # ── Drop NaNs BEFORE splitting so both splits are clean ──
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    x = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    # ── Preprocess train set (no dropna needed — already clean) ──
    train_data = x_train.join(y_train)
    train_data = preprocess(train_data)

    x_tr = train_data.drop("median_house_value", axis=1)
    y_tr = train_data["median_house_value"]
    feature_cols = list(x_tr.columns)

    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr)

    model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
    model.fit(x_tr_scaled, y_tr)

    # ── Preprocess test set; use y_test from processed frame (row-count stays equal) ──
    test_data = x_test.join(y_test)
    test_data = preprocess(test_data)          # dropna inside preprocess is now a no-op
    y_test_clean = test_data["median_house_value"]
    x_te = test_data.drop("median_house_value", axis=1)
    x_te = x_te.reindex(columns=feature_cols, fill_value=0)
    x_te_scaled = scaler.transform(x_te)
    y_pred = model.predict(x_te_scaled)

    metrics = {
        "r2":   r2_score(y_test_clean, y_pred),
        "mae":  mean_absolute_error(y_test_clean, y_pred),
        "mse":  mean_squared_error(y_test_clean, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test_clean, y_pred)),
    }
    return df, model, scaler, feature_cols, train_data, metrics, y_test_clean, y_pred


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 8px 0;">
        <div class="sidebar-brand">🏛️ HouseLens AI</div>
        <div class="sidebar-tagline">Price Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:#8892a4;margin-bottom:8px;">Navigation</p>', unsafe_allow_html=True)
    page = st.radio(
        label="Navigation",
        options=[
            "🏠  Home",
            "📊  Data Analysis",
            "📈  Visualizations",
            "🔮  Prediction",
            "🎯  Model Performance",
            "🔬  Advanced Analytics",
            "📤  Export",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown('''
    <p style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;
       color:#63b3ed;font-weight:700;margin-bottom:10px;">Upload Assets</p>
    ''', unsafe_allow_html=True)

    st.markdown('''<style>
    /* Upload section labels in sidebar */
    [data-testid="stSidebar"] .stFileUploader > label p {
        color: #dde3f0 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
        color: #8892a4 !important;
        font-size: 0.75rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(99,179,237,0.06) !important;
        border: 1px dashed rgba(99,179,237,0.35) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        color: #63b3ed !important;
        background: rgba(99,179,237,0.12) !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }
    </style>''', unsafe_allow_html=True)

    csv_file    = st.file_uploader("📂 Dataset (CSV)", type=["csv"])
    model_file  = st.file_uploader("🤖 Model (PKL)",   type=["pkl"])
    scaler_file = st.file_uploader("⚖️ Scaler (PKL)",  type=["pkl"])

    # Auto-train if CSV uploaded but no PKL
    if csv_file:
        with st.spinner("Training model on your dataset…"):
            raw_bytes = csv_file.read()
            (
                st.session_state.df_raw,
                st.session_state.model,
                st.session_state.scaler,
                st.session_state.feature_cols,
                st.session_state.train_data,
                st.session_state.metrics,
                st.session_state.y_test,
                st.session_state.y_pred,
            ) = train_model(raw_bytes)

        if model_file:
            try:
                st.session_state.model  = pickle.load(model_file)
            except Exception:
                pass
        if scaler_file:
            try:
                st.session_state.scaler = pickle.load(scaler_file)
            except Exception:
                pass

        st.success(f"✅ Ready · {len(st.session_state.df_raw):,} records")

    st.markdown("""
    <div style="margin-top:32px;padding:14px;background:rgba(99,179,237,0.07);border:1px solid rgba(99,179,237,0.15);border-radius:10px;">
        <p style="font-size:0.72rem;color:#8892a4;margin:0;line-height:1.6;">
            <b style="color:#63b3ed;">v2.0</b> — RandomForest · California Housing<br>
            Tuned · StandardScaler · 20% test split
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  GUARD: require dataset
# ─────────────────────────────────────────────
def require_data():
    if st.session_state.df_raw is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:60px 30px;">
            <div style="font-size:3rem;margin-bottom:16px;">📂</div>
            <div class="section-header" style="font-size:1.4rem;">No Dataset Loaded</div>
            <p class="hero-sub">Upload your <b>housing.csv</b> in the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        return False
    return True


# ─────────────────────────────────────────────
#  PAGE HELPER: section title
# ─────────────────────────────────────────────
def section_title(title, sub=""):
    st.markdown(f"""
    <div class="accent-line"></div>
    <div class="section-header">{title}</div>
    <div class="section-sub">{sub}</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 1 · HOME
# ══════════════════════════════════════════════
if page == "🏠  Home":
    # Hero
    st.markdown("""
    <div class="glass-card" style="margin-bottom:28px; background: linear-gradient(135deg, rgba(16,22,45,0.85) 0%, rgba(8,14,30,0.9) 100%);">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
            <span class="badge badge-cyan">AI-Powered</span>
            <span class="badge badge-gold">Enterprise</span>
            <span class="badge badge-green">Production</span>
        </div>
        <div class="hero-title">House Price<br>Intelligence Platform</div>
        <div class="hero-sub">
            End-to-end machine learning solution for California real estate valuation.<br>
            Powered by <b style="color:#63b3ed;">Random Forest</b> with hyperparameter tuning &amp; advanced feature engineering.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_raw is not None:
        df  = st.session_state.df_raw
        met = st.session_state.metrics

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "gold",   "🏘️", "Total Properties",   f"{len(df):,}",       "California Housing Dataset"),
            (c2, "cyan",   "📐", "Features",            f"{len(df.columns)-1}","Input variables"),
            (c3, "teal",   "🎯", "Model Accuracy",      f"{met['r2']:.1%}",   "R² Score on test set"),
            (c4, "violet", "💰", "Avg Error (MAE)",     f"${met['mae']:,.0f}", "Per-house precision"),
        ]
        for col, color, icon, label, val, sub in cards:
            with col:
                st.markdown(f"""
                <div class="metric-card {color}">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-delta">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # Two-column info section
        left, right = st.columns(2)

        with left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section_title("📋 Dataset Overview", "California Housing — 1990 Census")
            info_rows = [
                ("Source",          "California Census 1990"),
                ("Target Variable", "Median House Value ($)"),
                ("Samples",         f"{len(df):,}"),
                ("Features",        ", ".join([c for c in df.columns if c != 'median_house_value'][:4]) + "…"),
                ("Missing Handled", "Rows dropped"),
                ("Ocean Proximity", "One-hot encoded"),
                ("Log Transform",   "rooms · bedrooms · pop · households"),
                ("Engineered",      "bedroom_ratio · household_rooms"),
            ]
            for k, v in info_rows:
                st.markdown(f'<div class="info-box"><b style="color:#63b3ed;">{k}:</b> {v}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            section_title("🤖 Model Architecture", "Tuned RandomForestRegressor")
            model_rows = [
                ("Algorithm",       "Random Forest Regressor"),
                ("Estimators",      "100 trees"),
                ("Hyperparameter",  "RandomizedSearchCV (4 combos)"),
                ("Scaler",          "StandardScaler"),
                ("Train/Test Split","80% / 20%"),
                ("R² Score",        f"{met['r2']:.4f}"),
                ("MAE",             f"${met['mae']:,.2f}"),
                ("RMSE",            f"${met['rmse']:,.2f}"),
            ]
            for k, v in model_rows:
                st.markdown(f'<div class="info-box"><b style="color:#4fd1c5;">{k}:</b> {v}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Dataset head preview
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_title("👁️ Dataset Preview", f"First 8 rows of {len(df):,} records")
        st.dataframe(df.head(8).style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:60px;">
            <div style="font-size:4rem;margin-bottom:20px;">🏛️</div>
            <div class="hero-title" style="font-size:2rem;">Welcome to HouseLens AI</div>
            <div class="hero-sub">Upload your <b style="color:#63b3ed;">housing.csv</b> from the sidebar to unlock the full dashboard.</div>
        </div>
        """, unsafe_allow_html=True)

        feat_cols = st.columns(3)
        feats = [
            ("📊", "Data Analysis",    "Missing values, distributions, correlations"),
            ("🔮", "Live Prediction",  "Real-time house price estimation"),
            ("🎯", "Model Metrics",    "R², MAE, RMSE with visual diagnostics"),
        ]
        for col, (icon, title, desc) in zip(feat_cols, feats):
            with col:
                st.markdown(f"""
                <div class="metric-card cyan" style="text-align:center;">
                    <div style="font-size:2rem;">{icon}</div>
                    <div style="font-family:'Syne';font-weight:700;font-size:1rem;color:#e8eaf0;margin:8px 0 4px;">{title}</div>
                    <div style="font-size:0.82rem;color:#8892a4;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 2 · DATA ANALYSIS
# ══════════════════════════════════════════════
elif page == "📊  Data Analysis":
    if not require_data():
        st.stop()

    df = st.session_state.df_raw

    st.markdown('<div class="hero-title" style="font-size:2.2rem;">Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep-dive into the dataset structure, quality, and statistics.</div><br>', unsafe_allow_html=True)

    tabs = st.tabs(["📋 Overview", "🔍 Quality Check", "📈 Statistics", "🌡️ Correlations"])

    # ── Tab 1: Overview
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        stats = [
            (c1, "cyan",   "Rows",     f"{len(df):,}"),
            (c2, "gold",   "Columns",  f"{len(df.columns)}"),
            (c3, "teal",   "Numeric",  f"{len(df.select_dtypes(include=np.number).columns)}"),
            (c4, "violet", "Categorical", f"{len(df.select_dtypes(include='object').columns)}"),
        ]
        for col, clr, lbl, val in stats:
            with col:
                st.markdown(f'<div class="metric-card {clr}"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_title("Full Dataset Preview", "Scroll to explore all rows and columns")
        st.dataframe(df.style.background_gradient(cmap="Blues", axis=None, subset=df.select_dtypes(include=np.number).columns.tolist()), use_container_width=True, height=380)
        st.markdown('</div>', unsafe_allow_html=True)

        # Column types table
        st.markdown('<br><div class="glass-card">', unsafe_allow_html=True)
        section_title("Column Schema", "Data types and non-null counts")
        schema = pd.DataFrame({
            "Column": df.columns,
            "Type":   df.dtypes.astype(str).values,
            "Non-Null Count": df.count().values,
            "Null Count": df.isnull().sum().values,
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 2: Quality
    with tabs[1]:
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ["Feature", "Missing"]
        null_counts["Missing %"] = (null_counts["Missing"] / len(df) * 100).round(2)
        null_counts["Status"]    = null_counts["Missing"].apply(lambda x: "✅ Clean" if x == 0 else f"⚠️ {x} missing")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_title("Missing Value Analysis", "Per-column null audit")
        st.dataframe(null_counts, use_container_width=True)

        fig_null = px.bar(
            null_counts[null_counts["Missing"] > 0] if null_counts["Missing"].sum() > 0 else null_counts.head(5),
            x="Feature", y="Missing %", color="Missing %",
            color_continuous_scale="Blues",
            title="Missing Values (%) by Feature",
        )
        apply_dark(fig_null)
        st.plotly_chart(fig_null, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Duplicate Rows</div><div class="metric-value">{df.duplicated().sum()}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            total_null = df.isnull().sum().sum()
            st.markdown(f'<div class="metric-label">Total Missing Cells</div><div class="metric-value">{total_null}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 3: Statistics
    with tabs[2]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_title("Descriptive Statistics", "Mean, std, quartiles for all numeric columns")
        desc = df.describe().T.round(3)
        st.dataframe(desc.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        col_sel = st.selectbox("Select Feature for Distribution", num_cols)

        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(df, y=col_sel, title=f"Box Plot — {col_sel}", color_discrete_sequence=["#63b3ed"])
            apply_dark(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            fig_hist = px.histogram(df, x=col_sel, nbins=50, title=f"Histogram — {col_sel}", color_discrete_sequence=["#4fd1c5"])
            apply_dark(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── Tab 4: Correlations
    with tabs[3]:
        num_df = df.select_dtypes(include=np.number)
        corr   = num_df.corr()

        fig_heat = px.imshow(
            corr.round(2),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Pearson Correlation Matrix",
        )
        apply_dark(fig_heat)
        fig_heat.update_traces(textfont_size=10)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Top correlations with target
        if "median_house_value" in num_df.columns:
            corr_target = corr["median_house_value"].drop("median_house_value").sort_values(key=abs, ascending=False)
            fig_bar = px.bar(
                x=corr_target.values, y=corr_target.index, orientation="h",
                color=corr_target.values, color_continuous_scale="RdBu_r",
                title="Feature Correlation with Median House Value",
            )
            apply_dark(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 3 · VISUALIZATIONS
# ══════════════════════════════════════════════
elif page == "📈  Visualizations":
    if not require_data():
        st.stop()

    df         = st.session_state.df_raw
    train_data = st.session_state.train_data
    model      = st.session_state.model
    feature_cols = st.session_state.feature_cols

    st.markdown('<div class="hero-title" style="font-size:2.2rem;">Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Interactive Plotly charts — histograms, scatter maps, violin, KDE, pair plots & more.</div><br>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    tabs = st.tabs(["🗺️ Geo Map", "📊 Distributions", "🎻 Violin & Box", "📉 Scatter", "🌡️ Feature Importance", "💎 Price Trends"])

    # ── Geo Map
    with tabs[0]:
        section_title("Geographic Price Map", "California housing prices by location")
        fig_map = px.scatter(
            df, x="longitude", y="latitude",
            color="median_house_value",
            size="population",
            size_max=10,
            color_continuous_scale="plasma",
            opacity=0.7,
            hover_data=["median_house_value", "population", "households"],
            title="California House Prices (lon/lat · sized by population)",
        )
        apply_dark(fig_map)
        fig_map.update_layout(height=550)
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Distributions
    with tabs[1]:
        section_title("Feature Distributions", "Histograms + KDE overlays")
        c1, c2 = st.columns(2)
        with c1:
            feat1 = st.selectbox("Feature (left)", num_cols, index=0)
        with c2:
            feat2 = st.selectbox("Feature (right)", num_cols, index=min(1, len(num_cols)-1))

        col1, col2 = st.columns(2)
        for col, feat in [(col1, feat1), (col2, feat2)]:
            with col:
                fig = px.histogram(df, x=feat, nbins=60, marginal="violin", color_discrete_sequence=["#63b3ed"], title=f"Distribution — {feat}")
                apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)

        # All numeric histograms in grid
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        section_title("All Numeric Features Overview")
        n = len(num_cols)
        cols_per_row = 3
        rows = (n + cols_per_row - 1) // cols_per_row
        for r in range(rows):
            row_cols = st.columns(cols_per_row)
            for i, col in enumerate(row_cols):
                idx = r * cols_per_row + i
                if idx < n:
                    with col:
                        fig = px.histogram(df, x=num_cols[idx], nbins=40, color_discrete_sequence=["#4fd1c5"])
                        fig.update_layout(margin=dict(l=20,r=10,t=40,b=20), height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(11,15,30,0.6)", font_color="#8892a4", title_text=num_cols[idx], title_font_color="#e8eaf0")
                        st.plotly_chart(fig, use_container_width=True)

    # ── Violin & Box
    with tabs[2]:
        section_title("Violin & Box Plots", "Distribution shape by ocean proximity")
        feat_vio = st.selectbox("Feature", num_cols, key="vio_feat")

        c1, c2 = st.columns(2)
        with c1:
            if "ocean_proximity" in df.columns:
                fig_vio = px.violin(df, x="ocean_proximity", y=feat_vio, box=True, color="ocean_proximity", title=f"Violin — {feat_vio} by Ocean Proximity")
                apply_dark(fig_vio)
                st.plotly_chart(fig_vio, use_container_width=True)
        with c2:
            if "ocean_proximity" in df.columns:
                fig_box = px.box(df, x="ocean_proximity", y=feat_vio, color="ocean_proximity", title=f"Box Plot — {feat_vio} by Ocean Proximity")
                apply_dark(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)

    # ── Scatter
    with tabs[3]:
        section_title("Scatter Explorer", "Bi-variate relationships colored by house value")
        c1, c2 = st.columns(2)
        with c1:
            sx = st.selectbox("X axis", num_cols, index=num_cols.index("median_income") if "median_income" in num_cols else 0)
        with c2:
            sy = st.selectbox("Y axis", num_cols, index=num_cols.index("median_house_value") if "median_house_value" in num_cols else 1)

        fig_sc = px.scatter(
            df, x=sx, y=sy,
            color="median_house_value" if "median_house_value" in df.columns else None,
            color_continuous_scale="plasma",
            opacity=0.55,
            trendline="ols",
            title=f"{sx} vs {sy}",
        )
        apply_dark(fig_sc)
        fig_sc.update_layout(height=500)
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Feature Importance
    with tabs[4]:
        section_title("Feature Importance", "Top drivers of house price (Random Forest)")
        if model is not None and feature_cols is not None:
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Feature":    feature_cols,
                "Importance": importances,
            }).sort_values("Importance", ascending=True).tail(15)

            fig_fi = px.bar(
                feat_imp_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                title="Top 15 Features by Importance Score",
            )
            apply_dark(fig_fi)
            fig_fi.update_layout(height=500)
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Price Trends
    with tabs[5]:
        section_title("Price Trends", "How price varies across dataset dimensions")
        if "ocean_proximity" in df.columns:
            avg_price = df.groupby("ocean_proximity")["median_house_value"].mean().reset_index()
            avg_price.columns = ["Ocean Proximity", "Avg Price"]
            fig_trend = px.bar(avg_price, x="Ocean Proximity", y="Avg Price", color="Avg Price",
                               color_continuous_scale="Blues", title="Average House Price by Ocean Proximity")
            apply_dark(fig_trend)
            st.plotly_chart(fig_trend, use_container_width=True)

        if "median_income" in df.columns and "median_house_value" in df.columns:
            df["income_band"] = pd.cut(df["median_income"], bins=6, labels=["Very Low","Low","Medium","Medium-High","High","Very High"])
            income_avg = df.groupby("income_band")["median_house_value"].mean().reset_index()
            fig_inc = px.line(income_avg, x="income_band", y="median_house_value", markers=True,
                              title="Avg House Price by Income Band", color_discrete_sequence=["#63b3ed"])
            apply_dark(fig_inc)
            st.plotly_chart(fig_inc, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 4 · PREDICTION
# ══════════════════════════════════════════════
elif page == "🔮  Prediction":
    if not require_data():
        st.stop()

    df           = st.session_state.df_raw
    model        = st.session_state.model
    scaler       = st.session_state.scaler
    feature_cols = st.session_state.feature_cols

    st.markdown('<div class="hero-title" style="font-size:2.2rem;">Live Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Configure property attributes below and get an instant AI-powered valuation.</div><br>', unsafe_allow_html=True)

    # ── Prediction form custom CSS
    st.markdown("""
    <style>
    /* Prediction form — all labels bright white */
    [data-testid="stForm"] label,
    [data-testid="stForm"] label p,
    [data-testid="stForm"] [data-testid="stWidgetLabel"] p,
    [data-testid="stForm"] .stNumberInput label p,
    [data-testid="stForm"] .stSelectbox label p,
    [data-testid="stForm"] .stSlider label p {
        color: #ffffff !important;
        font-size: 0.93rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }
    /* Input boxes inside form */
    [data-testid="stForm"] input[type="number"],
    [data-testid="stForm"] input {
        color: #f0f4ff !important;
        background: rgba(10,16,38,0.95) !important;
        border: 1px solid rgba(99,179,237,0.4) !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    /* Selectbox inside form */
    [data-testid="stForm"] .stSelectbox > div > div {
        color: #f0f4ff !important;
        background: rgba(10,16,38,0.95) !important;
        border: 1px solid rgba(99,179,237,0.4) !important;
    }
    /* Slider value text */
    [data-testid="stForm"] [data-testid="stTickBarMin"],
    [data-testid="stForm"] [data-testid="stTickBarMax"] {
        color: #a8b4c8 !important;
    }
    /* Submit button */
    [data-testid="stForm"] .stFormSubmitButton button {
        background: linear-gradient(135deg, #1a5276 0%, #0d7a8a 100%) !important;
        border: 1px solid #63b3ed !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        border-radius: 10px !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 4px 24px rgba(99,179,237,0.35) !important;
    }
    [data-testid="stForm"] .stFormSubmitButton button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stForm"] .stFormSubmitButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(99,179,237,0.5) !important;
    }
    /* Column headers inside form */
    [data-testid="stForm"] .glass-card .section-header {
        color: #e8eaf0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('''
        <div style="background:rgba(16,22,45,0.7);border:1px solid rgba(99,179,237,0.2);border-radius:16px;padding:28px 30px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                <span style="font-size:1.5rem;">🏘️</span>
                <span style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#e8eaf0;">Property Attributes</span>
            </div>
            <p style="color:#a8b4c8;font-size:0.88rem;margin:0 0 20px 0;">Fill in the property details below to get an instant AI-powered valuation</p>
            <div style="width:48px;height:3px;background:linear-gradient(90deg,#63b3ed,#4fd1c5);border-radius:4px;margin-bottom:24px;"></div>
        </div>
        ''', unsafe_allow_html=True)

        # ── Column group labels
        lc, mc, rc = st.columns(3)
        with lc:
            st.markdown('<p style="color:#63b3ed;font-size:0.75rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">📍 Location</p>', unsafe_allow_html=True)
        with mc:
            st.markdown('<p style="color:#4fd1c5;font-size:0.75rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">🏠 Property Size</p>', unsafe_allow_html=True)
        with rc:
            st.markdown('<p style="color:#f0b429;font-size:0.75rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">💰 Demographics</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            longitude          = st.number_input("Longitude",          value=float(df["longitude"].median()),        format="%.4f")
            latitude           = st.number_input("Latitude",           value=float(df["latitude"].median()),         format="%.4f")
            housing_median_age = st.slider("Housing Median Age (yrs)", min_value=1, max_value=52, value=28)
        with c2:
            total_rooms        = st.number_input("Total Rooms",        value=int(df["total_rooms"].median()),        step=10)
            total_bedrooms     = st.number_input("Total Bedrooms",     value=int(df["total_bedrooms"].median()),     step=1)
            population         = st.number_input("Population",         value=int(df["population"].median()),         step=50)
        with c3:
            households         = st.number_input("Households",         value=int(df["households"].median()),         step=10)
            median_income      = st.number_input("Median Income",      value=float(df["median_income"].median()),    format="%.4f", step=0.1)
            ocean_proximity    = st.selectbox("Ocean Proximity",       ["<1H OCEAN","INLAND","ISLAND","NEAR BAY","NEAR OCEAN"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Predict House Price", use_container_width=True)

    if submitted and model is not None:
        # Build input dict
        raw_input = {
            "longitude":           longitude,
            "latitude":            latitude,
            "housing_median_age":  housing_median_age,
            "total_rooms":         total_rooms,
            "total_bedrooms":      total_bedrooms,
            "population":          population,
            "households":          households,
            "median_income":       median_income,
            "ocean_proximity":     ocean_proximity,
        }
        inp_df = pd.DataFrame([raw_input])
        # Log transforms
        inp_df["total_rooms"]    = np.log(inp_df["total_rooms"]    + 1)
        inp_df["total_bedrooms"] = np.log(inp_df["total_bedrooms"] + 1)
        inp_df["population"]     = np.log(inp_df["population"]     + 1)
        inp_df["households"]     = np.log(inp_df["households"]     + 1)
        # One-hot
        inp_df = pd.get_dummies(inp_df, columns=["ocean_proximity"], dtype=int)
        for cat in OCEAN_CATS:
            if cat not in inp_df.columns:
                inp_df[cat] = 0
        # Engineered
        inp_df["bedroom_ratio"]   = inp_df["total_bedrooms"] / inp_df["total_rooms"]
        inp_df["household_rooms"] = inp_df["total_rooms"]    / inp_df["households"]
        # Align
        inp_df = inp_df.reindex(columns=feature_cols, fill_value=0)
        # Scale
        inp_scaled = scaler.transform(inp_df)
        prediction = model.predict(inp_scaled)[0]

        # Confidence: IQR of training prices
        q25 = df["median_house_value"].quantile(0.25)
        q75 = df["median_house_value"].quantile(0.75)
        pct = (prediction - df["median_house_value"].min()) / (df["median_house_value"].max() - df["median_house_value"].min()) * 100

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prediction-result">
            <div style="font-size:0.85rem;letter-spacing:0.12em;text-transform:uppercase;color:#8892a4;margin-bottom:12px;">Estimated Market Value</div>
            <div class="prediction-price">${prediction:,.0f}</div>
            <div style="font-size:0.9rem;color:#8892a4;margin-top:10px;">±${st.session_state.metrics['mae']:,.0f} average error</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        cards_p = [
            (c1, "gold",   "Predicted Price", f"${prediction:,.0f}"),
            (c2, "cyan",   "Market Percentile", f"{pct:.0f}th"),
            (c3, "teal",   "Model Accuracy",  f"{st.session_state.metrics['r2']:.1%}"),
            (c4, "violet", "Avg Error (MAE)",  f"${st.session_state.metrics['mae']:,.0f}"),
        ]
        for col, clr, lbl, val in cards_p:
            with col:
                st.markdown(f'<div class="metric-card {clr}"><div class="metric-label">{lbl}</div><div class="metric-value" style="font-size:1.5rem;">{val}</div></div>', unsafe_allow_html=True)

        # Gauge chart
        st.markdown('<br>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={"reference": df["median_house_value"].median(), "valueformat": "$,.0f"},
            number={"prefix": "$", "valueformat": ",.0f", "font": {"size": 32, "color": "#f0b429", "family": "Syne"}},
            gauge={
                "axis": {"range": [df["median_house_value"].min(), df["median_house_value"].max()], "tickformat": "$,.0f", "tickcolor": "#8892a4"},
                "bar":  {"color": "#63b3ed"},
                "bgcolor": "rgba(11,15,30,0.6)",
                "borderwidth": 2,
                "bordercolor": "rgba(99,179,237,0.3)",
                "steps": [
                    {"range": [df["median_house_value"].min(), q25], "color": "rgba(252,129,129,0.2)"},
                    {"range": [q25, q75],                           "color": "rgba(99,179,237,0.2)"},
                    {"range": [q75, df["median_house_value"].max()],"color": "rgba(104,211,145,0.2)"},
                ],
                "threshold": {"line": {"color": "#f0b429", "width": 3}, "thickness": 0.85, "value": prediction},
            },
            title={"text": "Price Gauge vs Market Range", "font": {"size": 14, "color": "#8892a4", "family": "DM Sans"}},
        ))
        apply_dark(fig_gauge)
        fig_gauge.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Download prediction report
        report_df = pd.DataFrame({
            "Attribute":   list(raw_input.keys()),
            "Value":       [str(v) for v in raw_input.values()],
        })
        report_df.loc[len(report_df)] = ["Predicted Price", f"${prediction:,.2f}"]
        report_df.loc[len(report_df)] = ["Model R²",        f"{st.session_state.metrics['r2']:.4f}"]
        csv_bytes = report_df.to_csv(index=False).encode()
        st.download_button("📥 Download Prediction Report (CSV)", data=csv_bytes, file_name="prediction_report.csv", mime="text/csv")


# ══════════════════════════════════════════════
#  PAGE 5 · MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif page == "🎯  Model Performance":
    if not require_data():
        st.stop()

    met    = st.session_state.metrics
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    st.markdown('<div class="hero-title" style="font-size:2.2rem;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Comprehensive evaluation of the RandomForest model on the held-out test set.</div><br>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "gold",   "🎯", "R² Score",  f"{met['r2']:.4f}",      "Variance explained"),
        (c2, "cyan",   "📏", "MAE",       f"${met['mae']:,.2f}",    "Mean Absolute Error"),
        (c3, "teal",   "📐", "MSE",       f"{met['mse']:,.0f}",     "Mean Squared Error"),
        (c4, "violet", "⚡", "RMSE",      f"${met['rmse']:,.2f}",   "Root Mean Squared Error"),
    ]
    for col, clr, icon, lbl, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card {clr}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{lbl}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-delta">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Actual vs Predicted
    c1, c2 = st.columns(2)
    with c1:
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="#63b3ed", opacity=0.5, size=5),
            name="Predictions",
        ))
        min_v, max_v = float(y_test.min()), float(y_test.max())
        fig_avp.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines", line=dict(color="#f0b429", dash="dash", width=2),
            name="Perfect Prediction",
        ))
        fig_avp.update_layout(title="Actual vs Predicted Prices", xaxis_title="Actual ($)", yaxis_title="Predicted ($)")
        apply_dark(fig_avp)
        st.plotly_chart(fig_avp, use_container_width=True)

    with c2:
        residuals = np.array(y_pred) - np.array(y_test)
        fig_res = px.scatter(
            x=y_pred, y=residuals, opacity=0.5,
            color=residuals, color_continuous_scale="RdBu_r",
            title="Residual Plot",
            labels={"x": "Predicted ($)", "y": "Residual ($)"},
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="#f0b429", line_width=2)
        apply_dark(fig_res)
        st.plotly_chart(fig_res, use_container_width=True)

    # Error distribution
    c1, c2 = st.columns(2)
    with c1:
        fig_err = px.histogram(
            x=residuals, nbins=60, marginal="violin",
            color_discrete_sequence=["#4fd1c5"],
            title="Error Distribution",
            labels={"x": "Prediction Error ($)", "y": "Count"},
        )
        fig_err.add_vline(x=0, line_dash="dash", line_color="#f0b429", line_width=2)
        apply_dark(fig_err)
        st.plotly_chart(fig_err, use_container_width=True)

    with c2:
        # Feature Importance
        if st.session_state.model is not None and st.session_state.feature_cols is not None:
            importances = st.session_state.model.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Feature":    st.session_state.feature_cols,
                "Importance": importances,
            }).sort_values("Importance").tail(12)
            fig_fi = px.bar(feat_imp_df, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Blues",
                            title="Top Feature Importances")
            apply_dark(fig_fi)
            st.plotly_chart(fig_fi, use_container_width=True)

    # Comparison table
    st.markdown('<br><div class="glass-card">', unsafe_allow_html=True)
    section_title("Prediction Comparison Table", "Sample of actual vs predicted (first 20 rows)")
    comp_df = pd.DataFrame({
        "Actual Price ($)":    np.array(y_test)[:20].round(2),
        "Predicted Price ($)": np.array(y_pred)[:20].round(2),
        "Error ($)":           (np.array(y_pred) - np.array(y_test))[:20].round(2),
        "Error %":             (np.abs(np.array(y_pred) - np.array(y_test)) / np.array(y_test) * 100)[:20].round(2),
    })
    st.dataframe(comp_df.style.background_gradient(cmap="RdYlGn_r", subset=["Error ($)"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 6 · ADVANCED ANALYTICS
# ══════════════════════════════════════════════
elif page == "🔬  Advanced Analytics":
    if not require_data():
        st.stop()

    df           = st.session_state.df_raw
    model        = st.session_state.model
    feature_cols = st.session_state.feature_cols

    st.markdown('<div class="hero-title" style="font-size:2.2rem;">Advanced Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Outlier detection, correlation explorer, price segmentation, and deeper data insights.</div><br>', unsafe_allow_html=True)

    tabs = st.tabs(["🚨 Outliers", "🔗 Correlation Explorer", "💡 Data Insights", "🌍 Price Segments"])

    # ── Outliers
    with tabs[0]:
        section_title("Outlier Detection", "IQR-based outlier analysis per feature")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        outlier_feat = st.selectbox("Select Feature", num_cols, key="outlier_feat")

        Q1  = df[outlier_feat].quantile(0.25)
        Q3  = df[outlier_feat].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 1.5 * IQR
        hi  = Q3 + 1.5 * IQR
        out_mask = (df[outlier_feat] < lo) | (df[outlier_feat] > hi)
        n_out    = out_mask.sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card rose"><div class="metric-label">Outliers Detected</div><div class="metric-value">{n_out}</div><div class="metric-delta">{n_out/len(df)*100:.1f}% of data</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card cyan"><div class="metric-label">Lower Fence</div><div class="metric-value">{lo:,.1f}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card gold"><div class="metric-label">Upper Fence</div><div class="metric-value">{hi:,.1f}</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        fig_out = go.Figure()
        normal = df[~out_mask][outlier_feat]
        outliers = df[out_mask][outlier_feat]
        fig_out.add_trace(go.Scatter(y=normal,   mode="markers", marker=dict(color="#63b3ed", size=3, opacity=0.5), name="Normal"))
        fig_out.add_trace(go.Scatter(y=outliers, mode="markers", marker=dict(color="#fc8181", size=5, opacity=0.8), name="Outliers"))
        fig_out.add_hline(y=hi, line_dash="dash", line_color="#f0b429", annotation_text="Upper fence")
        fig_out.add_hline(y=lo, line_dash="dash", line_color="#f0b429", annotation_text="Lower fence")
        fig_out.update_layout(title=f"Outlier View — {outlier_feat}", yaxis_title=outlier_feat)
        apply_dark(fig_out)
        st.plotly_chart(fig_out, use_container_width=True)

    # ── Correlation Explorer
    with tabs[1]:
        section_title("Correlation Explorer", "Select any two features and explore their relationship")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1:
            f1 = st.selectbox("Feature X", num_cols, index=num_cols.index("median_income") if "median_income" in num_cols else 0)
        with c2:
            f2 = st.selectbox("Feature Y", num_cols, index=num_cols.index("median_house_value") if "median_house_value" in num_cols else 1)
        with c3:
            color_by = st.selectbox("Color By", ["None"] + (["ocean_proximity"] if "ocean_proximity" in df.columns else []) + num_cols[:3])

        corr_val = df[[f1, f2]].corr().iloc[0, 1]
        st.markdown(f'<div class="info-box">Pearson Correlation between <b style="color:#63b3ed;">{f1}</b> and <b style="color:#4fd1c5;">{f2}</b>: <b style="color:#f0b429;font-size:1.1rem;">{corr_val:.4f}</b></div>', unsafe_allow_html=True)

        fig_corr = px.scatter(
            df, x=f1, y=f2,
            color=None if color_by == "None" else color_by,
            opacity=0.5, trendline="ols",
            color_continuous_scale="plasma",
            title=f"{f1} vs {f2} (r = {corr_val:.3f})",
        )
        apply_dark(fig_corr)
        fig_corr.update_layout(height=480)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Data Insights
    with tabs[2]:
        section_title("Data Insights", "Auto-generated observations about the dataset")

        insights = []
        if "median_income" in df.columns and "median_house_value" in df.columns:
            r = df[["median_income","median_house_value"]].corr().iloc[0,1]
            insights.append(f"💰 <b>Income</b> is the strongest numeric predictor of house value (r = {r:.3f})")
        if "ocean_proximity" in df.columns:
            top_cat = df.groupby("ocean_proximity")["median_house_value"].mean().idxmax()
            insights.append(f"🌊 Properties in <b>{top_cat}</b> command the highest average prices")
        if "housing_median_age" in df.columns:
            r2 = df[["housing_median_age","median_house_value"]].corr().iloc[0,1]
            insights.append(f"🏚️ Housing age has a <b>{'positive' if r2>0 else 'negative'}</b> correlation (r = {r2:.3f}) with price")
        q10 = df["median_house_value"].quantile(0.10) if "median_house_value" in df.columns else None
        q90 = df["median_house_value"].quantile(0.90) if "median_house_value" in df.columns else None
        if q10 and q90:
            insights.append(f"📊 Price range (10th–90th pct): <b>${q10:,.0f}</b> → <b>${q90:,.0f}</b>")
        null_cols = df.columns[df.isnull().sum() > 0].tolist()
        if null_cols:
            insights.append(f"⚠️ Columns with missing data: <b>{', '.join(null_cols)}</b> — handled by row-drop")
        else:
            insights.append("✅ No missing values detected after initial cleaning")

        for ins in insights:
            st.markdown(f'<div class="info-box">{ins}</div>', unsafe_allow_html=True)

        # Skewness
        st.markdown('<br>', unsafe_allow_html=True)
        num_df = df.select_dtypes(include=np.number)
        skew   = num_df.skew().reset_index()
        skew.columns = ["Feature","Skewness"]
        skew["Type"] = skew["Skewness"].apply(lambda x: "Right-skewed" if x > 0.5 else ("Left-skewed" if x < -0.5 else "Symmetric"))
        fig_skew = px.bar(skew.sort_values("Skewness"), x="Skewness", y="Feature", orientation="h",
                          color="Skewness", color_continuous_scale="RdBu_r", title="Skewness by Feature")
        apply_dark(fig_skew)
        st.plotly_chart(fig_skew, use_container_width=True)

    # ── Price Segments
    with tabs[3]:
        section_title("Price Segmentation", "Distribution of properties across price tiers")
        if "median_house_value" in df.columns:
            bins   = [0, 100000, 200000, 300000, 400000, 500000, 1e9]
            labels = ["< $100K","$100K–$200K","$200K–$300K","$300K–$400K","$400K–$500K","> $500K"]
            df["price_segment"] = pd.cut(df["median_house_value"], bins=bins, labels=labels)
            seg_counts = df["price_segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment","Count"]

            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(seg_counts, names="Segment", values="Count",
                                 color_discrete_sequence=px.colors.sequential.Blues_r,
                                 title="Properties by Price Tier")
                apply_dark(fig_pie)
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                fig_seg = px.bar(seg_counts, x="Segment", y="Count",
                                 color="Count", color_continuous_scale="Blues",
                                 title="Count by Price Segment")
                apply_dark(fig_seg)
                st.plotly_chart(fig_seg, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 7 · EXPORT
# ══════════════════════════════════════════════
elif page == "📤  Export":
    if not require_data():
        st.stop()

    df     = st.session_state.df_raw
    met    = st.session_state.metrics
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    model  = st.session_state.model
    feature_cols = st.session_state.feature_cols

    st.markdown('<div class="hero-title" style="font-size:2.2rem;">Export Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Download your data, predictions, model metrics, and feature importance reports.</div><br>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # ── Raw dataset
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_title("📋 Raw Dataset", "Full cleaned CSV")
        clean_csv = df.dropna().to_csv(index=False).encode()
        st.download_button("⬇️ Download Dataset (CSV)", data=clean_csv, file_name="housing_clean.csv", mime="text/csv", use_container_width=True)
        st.markdown(f'<div class="info-box">{len(df.dropna()):,} rows · {len(df.columns)} columns</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Predictions
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_title("🔮 Predictions", "Actual vs Predicted test set")
        if y_test is not None and y_pred is not None:
            pred_df = pd.DataFrame({
                "Actual Price ($)":    np.array(y_test).round(2),
                "Predicted Price ($)": np.array(y_pred).round(2),
                "Error ($)":           (np.array(y_pred) - np.array(y_test)).round(2),
                "Abs Error ($)":       np.abs(np.array(y_pred) - np.array(y_test)).round(2),
                "Error %":             (np.abs(np.array(y_pred) - np.array(y_test)) / np.array(y_test) * 100).round(2),
            })
            pred_csv = pred_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions (CSV)", data=pred_csv, file_name="predictions.csv", mime="text/csv", use_container_width=True)
            st.markdown(f'<div class="info-box">{len(pred_df):,} predictions on test set</div>', unsafe_allow_html=True)
        else:
            st.info("Predictions not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Report
    with c3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_title("📊 Model Report", "Performance metrics summary")
        report = pd.DataFrame({
            "Metric": ["R² Score","MAE ($)","MSE","RMSE ($)"],
            "Value":  [f"{met['r2']:.6f}", f"${met['mae']:,.2f}", f"{met['mse']:,.2f}", f"${met['rmse']:,.2f}"],
        })
        report_csv = report.to_csv(index=False).encode()
        st.download_button("⬇️ Download Report (CSV)", data=report_csv, file_name="model_report.csv", mime="text/csv", use_container_width=True)
        st.dataframe(report, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature Importance
    st.markdown('<br><div class="glass-card">', unsafe_allow_html=True)
    section_title("🌟 Feature Importance", "Download the full feature importance table")
    if model is not None and feature_cols is not None:
        fi_df = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": model.feature_importances_,
            "Importance %": (model.feature_importances_ / model.feature_importances_.sum() * 100).round(2),
        }).sort_values("Importance", ascending=False)
        st.dataframe(fi_df, use_container_width=True)
        fi_csv = fi_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Feature Importance (CSV)", data=fi_csv, file_name="feature_importance.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Descriptive Stats
    st.markdown('<br><div class="glass-card">', unsafe_allow_html=True)
    section_title("📐 Descriptive Statistics", "Statistical summary of the full dataset")
    desc = df.describe().T.round(4)
    st.dataframe(desc, use_container_width=True)
    desc_csv = desc.to_csv().encode()
    st.download_button("⬇️ Download Descriptive Stats (CSV)", data=desc_csv, file_name="descriptive_stats.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)
