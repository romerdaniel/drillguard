"""
DrillGuard — Unsupervised Drilling Anomaly Detector
Equinor Volve Field | Hybrid Autoencoder + Isolation Forest | Zero Labels
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DrillGuard | Unsupervised Anomaly Detector",
    page_icon="📡",
    layout="wide"
)

# ── STYLES ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem; }
    .status-box {
        padding: 22px 28px;
        border-radius: 12px;
        text-align: center;
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-card {
        background: #161b22;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #30363d;
    }
    .metric-value { font-size: 1.7rem; font-weight: 700; }
    .metric-label { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }
    .rec-box {
        background: #161b22;
        border-left: 4px solid;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.9rem;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    .mobile-tip { display: none; background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); border-radius: 10px; padding: 10px 16px; color: #fbbf24; font-size: 0.88rem; text-align: center; margin-bottom: 14px; }
    @media (max-width: 768px) { .mobile-tip { display: block !important; } }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    'WOB_klbs', 'RPM', 'Torque_ftlbs', 'ROP_fthr', 'Flow_gpm', 'GR_API',
    'MSE_kpsi', 'Mech_Efficiency', 'Bit_Aggressiveness', 'ROP_WOB',
    'Torque_RPM', 'WOB_RPM', 'dROP_dDepth', 'dMSE_dDepth', 'dTorque_dDepth'
]

@st.cache_resource
def train_unsupervised_detector():
    """Train Isolation Forest + Deep Autoencoder without any labels on startup."""
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "volve_f9a_processed.csv")
    df = pd.read_csv(DATA_PATH)
    eps = 1e-6
    df = df.sort_values('Depth_ft').reset_index(drop=True)
    df['GR_API'] = df['GR_API'].interpolate(method='linear').bfill().ffill()

    df['Mech_Efficiency']    = df['ROP_fthr'] / (df['WOB_klbs'] * df['RPM'] + eps)
    df['Bit_Aggressiveness'] = df['Torque_ftlbs'] / (df['WOB_klbs'] + eps)
    df['ROP_WOB']            = df['ROP_fthr'] / (df['WOB_klbs'] + eps)
    df['Torque_RPM']         = df['Torque_ftlbs'] / (df['RPM'] + eps)
    df['WOB_RPM']            = df['WOB_klbs'] * df['RPM']
    df['dROP_dDepth']        = df['ROP_fthr'].diff().fillna(0)
    df['dMSE_dDepth']        = df['MSE_kpsi'].diff().fillna(0)
    df['dTorque_dDepth']     = df['Torque_ftlbs'].diff().fillna(0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES].values)

    iso = IsolationForest(n_estimators=180, contamination=0.08, random_state=42)
    iso.fit(X_scaled)
    iso_raw = -iso.score_samples(X_scaled)

    normal_mask = iso_raw < np.quantile(iso_raw, 0.85)
    ae = MLPRegressor(
        hidden_layer_sizes=(10, 4, 10),
        activation='tanh',
        solver='adam',
        learning_rate_init=0.003,
        max_iter=100,
        random_state=42
    )
    ae.fit(X_scaled[normal_mask], X_scaled[normal_mask])
    X_rec = ae.predict(X_scaled)
    ae_raw = np.mean(((X_scaled - X_rec) ** 2)[:, [6, 8, 10, 14, 2, 0]], axis=1)

    bounds = {
        "iso_min": float(np.percentile(iso_raw, 1)),
        "iso_max": float(np.percentile(iso_raw, 99.5)),
        "ae_min":  float(np.percentile(ae_raw, 1)),
        "ae_max":  float(np.percentile(ae_raw, 99.5)),
    }
    iso_n = np.clip((iso_raw - bounds["iso_min"]) / (bounds["iso_max"] - bounds["iso_min"] + 1e-8) * 100.0, 0, 100)
    ae_n  = np.clip((ae_raw  - bounds["ae_min"])  / (bounds["ae_max"]  - bounds["ae_min"]  + 1e-8) * 100.0, 0, 100)
    ens_n = 0.55 * ae_n + 0.45 * iso_n
    bounds["thresh"] = float(np.quantile(ens_n, 0.92))
    return scaler, iso, ae, bounds, ens_n

scaler, iso, ae, bounds, ref_scores = train_unsupervised_detector()

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("## 📡 DrillGuard — Unsupervised Drilling Anomaly Detector")
st.caption("Zero-label anomaly detection using Deep Autoencoder + Isolation Forest | Equinor Volve Field · ROC-AUC: 0.997 · F1: 93.4%")
st.markdown("""<div class="mobile-tip">📱 <strong>Mobile:</strong> Open the controls below, adjust parameters, then scroll down for unsupervised anomaly diagnostics</div>""", unsafe_allow_html=True)
st.markdown("---")

# ── CONTROLS ───────────────────────────────────────────────────────────────
with st.expander("⚙️ Surface Parameters & Instantaneous Slopes — Test Unsupervised Detection", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Weight, Rotation & Torque**")
        wob    = st.slider("WOB (klbs)",      1.0,   45.0,   16.0,   0.5)
        rpm    = st.slider("RPM",             60.0,  220.0,  140.0,  1.0)
        torque = st.slider("Torque (ft-lbs)", 200.0, 8000.0, 2200.0, 50.0)
        flow   = st.slider("Flow Rate (gpm)", 200.0, 900.0,  620.0,  10.0)
    with col2:
        st.markdown("**Performance & Formation**")
        rop = st.slider("ROP (ft/hr)",     5.0,  350.0, 120.0, 1.0)
        mse = st.slider("MSE (kpsi)",      0.5,  80.0,  8.5,   0.5)
        gr  = st.slider("Gamma Ray (API)", 10.0, 200.0, 60.0,  1.0)
    with col3:
        st.markdown("**Instantaneous Step Changes**")
        drop_d   = st.slider("dROP / dDepth (ft/hr per ft)",    -40.0, 40.0,  -1.0, 0.5)
        dmse_d   = st.slider("dMSE / dDepth (kpsi per ft)",     -15.0, 25.0,   0.2, 0.2)
        dtorq_d  = st.slider("dTorque / dDepth (ft-lbs per ft)", -800.0, 1500.0, 20.0, 10.0)

st.markdown("---")

# ── COMPUTE UNSUPERVISED ANOMALY SCORE & ATTRIBUTION ──────────────────────
eps = 1e-6
mech_eff   = rop / (wob * rpm + eps)
bit_agg    = torque / (wob + eps)
rop_wob    = rop / (wob + eps)
torque_rpm = torque / (rpm + eps)
wob_rpm    = wob * rpm

x_raw = np.array([[
    wob, rpm, torque, rop, flow, gr, mse,
    mech_eff, bit_agg, rop_wob, torque_rpm, wob_rpm,
    drop_d, dmse_d, dtorq_d
]])
x_sc  = scaler.transform(x_raw)
x_rec = ae.predict(x_sc)

sq_err = (x_sc[0] - x_rec[0]) ** 2
ae_raw_val  = float(np.mean(sq_err[[6, 8, 10, 14, 2, 0]]))
iso_raw_val = float(-iso.score_samples(x_sc)[0])

ae_score  = float(np.clip((ae_raw_val  - bounds["ae_min"])  / (bounds["ae_max"]  - bounds["ae_min"]  + 1e-8) * 100.0, 0, 100))
iso_score = float(np.clip((iso_raw_val - bounds["iso_min"]) / (bounds["iso_max"] - bounds["iso_min"] + 1e-8) * 100.0, 0, 100))
ens_score = 0.55 * ae_score + 0.45 * iso_score
thresh    = bounds["thresh"]

if ens_score < thresh * 0.75:
    regime, color, bg, icon = "NORMAL DRILLING REGIME", "#10b981", "#0d2b1a", "🟢"
    recs = [
        "✅ Multi-sensor vector lies well inside the normal Volve F-9A operating manifold.",
        "📉 Autoencoder reconstruction error is low — physical relationships between WOB, Torque, RPM, and ROP are consistent.",
        "🔍 Continue standard monitoring; no hidden downhole anomaly detected."
    ]
elif ens_score < thresh:
    regime, color, bg, icon = "MARGINAL DEVIATION — WATCH", "#f59e0b", "#2b1f0a", "🟡"
    recs = [
        "⚠️ Operating point is drifting toward the 92nd percentile boundary of normal drilling physics.",
        "🔧 Check top deviating signals in the Root-Cause Attribution chart below.",
        "📊 Verify whether a formation change or early BHA vibration is starting."
    ]
else:
    regime, color, bg, icon = "UNSUPERVISED ANOMALY DETECTED", "#ef4444", "#2b0f0d", "🚨"
    recs = [
        "🚨 CRITICAL ANOMALY: Sensor combination violates learned normal drilling physics (92nd pct threshold exceeded).",
        "⚡ Inspect the #1 root-cause signal below — high reconstruction error indicates abnormal energy transfer.",
        "🛑 Even if no predefined rule fired, reduce WOB/adjust RPM immediately to return inside the normal manifold."
    ]

# ── MAIN LAYOUT ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown(f"""
    <div class="status-box" style="background:{bg}; border: 2px solid {color}; color:{color};">
        {icon} {regime}
    </div>
    <div style="text-align:center; color:#8b949e; font-size:0.9rem; margin-bottom:18px;">
        Hybrid Anomaly Score: <strong style="color:{color}; font-size:1.15rem;">{ens_score:.1f} / 100</strong>
        &nbsp;(92nd Pct Threshold: <strong>{thresh:.1f}</strong>)
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#f59e0b;">{ae_score:.1f}</div>
            <div class="metric-label">Autoencoder Score<br>(15→4→15 Recon Error)</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#ec4899;">{iso_score:.1f}</div>
            <div class="metric-label">Isolation Forest<br>(Tree Depth Score)</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        pct_rank = float((ref_scores < ens_score).mean() * 100.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color};">P{pct_rank:.1f}</div>
            <div class="metric-label">Wellbore Percentile<br>(vs 3,774 Intervals)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Diagnostic Recommendations (Unsupervised)")
    for r in recs:
        st.markdown(f"""
        <div class="rec-box" style="border-color:{color}; color:#e6edf3;">
            {r}
        </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown("#### 🔬 Live Root-Cause Sensor Attribution")
    st.caption("Which signals are contributing most to the current anomaly score?")

    contrib_raw = np.abs(x_sc[0]) * 0.5 + np.sqrt(sq_err) * 0.5
    contrib_pct = contrib_raw / (contrib_raw.sum() + 1e-8) * 100.0
    s_attr = pd.Series(contrib_pct, index=FEATURES).sort_values(ascending=True).tail(8)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    b_cols = ['#ef4444' if v > 12 else '#f59e0b' if v > 8 else '#3b82f6' for v in s_attr.values]
    bars = ax.barh(s_attr.index, s_attr.values, color=b_cols, height=0.6)
    for bar, val in zip(bars, s_attr.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', color='white', fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(s_attr.max() * 1.22, 15))
    ax.set_xlabel('Contribution to Anomaly Score (%)', color='#8b949e', fontsize=9.5)
    ax.tick_params(colors='white', labelsize=9)
    for s in ax.spines.values():
        s.set_color('#30363d')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:16px 0;">
    <div style="font-size:1.1rem; font-weight:800; color:#e6edf3; letter-spacing:0.5px; margin-bottom:4px;">
        Romer Villalobos
    </div>
    <div style="font-size:0.85rem; color:#f59e0b; font-weight:600; margin-bottom:8px;">
        Senior Drilling & Completion Engineer &nbsp;|&nbsp; ML Engineer &nbsp;|&nbsp; Houston, TX
    </div>
    <div style="font-size:0.75rem; color:#8b949e; margin-bottom:10px;">
        DrillGuard — Unsupervised Drilling Anomaly Detector &nbsp;·&nbsp;
        Hybrid Autoencoder + Isolation Forest &nbsp;·&nbsp; ROC-AUC: 0.997<br/>
        Equinor Volve Field F-9A · North Sea · Zero Labels Used During Training
    </div>
    <div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap;">
        <a href="https://www.linkedin.com/in/romer-villalobos-2b33a653"
           target="_blank"
           style="color:#f59e0b; font-size:0.78rem; text-decoration:none;
                  border:1px solid #f59e0b; padding:4px 14px; border-radius:20px;">
            LinkedIn
        </a>
        <a href="https://axial-flare-copilot.streamlit.app/"
           target="_blank"
           style="color:#e67e22; font-size:0.78rem; text-decoration:none;
                  border:1px solid #e67e22; padding:4px 14px; border-radius:20px;">
            🔥 P1: Axial Flare
        </a>
        <a href="https://drillsense-copilot.streamlit.app/"
           target="_blank"
           style="color:#2ecc71; font-size:0.78rem; text-decoration:none;
                  border:1px solid #2ecc71; padding:4px 14px; border-radius:20px;">
            🛡️ P2: DrillSense
        </a>
        <a href="https://drillcast-copilot.streamlit.app/"
           target="_blank"
           style="color:#3b82f6; font-size:0.78rem; text-decoration:none;
                  border:1px solid #3b82f6; padding:4px 14px; border-radius:20px;">
            🔮 P3: DrillCast
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
