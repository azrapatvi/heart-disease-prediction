import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio · Health Assessment",
    page_icon="🫀",
    layout="centered",
)

# ── LUXURY CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=Jost:wght@300;400;500&display=swap');

:root {
    --white:    #ffffff;
    --off:      #f7f5f2;
    --light:    #ede9e3;
    --border:   #ddd8d0;
    --text:     #1a1714;
    --muted:    #9e9890;
    --accent:   #c0392b;
    --gold:     #b09060;
}

html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background-color: var(--off);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 4rem;
    padding-bottom: 5rem;
    max-width: 760px;
}

/* ── Logo ── */
.logo-wrap { text-align: center; margin-bottom: 2.8rem; }
.logo-circle {
    display: inline-flex; align-items: center; justify-content: center;
    width: 64px; height: 64px; border-radius: 50%;
    border: 1.5px solid var(--gold); font-size: 1.5rem; margin-bottom: 1.2rem;
    background: var(--white); box-shadow: 0 2px 24px rgba(176,144,96,0.12);
}
.brand-name {
    font-family: 'Cormorant Garamond', serif; font-size: 2.8rem;
    font-weight: 300; letter-spacing: 0.18em; color: var(--text);
    text-transform: uppercase; line-height: 1;
}
.brand-sub {
    font-size: 0.7rem; letter-spacing: 0.25em; text-transform: uppercase;
    color: var(--muted); margin-top: 0.5rem;
}
.thin-rule {
    border: none; border-top: 1px solid var(--border);
    margin: 0 auto 3rem auto; width: 60%;
}

/* ── Section heading ── */
.sec-head {
    font-family: 'Cormorant Garamond', serif; font-size: 1.05rem;
    font-weight: 500; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--gold); margin-bottom: 1.2rem; margin-top: 2.4rem;
}

/* ── Model info cards ── */
.info-lr {
    background: #eaf0fb; border: 1px solid #b8d0f0; border-radius: 6px;
    padding: 0.75rem 1.2rem; font-size: 0.78rem; color: #2563a8; margin-top: 0.4rem;
}
.info-ann {
    background: #f0faf3; border: 1px solid #a8dbb8; border-radius: 6px;
    padding: 0.75rem 1.2rem; font-size: 0.78rem; color: #1a7a3c; margin-top: 0.4rem;
}

/* ── Input overrides ── */
label {
    font-size: 0.72rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; color: var(--muted) !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div > input {
    background-color: var(--white) !important; border: 1px solid var(--border) !important;
    border-radius: 4px !important; color: var(--text) !important;
    font-family: 'Jost', sans-serif !important; font-size: 0.9rem !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within > input {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(176,144,96,0.12) !important;
}

/* ── Button ── */
.stButton > button {
    background: var(--text) !important; color: var(--white) !important;
    border: none !important; border-radius: 4px !important;
    padding: 0.85rem 0 !important; width: 100% !important;
    font-family: 'Jost', sans-serif !important; font-size: 0.72rem !important;
    font-weight: 500 !important; letter-spacing: 0.3em !important;
    text-transform: uppercase !important; cursor: pointer !important;
    transition: background 0.25s ease, box-shadow 0.25s ease !important;
    margin-top: 1.6rem !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    box-shadow: 0 6px 32px rgba(192,57,43,0.18) !important;
}

/* ── Result card ── */
.result-card {
    background: var(--white); border: 1px solid var(--border);
    border-radius: 6px; padding: 2rem 2.4rem;
    margin-top: 2rem; text-align: center;
}
.result-label {
    font-size: 0.68rem; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem;
}
.result-model-tag {
    font-size: 0.65rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 1rem; opacity: 0.7;
}
.result-verdict {
    font-family: 'Cormorant Garamond', serif; font-size: 2rem;
    font-weight: 500; letter-spacing: 0.04em;
}
.result-positive { color: var(--accent); }
.result-negative { color: #2e7d52; }

/* ── Hide default streamlit alerts ── */
.stAlert { display: none !important; }

/* ── Footer ── */
.lux-footer {
    text-align: center; margin-top: 5rem; font-size: 0.65rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--border);
}
</style>
""", unsafe_allow_html=True)

# ── LOGO / HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="logo-wrap">
    <div class="logo-circle">🫀</div>
    <div class="brand-name">Cardio</div>
    <div class="brand-sub">Cardiac Health Assessment</div>
</div>
<hr class="thin-rule">
""", unsafe_allow_html=True)

# ── MODEL SELECTOR ────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Select Model</div>', unsafe_allow_html=True)

selected_option = st.selectbox(
    "Choose the prediction model",
    ['Logistic Regression', 'ANN'],
    format_func=lambda x: f"🔵  {x}" if x == "Logistic Regression" else f"🟢  {x}",
)

if selected_option == 'Logistic Regression':
    st.markdown("""
    <div class="info-lr">
        📘 &nbsp;<strong>Logistic Regression</strong> — A simple, fast, and interpretable
        statistical model. Great for linearly separable data.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-ann">
        🧠 &nbsp;<strong>ANN (Neural Network)</strong> — A deep learning model that captures
        complex non-linear patterns in the data.
    </div>
    """, unsafe_allow_html=True)

# ── INPUTS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Patient Profile</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    age     = st.number_input("Age", min_value=1, max_value=120, value=50)
    cp      = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
with c2:
    sex      = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    exang    = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])

st.markdown('<div class="sec-head">Clinical Indicators</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2, gap="large")
with c3:
    thalach = st.number_input("Max Heart Rate", value=150)
    slope   = st.selectbox("Slope (0-2)", [0, 1, 2])
    thal    = st.selectbox("Thal (1-3)", [1, 2, 3])
with c4:
    oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0)
    ca      = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

# ── DATAFRAME ─────────────────────────────────────────────────────────────────
new_df = pd.DataFrame({
    'age':      [age],
    'sex':      [sex],
    'cp':       [cp],
    'trestbps': [trestbps],
    'restecg':  [restecg],
    'thalach':  [thalach],
    'exang':    [exang],
    'oldpeak':  [oldpeak],
    'slope':    [slope],
    'ca':       [ca],
    'thal':     [thal],
})

# ── LOAD SCALER (shared by both models) ───────────────────────────────────────
heart_model = joblib.load('heart_model.pkl')
scaler      = heart_model['scaler']

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
predict_clicked = st.button("Run Assessment")

# ── RESULTS ───────────────────────────────────────────────────────────────────
if predict_clicked:

    st.markdown('<div class="sec-head">Input Summary</div>', unsafe_allow_html=True)
    st.dataframe(new_df, use_container_width=True)

    new_df_scaled = scaler.transform(new_df)

    # ────────────────────────────────────────────────────────────────────────
    # LOGISTIC REGRESSION branch (your original logic)
    # ────────────────────────────────────────────────────────────────────────
    if selected_option == 'Logistic Regression':

        model      = heart_model['model']
        prediction = model.predict(new_df_scaled)
        detected   = prediction[0] == 1

        tag = "Logistic Regression"

        # probability pie chart
        prob   = model.predict_proba(new_df_scaled)[0]
        labels = ["No Disease", "Disease"]

    # ────────────────────────────────────────────────────────────────────────
    # ANN branch (your original logic)
    # ────────────────────────────────────────────────────────────────────────
    else:

        model      = load_model('ann_model.h5')
        prediction = model.predict(new_df_scaled)
        pred_class = (prediction > 0.5).astype(int)
        detected   = pred_class[0][0] == 1

        tag = "ANN · Neural Network"

        # build prob array for pie chart
        p1     = float(prediction[0][0])
        prob   = [1 - p1, p1]
        labels = ["No Disease", "Disease"]

    # ── Result card ──────────────────────────────────────────────────────────
    if detected:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Assessment Result</div>
            <div class="result-model-tag">via {tag}</div>
            <div class="result-verdict result-positive">Heart Disease Detected</div>
            <div style="font-size:0.75rem;color:#9e9890;margin-top:0.6rem;letter-spacing:0.1em;">
                Please consult a cardiologist immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Assessment Result</div>
            <div class="result-model-tag">via {tag}</div>
            <div class="result-verdict result-negative">No Heart Disease Detected</div>
            <div style="font-size:0.75rem;color:#9e9890;margin-top:0.6rem;letter-spacing:0.1em;">
                Continue regular health monitoring.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability pie chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    colors = ['#2e7d52', '#c0392b']
    wedges, texts, autotexts = ax.pie(
        prob, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops={'linewidth': 3, 'edgecolor': '#ffffff'},
        textprops={'fontfamily': 'serif', 'fontsize': 11, 'color': '#1a1714'},
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_color('#ffffff')
        at.set_fontsize(10)

    ax.set_title(f"Risk Probability · {tag}", color='#9e9890', fontsize=8,
                 pad=14, fontfamily='monospace', loc='center')

    st.markdown('<div class="sec-head">Probability Breakdown</div>', unsafe_allow_html=True)
    pie_col, _ = st.columns([1, 1])
    with pie_col:
        st.pyplot(fig)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lux-footer">
    Cardio · Clinical Reference Only · Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)
