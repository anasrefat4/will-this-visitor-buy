"""Will This Visitor Buy? Real-time purchase intent app (Streamlit)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import CATEGORICAL_COLS, MONTHS, NUMERIC_COLS, RAW_COLS, VISITOR_TYPES, SessionFeatures  # noqa: F401

ROOT = Path(__file__).parent
st.set_page_config(page_title="Will This Visitor Buy?", page_icon="🛒", layout="wide")

STYLE = """
<style>
@font-face { font-family:'Orbitron'; src:url('app/static/Orbitron.ttf') format('truetype'); font-weight:400 900; font-display:swap; }
@font-face { font-family:'Exo 2'; src:url('app/static/Exo2.ttf') format('truetype'); font-weight:100 900; font-display:swap; }
:root { --sky:#38BDF8; --pink:#FF3EA5; --glass:rgba(255,255,255,0.10); --line:rgba(255,255,255,0.28); }
.stApp {
  background: radial-gradient(1200px 600px at 85% -10%, rgba(255,62,165,0.28), transparent 60%),
              radial-gradient(900px 500px at 0% 0%, rgba(125,211,252,0.35), transparent 60%),
              linear-gradient(160deg, #075985 0%, #0369A1 45%, #0284C7 100%);
  color: #fff;
}
html, body, [class*="css"], .stMarkdown, p, label, span, div, input, button, textarea { font-family: 'Exo 2', sans-serif; }
h1, h2, h3, h1 *, h2 *, h3 *, [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.02em; color:#fff !important; }
h1 { font-size: 2.6rem !important; text-shadow: 0 0 24px rgba(125,211,252,0.55); }
.hero-tag { font-family:'Orbitron',sans-serif !important; display:inline-block; padding:4px 12px; border:1px solid var(--pink); border-radius:999px;
            color:#fff; background:rgba(255,62,165,0.18); font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase; }
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] { background: var(--glass); border:1px solid var(--line) !important;
            border-radius:18px; backdrop-filter: blur(8px); }
[data-testid="stNumberInputContainer"], [data-baseweb="select"] > div, [data-baseweb="input"] {
            background: rgba(2,44,74,0.45) !important; border-color: var(--line) !important; border-radius:10px; }
input { color:#fff !important; }
.stButton > button, [data-testid="stBaseButton-secondary"] { background: var(--glass); color:#fff; border:1px solid var(--line); border-radius:12px; }
.stButton > button:hover { border-color: var(--pink); color:#fff; box-shadow: 0 0 16px rgba(255,62,165,0.45); }
[data-testid="stFormSubmitButton"] button, [data-testid="stBaseButton-primaryFormSubmit"] {
            background: linear-gradient(90deg, #FF3EA5, #FF7AC6); border:none; color:#fff; font-weight:600; border-radius:12px;
            box-shadow: 0 6px 20px rgba(255,62,165,0.45); }
[data-testid="stProgress"] div[role="progressbar"] > div > div { background: linear-gradient(90deg, #7DD3FC, #FF3EA5) !important; }
[data-testid="stMetricValue"] { font-size: 3.4rem !important; text-shadow: 0 0 22px rgba(255,62,165,0.55); }
[data-testid="stAlert"], [data-testid="stAlertContainer"] { background: rgba(255,255,255,0.14) !important; border:1px solid var(--line); border-radius:14px; }
[data-testid="stAlert"] * { color:#fff !important; }
.stTabs [data-baseweb="tab"] { color: rgba(255,255,255,0.8); }
.stTabs [aria-selected="true"] { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--pink) !important; }
.action-card { padding:14px 18px; border-radius:16px; border:1px solid var(--pink); background:rgba(255,62,165,0.16);
            font-family:'Orbitron',sans-serif; font-size:1.35rem; letter-spacing:0.02em; font-weight:700; color:#fff; margin:6px 0 10px;
            box-shadow: 0 0 22px rgba(255,62,165,0.30); }
a { color:#BAE6FD !important; }
[data-testid="stCaptionContainer"] { color: rgba(255,255,255,0.78) !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display:none; }
header[data-testid="stHeader"] { background: transparent; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "model" / "purchase_intent_pipeline.joblib")
    meta = json.loads((ROOT / "model" / "metadata.json").read_text())
    return model, meta


model, meta = load_model()
LOW, HIGH = meta["low_cut"], meta["high_cut"]

ACTIONS = {
    "No action": ("✅", "Likely buyer, so no coupon needed. Giving a discount here would only lose margin."),
    "Show coupon": ("🎟️", "Interested but hesitant. This is the most persuadable group, so show a time-limited coupon."),
    "Show chat": ("💬", "Low purchase intent. Offer a chat assistant to answer questions; no discount spend."),
}

PRESETS = {
    "Ready to buy": dict(product_pages=35, product_min=25.0, admin_pages=3, admin_min=2.0, info_pages=0,
                         info_min=0.0, bounce=0.3, exit=0.9, page_value=60.0, month="Nov",
                         visitor="Returning_Visitor", weekend=False, special=0.0),
    "Hesitant shopper": dict(product_pages=25, product_min=15.0, admin_pages=2, admin_min=1.0, info_pages=0,
                             info_min=0.0, bounce=1.0, exit=2.5, page_value=5.0, month="Mar",
                             visitor="Returning_Visitor", weekend=False, special=0.0),
    "Window shopper": dict(product_pages=6, product_min=3.0, admin_pages=0, admin_min=0.0, info_pages=0,
                           info_min=0.0, bounce=5.0, exit=8.0, page_value=0.0, month="May",
                           visitor="Returning_Visitor", weekend=True, special=0.0),
}


def band_for(p):
    if p >= HIGH:
        return "No action"
    if p >= LOW:
        return "Show coupon"
    return "Show chat"


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


for k, v in PRESETS["Hesitant shopper"].items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------- header
st.markdown('<span class="hero-tag">Real-time ML · E-commerce</span>', unsafe_allow_html=True)
st.title("🛒 Will This Visitor Buy?")
st.caption("Real-time purchase intent for an online store. The model scores a browsing session and "
           "suggests the best action: **no action**, **show a coupon** or **open a chat**.")

tab_single, tab_batch, tab_about = st.tabs(["Score a session", "Score a CSV of sessions", "About the model"])

# ---------------------------------------------------------------- single session
with tab_single:
    st.write("**Try an example:**")
    cols = st.columns(len(PRESETS))
    for col, name in zip(cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")

    left, right = st.columns([1.15, 1])
    with left:
        with st.form("session"):
            st.subheader("What did the visitor do?")
            c1, c2 = st.columns(2)
            c1.number_input("Product pages viewed", 0, 1000, step=1, key="product_pages")
            c2.number_input("Minutes on product pages", 0.0, 1500.0, step=0.5, key="product_min")
            c1.number_input("Account / admin pages", 0, 100, step=1, key="admin_pages")
            c2.number_input("Minutes on account pages", 0.0, 100.0, step=0.5, key="admin_min")
            c1.number_input("Info pages (shipping, contact…)", 0, 100, step=1, key="info_pages")
            c2.number_input("Minutes on info pages", 0.0, 100.0, step=0.5, key="info_min")

            st.subheader("Page quality signals")
            c3, c4, c5 = st.columns(3)
            c3.number_input("Bounce rate (%)", 0.0, 20.0, step=0.1, key="bounce",
                            help="Average bounce rate of the pages visited (Google Analytics).")
            c4.number_input("Exit rate (%)", 0.0, 20.0, step=0.1, key="exit",
                            help="Average exit rate of the pages visited (Google Analytics).")
            c5.number_input("Page value", 0.0, 400.0, step=1.0, key="page_value",
                            help="Average value of the pages visited before a purchase (Google Analytics). "
                                 "0 means none of the pages usually lead to a sale.")

            st.subheader("Context")
            c6, c7, c8 = st.columns(3)
            c6.selectbox("Month", MONTHS, key="month")
            c7.selectbox("Visitor type", VISITOR_TYPES, key="visitor",
                         format_func=lambda v: v.replace("_", " "))
            c8.slider("Closeness to a special day", 0.0, 1.0, step=0.2, key="special",
                      help="0 = far from holidays like Mother's Day, 1 = on the day.")
            st.checkbox("Weekend visit", key="weekend")

            with st.expander("Technical details (optional)"):
                t1, t2, t3, t4 = st.columns(4)
                os_ = t1.number_input("Operating system code", 1, 8, 2)
                browser = t2.number_input("Browser code", 1, 13, 2)
                region = t3.number_input("Region code", 1, 9, 1)
                traffic = t4.number_input("Traffic source code", 1, 20, 2)

            submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    s = st.session_state
    session = pd.DataFrame([{
        "Administrative": s.admin_pages, "Administrative_Duration": s.admin_min * 60,
        "Informational": s.info_pages, "Informational_Duration": s.info_min * 60,
        "ProductRelated": s.product_pages, "ProductRelated_Duration": s.product_min * 60,
        "BounceRates": s.bounce / 100, "ExitRates": s.exit / 100, "PageValues": s.page_value,
        "SpecialDay": s.special, "Month": s.month, "OperatingSystems": os_, "Browser": browser,
        "Region": region, "TrafficType": traffic, "VisitorType": s.visitor, "Weekend": s.weekend,
    }])

    with right:
        p = float(model.predict_proba(session)[0, 1])
        band = band_for(p)
        icon, why = ACTIONS[band]
        st.subheader("Prediction")
        st.metric("Purchase probability", f"{p:.0%}")
        st.progress(min(max(p, 0.0), 1.0))
        st.markdown(f'<div class="action-card">{icon} Suggested action: {band}</div>', unsafe_allow_html=True)
        st.info(why)

        st.markdown("**Signals in this session**")
        notes = []
        if s.page_value > 0:
            notes.append(f"🟢 Visited pages that usually lead to a sale (page value {s.page_value:.0f}).")
        else:
            notes.append("🔴 No high-value pages visited (page value 0). This is the strongest sign of no purchase.")
        notes.append("🟢 Low exit rate." if s.exit < 2.5 else "🔴 High exit rate: the pages visited often end sessions.")
        if s.product_pages >= 29:
            notes.append(f"🟢 Deep product browsing ({s.product_pages} pages).")
        elif s.product_pages < 10:
            notes.append(f"🔴 Little product browsing ({s.product_pages} pages).")
        if s.month == "Nov":
            notes.append("🟢 November is the highest-converting month.")
        elif s.month in ("Feb", "Mar", "May"):
            notes.append(f"🟠 {s.month} converts below average.")
        if s.visitor == "New_Visitor":
            notes.append("🟢 New visitors convert more often than returning ones.")
        for n in notes:
            st.write(n)
        st.caption(f"Action bands: below {LOW:.0%} → chat · {LOW:.0%}–{HIGH:.0%} → coupon · "
                   f"{HIGH:.0%} and above → no action.")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with the same columns as the "
             "[UCI Online Shoppers dataset](https://archive.ics.uci.edu/dataset/468) "
             "(a `Revenue` column is ignored if present). Each row gets a probability and an action.")
    with st.expander("Required columns"):
        st.code(", ".join(RAW_COLS))
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as e:  # unreadable file
            st.error(f"Could not read this file as CSV ({e}).")
            st.stop()
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error("The file is missing these columns: " + ", ".join(missing))
            st.stop()
        X = data[RAW_COLS].copy()
        for c in NUMERIC_COLS + ["OperatingSystems", "Browser", "Region", "TrafficType"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X["Weekend"] = X["Weekend"].astype(str).str.lower().isin(["true", "1", "yes"])
        bad = X[NUMERIC_COLS + ["OperatingSystems", "Browser", "Region", "TrafficType"]].isna().any(axis=1) \
            | (X[NUMERIC_COLS] < 0).any(axis=1)
        if bad.any():
            st.warning(f"{int(bad.sum())} row(s) have missing, non-numeric or negative values and were skipped.")
        good = X[~bad]
        if good.empty:
            st.error("No valid rows to score.")
            st.stop()
        out = data.loc[good.index].copy()
        out["purchase_probability"] = model.predict_proba(good)[:, 1].round(3)
        out["action"] = [band_for(v) for v in out["purchase_probability"]]
        c1, c2, c3 = st.columns(3)
        for col, name in zip((c1, c2, c3), ACTIONS):
            col.metric(f"{ACTIONS[name][0]} {name}", int((out.action == name).sum()))
        st.dataframe(out.sort_values("purchase_probability", ascending=False), width="stretch")
        st.download_button("Download results", out.to_csv(index=False), "scored_sessions.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab_about:
    m = meta["test_metrics"]
    st.markdown(f"""
**Model:** {meta['model']} in a scikit-learn pipeline (feature engineering → log + scaling → one-hot encoding → model).

**Data:** [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468) (UCI), 12,330 real sessions,
15.5% of which end in a purchase. Trained on {meta['train_rows']:,} sessions; scores below are from
{meta['test_rows']:,} held-out sessions the model never saw.
""")
    st.table(pd.DataFrame({"Metric": list(m.keys()), "Test score": [f"{v:.3f}" for v in m.values()]})
             .set_index("Metric"))
    st.markdown(f"""
Precision, recall and F1 use the decision threshold **{meta['threshold']:.2f}**, chosen on training data to maximise F1.
A model that always says "no purchase" would score 0.845 accuracy but find **zero** buyers, which is why ROC-AUC and PR-AUC are used instead.

**Limitations:** one shop and one year of data; relies heavily on Google Analytics *page value*;
the coupon effect is not measured, so it should be A/B tested before rollout.
""")
    st.caption("Built by Anas Refaat · NextGen Institute Machine Learning final project")
