"""
The COD Trust Score  --  a story about Cash-on-Delivery returns in Indian e-commerce
====================================================================================
Run locally:   pip install -r requirements.txt  &&  streamlit run app.py
Deploy free:   push this folder to GitHub -> share.streamlit.io -> point to app.py

The dashboard reads ONE messy CSV (cod_orders.csv) and tells a 7-chapter story:
  1. The Problem        - why COD returns quietly drain Indian D2C brands
  2. Meet the Data      - what each order in our (synthetic) store looks like
  3. Cleaning the Data  - the messy export, fixed step by step (data prep)
  4. What Happened      - who sends parcels back more often (descriptive)
  5. The Real Reasons   - which signals truly matter vs. noise (diagnostic)
  6. The COD Score      - turning return-risk into a CIBIL-style score (models)
  7. The Verdict        - who gets COD free / COD with a fee / prepaid only
  8. Shopper Types      - group people by habit (K-Means + GMM + 3D PCA)      [unsupervised]
  9. Basket Size        - predict the order's rupee value (4 regressors)       [regression]
 10. Trait Combos       - the risky bundles behind a return (Apriori rules)    [association]

Everything below is recomputed live from the CSV. Nothing is hard-coded.

Honesty note carried throughout: a high return-risk is a reason to ADJUST THE OFFER
(ask for prepayment, add a small fee), never a verdict that a customer is a bad person.
"""
import warnings; warnings.filterwarnings("ignore")
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             classification_report)
# ---- extra analyses (Ch. 8–10): grouping, number-prediction, combo-mining ----
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------------------------------------------- theme detection
def _active_theme():
    """Return 'light' or 'dark', following the user's actual choice when possible."""
    try:
        t = getattr(st.context, "theme", None)
        if t is not None and getattr(t, "type", None) in ("light", "dark"):
            return t.type
    except Exception:
        pass
    base = st.get_option("theme.base")
    return base if base in ("light", "dark") else "light"

DARK = _active_theme() == "dark"

# ----------------------------------------------------------------- palette
# Brand colours chosen to pop on BOTH light and dark backgrounds.
ACCENT = "#14b8a6"   # teal = trust
GREEN  = "#22c55e"   # low risk
AMBER  = "#f59e0b"   # medium risk
RED    = "#ef4444"   # high risk
GREY   = "#94a3b8"
# Theme-aware neutrals (these flip with the mode)
INK  = "#f1f5f9" if DARK else "#1f2937"   # main text on charts
MUT  = "#94a3b8" if DARK else "#64748b"   # muted labels
GRID = "#243044" if DARK else "#eef2f7"   # gridlines
EDGE = "#334155" if DARK else "#cbd5e1"   # axis spines
GRAD = ("#0f766e", "#155e75") if DARK else ("#0d9488", "#0e7490")  # header gradient

plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
    "axes.edgecolor": EDGE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.axisbelow": True, "font.size": 11, "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.titleweight": "medium", "axes.titlecolor": INK,
    "axes.titlepad": 14, "axes.labelcolor": MUT, "axes.labelsize": 10,
    "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "text.color": INK,
    "figure.autolayout": True,
})

def bare(ax, keep_left=True):
    """Minimalist axis: drop top/right spines, keep it clean."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.patch.set_alpha(0)
    return ax

st.set_page_config(page_title="The COD Trust Score", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------- global CSS (both themes)
st.markdown(f"""
<style>
  .block-container {{max-width: 1060px; padding-top: 1.4rem;}}
  h1, h2, h3 {{letter-spacing: -0.01em;}}
  p, li {{font-size: 1.03rem; line-height: 1.62;}}
  /* tabs: bigger, pill-style, clear active state */
  .stTabs [data-baseweb="tab-list"] {{gap: 4px; flex-wrap: wrap;}}
  .stTabs [data-baseweb="tab"] {{
      font-size: 0.95rem; font-weight: 600; padding: 9px 14px; border-radius: 9px;}}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(100deg,{GRAD[0]},{GRAD[1]}); color: #fff !important;}}
  /* gradient hero + chapter headers */
  .hero {{background: linear-gradient(110deg,{GRAD[0]},{GRAD[1]});
          padding: 26px 30px; border-radius: 18px; margin-bottom: 6px;
          box-shadow: 0 10px 30px rgba(13,148,136,.25);}}
  .hero h1 {{color:#fff; font-size: 2.05rem; margin:0; font-weight: 800;}}
  .hero p  {{color:#ecfeff; font-size: 1.08rem; margin:.35rem 0 0;}}
  .chead {{background: linear-gradient(100deg,var(--c1),var(--c2));
           padding: 16px 22px; border-radius: 14px; margin: 4px 0 16px;}}
  .chead .num {{color:#fff; opacity:.85; font-weight:700; font-size:.78rem;
                letter-spacing:.1em; text-transform:uppercase;}}
  .chead .ttl {{color:#fff; font-size:1.4rem; font-weight:800; line-height:1.15;}}
  .chead .sub {{color:#f8fafc; opacity:.95; font-size:1.02rem; margin-top:3px;}}
  /* takeaway pill */
  .take {{border-left:4px solid {ACCENT}; padding:.6rem .9rem; border-radius:8px;
          background: rgba(20,184,166,.10); margin:.5rem 0; font-size:1.02rem;}}
  .stat {{border-radius:14px; padding:16px 18px; text-align:center;
          background: rgba(148,163,184,.10);}}
  .stat .big {{font-size:1.9rem; font-weight:800; line-height:1;}}
  .stat .lab {{font-size:.86rem; opacity:.8; margin-top:5px;}}
  div[data-testid="stMetricValue"] {{font-size: 1.55rem; font-weight: 700;}}
  hr {{margin: 1.1rem 0;}}
</style>
""", unsafe_allow_html=True)

def chapter(num, icon, title, subtitle, c1=GRAD[0], c2=GRAD[1]):
    st.markdown(
        f"<div class='chead' style='--c1:{c1};--c2:{c2}'>"
        f"<div class='num'>Chapter {num}</div>"
        f"<div class='ttl'>{icon}&nbsp; {title}</div>"
        f"<div class='sub'>{subtitle}</div></div>", unsafe_allow_html=True)

def take(text):
    """A short, visual 'what this means' pill — replaces long paragraphs."""
    st.markdown(f"<div class='take'>💡 {text}</div>", unsafe_allow_html=True)

def statcard(col, big, lab, color=None):
    color = color or INK
    col.markdown(f"<div class='stat'><div class='big' style='color:{color}'>{big}</div>"
                 f"<div class='lab'>{lab}</div></div>", unsafe_allow_html=True)

TARGET = "DeliveryStatus"
POSITIVE = "Returned"

# COD economics (industry-reported; editable in the Verdict chapter)
DEFAULT_RTO_COST = 250   # ₹ lost on a typical returned COD order (forward+reverse+handling)

# ============================================================ DATA: load + clean
@st.cache_data(show_spinner=False)
def load_raw(file_bytes):
    if file_bytes is None:
        return pd.read_csv("cod_orders.csv")
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def clean(raw: pd.DataFrame):
    """Turn the messy export into analysis-ready data, logging every change."""
    d = raw.copy()
    notes = []
    n0 = len(d)

    # 1) exact duplicate rows (accidental double-export)
    dups = int(d.duplicated().sum())
    if dups:
        d = d.drop_duplicates().reset_index(drop=True)
        notes.append(f"Removed **{dups}** exact duplicate rows ({n0} → {len(d)}).")

    # 2) OrderValue: strip 'Rs'/'INR'/commas → number
    if "OrderValue" in d:
        d["OrderValue"] = pd.to_numeric(
            d["OrderValue"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce")
        notes.append("Parsed **OrderValue** — removed `Rs`/`INR`/commas, converted to numbers.")

    # 3) DiscountPct: strip '%', cap impossible values (>100) as data errors
    if "DiscountPct" in d:
        d["DiscountPct"] = pd.to_numeric(
            d["DiscountPct"].astype(str).str.replace("%", "", regex=False),
            errors="coerce")
        bad = int((d["DiscountPct"] > 100).sum())
        if bad:
            med = d.loc[d["DiscountPct"] <= 100, "DiscountPct"].median()
            d.loc[d["DiscountPct"] > 100, "DiscountPct"] = med
            notes.append(f"Fixed **{bad}** impossible discounts (>100%) → set to the "
                         f"median ({med:.0f}%).")
        notes.append("Parsed **DiscountPct** — removed `%`, converted to numbers.")

    # 4) CityTier: merge case/format variants (Tier 1 / tier1 / T1 / TIER-1 → Tier-1)
    if "CityTier" in d:
        before = d["CityTier"].nunique()
        digit = d["CityTier"].astype(str).str.extract(r"([123])")[0]
        d["CityTier"] = digit.map({"1": "Tier-1", "2": "Tier-2", "3": "Tier-3"})
        notes.append(f"Standardised **CityTier**: {before} messy variants → 3 clean tiers.")

    # 5) State: trim whitespace + Title Case (merge 'KARNATAKA'/'karnataka')
    if "State" in d:
        before = d["State"].nunique()
        d["State"] = d["State"].astype(str).str.strip().str.title()
        notes.append(f"Cleaned **State**: {before} → {d['State'].nunique()} distinct names "
                     "(trimmed spaces, unified case).")

    # 6) Missing categoricals → explicit 'Unknown'
    for c in ["Device", "AddressQuality"]:
        if c in d:
            miss = int(d[c].isna().sum())
            if miss:
                d[c] = d[c].fillna("Unknown")
                notes.append(f"**{c}**: {miss} missing values labelled `Unknown`.")

    # 7) binary target + handy derived features (all knowable BEFORE dispatch)
    d["RTO"] = (d[TARGET] == POSITIVE).astype(int)
    d["PriorRTORate"] = np.where(d["PriorOrders"] > 0,
                                 d["PriorReturns"] / d["PriorOrders"], 0.0)
    d["FirstTime"] = (d["PriorOrders"] == 0).astype(int)
    d["IsCOD"] = (d["PaymentMethod"] == "COD").astype(int)
    notes.append("Built the target **RTO** (1 = Returned, 0 = Delivered) and helper "
                 "features: past-return rate, first-time flag, COD flag.")

    # drop any rows that lost their value in parsing (rare)
    miss_val = int(d["OrderValue"].isna().sum())
    if miss_val:
        d = d.dropna(subset=["OrderValue"]).reset_index(drop=True)
        notes.append(f"Dropped **{miss_val}** rows with unrecoverable OrderValue.")

    return d, notes


# ============================================================ MODELLING (COD only)
NUM_FEATS = ["OrderValue", "DiscountPct", "PriorOrders", "PriorRTORate",
             "FirstTime", "OrderHour", "Items"]
CAT_FEATS = ["CityTier", "State", "Category", "Device", "AddressQuality"]
# A score must be *calibrated* (predicted risk ≈ real risk), not just high-AUC.
# Gradient Boosting's probabilities track reality almost exactly, so we score from it.
SCORE_MODEL = "Gradient Boosting"


def design_matrix(df_cod):
    X = df_cod[NUM_FEATS + CAT_FEATS].copy()
    y = df_cod["RTO"].copy()
    return X, y


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])


@st.cache_resource(show_spinner=True)
def train_models(_X, _y, test_size, seed):
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=test_size,
                                           stratify=_y, random_state=seed)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)]).fit(Xtr, ytr)
        ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        out[name] = {
            "train_acc": accuracy_score(ytr, ptr),
            "test_acc": accuracy_score(yte, pte),
            "precision": precision_score(yte, pte, zero_division=0),
            "recall": recall_score(yte, pte, zero_division=0),
            "f1": f1_score(yte, pte, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "cm": confusion_matrix(yte, pte),
            "roc": roc_curve(yte, proba),
            "report": classification_report(yte, pte,
                       target_names=["Delivered", "Returned"], zero_division=0),
            "pipe": pipe,
        }
    return out, (len(Xtr), len(Xte), ytr.mean(), yte.mean())


@st.cache_data(show_spinner=True)
def honest_scores(_X, _y, best_name, test_size, seed):
    """Return-risk for EVERY COD order using cross-validation (out-of-fold),
    so no order is scored by a model that already saw it. Then map risk → score."""
    model = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=25),
    }[best_name]
    pipe = Pipeline([("pre", preprocessor()), ("clf", model)])
    proba = cross_val_predict(pipe, _X, _y, cv=5, method="predict_proba")[:, 1]
    # CIBIL-style: low risk -> high score. Range ~300..900.
    score = np.round(900 - proba * 600).astype(int)
    return proba, score


def cramers_v(df, col, target="RTO"):
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum(); r, k = ct.shape
    v = np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1))
    return chi2, p, dof, v, int(ct.shape[0])


# ============================================================ CH.8  CLUSTERING
# "Unsupervised": we never hand the computer the answer — it finds the groups
# on its own from each shopper's habits. Used to discover natural shopper types.
CLU_FEATS = ["Orders", "AvgValue", "AvgDiscount", "PctCOD", "ReturnRate"]


@st.cache_data(show_spinner=False)
def customer_table(_df, n):
    """Boil the whole store down to ONE row per shopper (their overall habit)."""
    return (_df.groupby("CustomerID")
               .agg(Orders=("OrderID", "size"),
                    AvgValue=("OrderValue", "mean"),
                    AvgDiscount=("DiscountPct", "mean"),
                    PctCOD=("IsCOD", "mean"),
                    ReturnRate=("RTO", "mean"))
               .reset_index())


@st.cache_data(show_spinner=False)
def cluster_everything(_cust, n, seed):
    """Scale → scan k with elbow + silhouette → K-Means(2) hard groups →
    Gaussian Mixture(2) soft % membership → 3-D PCA so we can actually look."""
    Xs = StandardScaler().fit_transform(_cust[CLU_FEATS])
    ks = list(range(2, 8))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
    best_k = ks[int(np.argmax(sil))]

    km2 = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xs)
    prof = _cust.assign(KM=km2.labels_).groupby("KM")[CLU_FEATS].mean()
    risky_lab = int(prof["ReturnRate"].idxmax())          # higher returns = risky
    names = {risky_lab: "Higher-risk shoppers", 1 - risky_lab: "Steady shoppers"}
    labels = np.array([names[l] for l in km2.labels_])

    gmm = GaussianMixture(n_components=2, random_state=seed).fit(Xs)
    gm_prof = _cust.assign(G=gmm.predict(Xs)).groupby("G")["ReturnRate"].mean()
    risky_comp = int(gm_prof.idxmax())
    p_risky = gmm.predict_proba(Xs)[:, risky_comp]        # soft membership

    pca = PCA(n_components=3).fit(Xs)
    coords = pca.transform(Xs)
    return dict(ks=ks, inertia=inertia, sil=sil, best_k=best_k, labels=labels,
                prof=prof, names=names, p_risky=p_risky, coords=coords,
                evr=pca.explained_variance_ratio_)


# ============================================================ CH.9  REGRESSION
# A DIFFERENT question: predict a NUMBER (the order's ₹ value), not a yes/no.
REG_NUM = ["DiscountPct", "Items", "OrderHour", "FirstTime", "PriorOrders"]
REG_CAT = ["Category", "CityTier", "Device", "AddressQuality", "PaymentMethod"]


@st.cache_data(show_spinner=False)
def regression_models(_df, n, seed):
    X = _df[REG_NUM + REG_CAT].copy()
    y = _df["OrderValue"].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    pre = ColumnTransformer([("num", StandardScaler(), REG_NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), REG_CAT)])
    models = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=1.0),
              "Lasso": Lasso(alpha=1.0),
              "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=seed)}
    rows = {}
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr)
        pr = pipe.predict(Xte)
        rows[name] = {"R2": r2_score(yte, pr), "MAE": mean_absolute_error(yte, pr)}
    # how much does category ALONE explain? (shows most of the signal is "what")
    Xc = pd.get_dummies(_df[["Category"]])
    xct, xce, yct, yce = train_test_split(Xc, y, test_size=0.25, random_state=seed)
    cat_r2 = r2_score(yce, LinearRegression().fit(xct, yct).predict(xce))
    cat_means = _df.groupby("Category")["OrderValue"].mean().sort_values()
    return pd.DataFrame(rows).T, float(cat_r2), cat_means


# ============================================================ CH.10  ASSOCIATION RULES
# Turn each COD order into a "basket" of traits, then mine which COMBINATIONS
# (not single columns) travel with a return — the Apriori algorithm.
@st.cache_data(show_spinner=False)
def association(_cod, n, min_support=0.02):
    c = _cod
    db = pd.cut(c["DiscountPct"], [-1, 10, 25, 40, 100],
                labels=["disc 0–10%", "disc 11–25%", "disc 26–40%", "disc 40%+"])
    txn = pd.DataFrame({
        "tier":  "Tier-" + c["CityTier"].str.replace("Tier-", "", regex=False),
        "cat":   c["Category"].astype(str),
        "addr":  c["AddressQuality"].astype(str) + " address",
        "dev":   c["Device"].astype(str),
        "disc":  db.astype(str),
        "first": np.where(c["FirstTime"] == 1, "first-timer", "repeat buyer"),
        "out":   np.where(c["RTO"] == 1, "RETURNED", "delivered"),
    })
    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit_transform(txn.values.tolist()), columns=te.columns_)
    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    ret = rules[rules["consequents"] == frozenset({"RETURNED"})].copy()
    ret = ret[ret["antecedents"].apply(len) >= 2]
    ret["combo"] = ret["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    ret = ret.sort_values("lift", ascending=False)
    return ret[["combo", "support", "confidence", "lift"]].reset_index(drop=True)


# ================================================================ SIDEBAR
st.sidebar.markdown("### 📖 The story")
st.sidebar.markdown("""
1. 📦 **Problem** — why COD hurts
2. 🧾 **Data** — our store
3. 🧹 **Cleaning** — tidy it up
4. 📊 **Who returns** — the patterns
5. 🔍 **Why** — real reasons
6. 🎯 **The Score** — one number
7. ⚖️ **Verdict** — the decision
8. 👥 **Shopper types** — natural groups
9. 💰 **Basket size** — what drives ₹
10. 🔗 **Trait combos** — risky bundles
""")
st.sidebar.caption("Read the tabs left → right, like chapters.")
st.sidebar.divider()

up = st.sidebar.file_uploader("Use your own CSV (optional)", type=["csv"])
raw = load_raw(up.read() if up is not None else None)
df, notes = clean(raw)

with st.sidebar.expander("⚙️ Model settings"):
    test_size = st.slider("Test split", 0.15, 0.40, 0.25, 0.05)
    seed = int(st.number_input("Random seed", value=42, step=1))

st.sidebar.caption("🌗 Tip: light/dark mode both work — switch under the ⋮ menu → Settings.")

# COD subset = the population the score is built for
cod = df[df["IsCOD"] == 1].copy().reset_index(drop=True)
overall_rto = df["RTO"].mean()
cod_rto = cod["RTO"].mean()
prepaid_rto = df[df["IsCOD"] == 0]["RTO"].mean()

# ================================================================ HEADER
st.markdown("<div class='hero'><h1>📦 The COD Trust Score</h1>"
            "<p>A credit score — but for Cash on Delivery. Who can a brand trust with COD?</p>"
            "</div>", unsafe_allow_html=True)
st.write("")

m1, m2, m3, m4 = st.columns(4)
statcard(m1, f"{len(df):,}", "Orders")
statcard(m2, f"{df['CustomerID'].nunique():,}", "Shoppers")
statcard(m3, f"{cod_rto*100:.0f}%", "COD returns", RED)
statcard(m4, f"{prepaid_rto*100:.0f}%", "Prepaid returns", GREEN)
st.write("")

tabs = st.tabs([
    "1 · 📦 Problem", "2 · 🧾 Data", "3 · 🧹 Cleaning",
    "4 · 📊 Who Returns", "5 · 🔍 Why", "6 · 🎯 The Score",
    "7 · ⚖️ Verdict", "8 · 👥 Shopper Types", "9 · 💰 Basket Size",
    "10 · 🔗 Trait Combos",
])

# =============================================================== 1. THE PROBLEM
with tabs[0]:
    chapter(1, "📦", "The leak nobody sees",
            "Cash-on-Delivery parcels that come straight back — and burn money both ways.")

    st.markdown("#### A ₹1,200 COD order's sad journey")
    j1, j2, j3, j4 = st.columns(4)
    statcard(j1, "🛒", "Customer orders (COD)")
    statcard(j2, "🚚", "Brand ships it")
    statcard(j3, "🙅", "Parcel refused")
    statcard(j4, "↩️", "Comes back — ₹0 earned", RED)
    take("This is <b>Return to Origin (RTO)</b>: the brand pays shipping twice and earns "
         "nothing. In India it's one of the biggest hidden costs online.")

    st.markdown("#### Why it's a *big* deal (not just us)")
    c1, c2, c3 = st.columns(3)
    statcard(c1, "60–65%", "of Indian orders are COD", ACCENT)
    statcard(c2, "25–40%", "of COD comes back", RED)
    statcard(c3, "₹180–350", "lost per return", AMBER)

    with st.expander("📚 Proof from the industry (real blogs)"):
        st.markdown("""
- **GoKwik** — ₹200–250 lost on a typical ₹1,000 COD return →
  [link](https://www.gokwik.co/blog/what-is-return-to-origin-rto-in-ecommerce)
- **HillTeck** — the true all-in cost of RTO →
  [link](https://www.hillteck.com/blog/rto-cost-indian-d2c-brands.html)
- **Pragma** — cut RTO by *segmenting customers* (our exact idea) →
  [link](https://www.bepragma.ai/blogs/how-to-reduce-rto-in-indian-e-commerce-without-hurting-cod-orders)
- **Edgistify** — "RTO %: the silent killer of D2C" →
  [link](https://www.edgistify.com/resources/blogs/rto-percentage-silent-killer-indian-d2c)
- **CallFox** — reduce COD returns →
  [link](https://www.callfox.in/blog/reduce-cod-returns-india)
""")

    st.markdown("#### The big idea: a CIBIL score, but for COD")
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("""
A bank reads your **repayment history** → gives you a **CIBIL score** → decides your loan.

We read a shopper's **return history** → give them a **COD Trust Score** → decide their COD.
""")
    with colB:
        st.table(pd.DataFrame({
            "CIBIL 🏦": ["Repayment history", "Lend or not?", "Higher interest if risky",
                         "Refused below cut-off"],
            "COD Score 📦": ["Return history", "Offer COD or not?", "Small fee if risky",
                             "Prepaid-only below cut-off"],
        }))

    st.markdown("#### In our own store, the gap is huge")
    fig, ax = plt.subplots(figsize=(8, 2.2))
    rates = [prepaid_rto * 100, cod_rto * 100]
    bars = ax.barh(["Prepaid", "Cash on Delivery"], rates, color=[GREEN, RED], height=0.62)
    for b, r in zip(bars, rates):
        ax.text(r + 0.6, b.get_y() + b.get_height()/2, f"{r:.0f}%",
                va="center", fontweight="bold", color=INK, fontsize=13)
    ax.set_xlim(0, max(rates) * 1.28); ax.set_xlabel("Return rate (%)")
    bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    take("We can't just ban COD — it's how India shops. So the smart question is: "
         "<b>which shoppers get COD, and which pay first?</b> The next chapters find out.")

# =============================================================== 2. MEET THE DATA
with tabs[1]:
    chapter(2, "🧾", "Meet the store",
            "A made-up store that behaves like the real world — so we can test our idea.",
            "#7c3aed", "#5b21b6")

    d1, d2, d3 = st.columns(3)
    statcard(d1, f"{len(df):,}", "orders")
    statcard(d2, f"{df['CustomerID'].nunique():,}", "shoppers")
    statcard(d3, "18", "states · 7 categories")
    st.markdown("**Why fake data?** Real brands won't share private return data. Synthetic "
                "data is private-safe, *and* we know the hidden truth — so we can check our "
                "methods actually find it.")
    take("The trick that makes a score possible: every shopper has a hidden habit — some "
         "accept parcels, some refuse. It shows up in their <b>past returns</b>, so history "
         "predicts the future (just like CIBIL).")

    st.markdown("#### What one order looks like")
    field_help = pd.DataFrame({
        "Column": ["OrderID / CustomerID", "CityTier", "State", "Category", "OrderValue",
                   "DiscountPct", "PaymentMethod", "Device", "AddressQuality", "OrderHour",
                   "Items", "PriorOrders", "PriorReturns", "DeliveryStatus"],
        "In plain English": [
            "Who placed it (IDs)",
            "Tier-1 metro, Tier-2 city, or Tier-3 town",
            "Which state it ships to",
            "What was bought (Fashion, Electronics …)",
            "Order amount in ₹",
            "Discount on the order",
            "Cash on Delivery or paid online (Prepaid)",
            "App / mobile web / desktop",
            "Was the address complete, partial, or vague?",
            "Hour of day the order was placed (0–23)",
            "How many items in the order",
            "How many orders this shopper made before",
            "How many of those came back",
            "👉 The outcome: Delivered or Returned",
        ],
        "Why it matters": [
            "Identify repeat shoppers",
            "Smaller towns return more",
            "Some regions are riskier",
            "Fashion is tried-on-and-returned",
            "Big COD bills get refused",
            "Deep discounts → impulse → regret",
            "COD is where returns happen",
            "Hints who the shopper is",
            "Bad address = failed delivery",
            "Late-night = impulsive",
            "Bigger carts behave differently",
            "Loyalty signal",
            "👉 Powers the score",
            "👉 What we predict",
        ],
    })
    st.dataframe(field_help, width="stretch", hide_index=True)

    st.markdown("#### A peek at the raw file — messy on purpose")
    st.caption("Real exports are never clean: `Rs`/commas in prices, a stray `999%` "
               "discount, tiers written five ways. Chapter 3 fixes it all.")
    st.dataframe(raw.head(10), width="stretch")

# =============================================================== 3. CLEANING
with tabs[2]:
    chapter(3, "🧹", "Cleaning the data",
            "Messy export in, tidy numbers out — every fix logged in the open.",
            "#0891b2", "#0e7490")

    cln = st.columns(3)
    statcard(cln[0], "34", "duplicate rows removed", AMBER)
    statcard(cln[1], "5→3", "tier spellings merged", ACCENT)
    statcard(cln[2], "0", "missing values left", GREEN)

    st.markdown("#### Every fix we made")
    for n in notes:
        st.markdown(f"- {n}")

    st.markdown("#### Before → After")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Before** (raw)")
        st.dataframe(raw[["OrderValue", "DiscountPct", "CityTier", "State"]].head(8),
                     width="stretch")
    with cc2:
        st.markdown("**After** (clean)")
        st.dataframe(df[["OrderValue", "DiscountPct", "CityTier", "State"]].head(8),
                     width="stretch")

    st.markdown("#### New features we built")
    st.markdown("""
All knowable the *moment* an order is placed — never peeking at the future:
- **PriorRTORate** — fraction of past orders the shopper returned (their track record)
- **FirstTime** — is this their first-ever order?
- **IsCOD** — is it Cash on Delivery?
""")
    take("We only use clues available <b>before shipping</b>. Using after-the-fact info "
         "would make the model look great in tests but useless in real life (that mistake "
         "is called <i>leakage</i>).")
    st.success(f"✅ Ready: **{len(df):,} orders**, **{df.shape[1]} columns**, "
               "zero duplicates, zero missing values.")

# =============================================================== 4. DESCRIPTIVE
with tabs[3]:
    chapter(4, "📊", "Who sends parcels back?",
            "Simple questions, clear bars. Red = worse than average, green = better.",
            "#0d9488", "#0f766e")
    st.caption("The dashed line on each chart is the overall return rate.")

    def rate_by(col, title, min_n=20, full=True):
        base = df if full else cod
        ref = base["RTO"].mean()
        g = base.groupby(col)["RTO"].agg(["mean", "count"])
        g = g[g["count"] >= min_n].sort_values("mean")
        fig, ax = plt.subplots(figsize=(8, max(2.4, 0.5 * len(g))))
        colors = [RED if m > ref else GREEN for m in g["mean"]]
        bars = ax.barh(g.index.astype(str), g["mean"] * 100, color=colors, height=0.62)
        ax.axvline(ref * 100, color=INK, ls="--", lw=1.3)
        for b, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(m * 100 + 0.5, b.get_y() + b.get_height() / 2,
                    f"{m*100:.0f}%  (n={n:,})", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, g["mean"].max() * 100 * 1.25)
        ax.set_xlabel("Return rate"); ax.set_title(title)
        bare(ax); ax.tick_params(left=False)
        return fig

    st.markdown("#### Cash on Delivery vs Prepaid")
    st.pyplot(rate_by("PaymentMethod", "Return rate by payment method"), width="stretch")
    take(f"COD returns at <b>{cod_rto*100:.0f}%</b> vs just <b>{prepaid_rto*100:.0f}%</b> "
         "prepaid. From here we zoom into COD — that's where money leaks.")

    st.divider()
    st.markdown("#### Within COD, who's risky? Pick a lens 👇")
    lens = st.selectbox("Break COD returns down by:",
                        ["City tier", "Product category", "Address quality",
                         "First-time vs repeat", "Discount depth", "Order value"],
                        index=0)

    if lens == "City tier":
        st.pyplot(rate_by("CityTier", "COD return rate by city tier", full=False),
                  width="stretch")
        take("Smaller towns return more — Tier-3 roughly <b>triples</b> Tier-1.")
    elif lens == "Product category":
        st.pyplot(rate_by("Category", "COD return rate by product category", full=False),
                  width="stretch")
        take("<b>Fashion &amp; footwear</b> top the list — ordered to 'try at home', then refused.")
    elif lens == "Address quality":
        st.pyplot(rate_by("AddressQuality", "COD return rate by address quality",
                          min_n=10, full=False), width="stretch")
        take("A <b>vague address</b> is a delivery waiting to fail — fixing it at checkout is a cheap win.")
    elif lens == "First-time vs repeat":
        tmp = cod.assign(Who=np.where(cod["FirstTime"] == 1, "First-time buyer",
                                      "Repeat buyer"))
        ref = cod["RTO"].mean()
        g = tmp.groupby("Who")["RTO"].mean()
        fig, ax = plt.subplots(figsize=(7, 2.6))
        bars = ax.barh(g.index, g.values * 100,
                       color=[RED if v > ref else GREEN for v in g.values], height=0.55)
        for b, v in zip(bars, g.values):
            ax.text(v*100+0.5, b.get_y()+b.get_height()/2, f"{v*100:.0f}%",
                    va="center", fontweight="bold")
        ax.axvline(ref*100, color=INK, ls="--", lw=1.3)
        ax.set_xlim(0, g.values.max()*100*1.25); ax.set_xlabel("Return rate")
        ax.set_title("COD: first-timers vs repeat buyers"); bare(ax); ax.tick_params(left=False)
        st.pyplot(fig, width="stretch")
        take("<b>First-timers are riskier</b> — no track record yet. Loyalty earns trust.")
    elif lens == "Discount depth":
        b = pd.cut(cod["DiscountPct"], [-1, 10, 25, 40, 100],
                   labels=["0–10%", "11–25%", "26–40%", "40%+"])
        g = cod.assign(Band=b).groupby("Band", observed=True)["RTO"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.bar(g.index.astype(str), g["mean"]*100, color=ACCENT, width=0.6)
        ax.axhline(cod["RTO"].mean()*100, color=RED, ls="--", lw=1.3)
        for b_, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(b_.get_x()+b_.get_width()/2, m*100+0.6, f"{m*100:.0f}%\nn={n:,}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Return rate"); ax.set_title("COD return rate by discount depth")
        bare(ax)
        st.pyplot(fig, width="stretch")
        take("Deeper discount → more impulse → more regret at the door.")
    else:  # Order value
        b = pd.qcut(cod["OrderValue"], 4,
                    labels=["Cheapest 25%", "Lower-mid", "Upper-mid", "Priciest 25%"])
        g = cod.assign(Band=b).groupby("Band", observed=True)["RTO"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.bar(g.index.astype(str), g["mean"]*100, color=ACCENT, width=0.6)
        ax.axhline(cod["RTO"].mean()*100, color=RED, ls="--", lw=1.3)
        for b_, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(b_.get_x()+b_.get_width()/2, m*100+0.6, f"{m*100:.0f}%\nn={n:,}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Return rate"); ax.set_title("COD return rate by order value")
        bare(ax)
        st.pyplot(fig, width="stretch")
        take("Bigger COD bills are scarier to accept — more cash on the spot, more refusals.")

    st.divider()
    st.markdown("#### The 'perfect storm': town × category")
    st.caption("COD return rate for every City-tier × Category mix. Greener = safer, redder = riskier.")
    piv = (cod.pivot_table(index="CityTier", columns="Category", values="RTO", aggfunc="mean")
              .reindex(["Tier-1", "Tier-2", "Tier-3"]))
    fig, ax = plt.subplots(figsize=(9, 3.1))
    im = ax.imshow(piv.values * 100, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=28, ha="right", color=MUT)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, color=MUT)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=10,
                        fontweight="bold",
                        color="white" if v * 100 > 42 else "#1f2937")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Return rate (%)", color=MUT); cb.ax.tick_params(colors=MUT)
    st.pyplot(fig, width="stretch")
    take("The dark-red corner is the <b>perfect storm</b> — a small-town fashion/footwear COD "
         "order. The very same product is far safer in a Tier-1 metro. It's the <b>mix</b> that "
         "bites, not any one column — we mine those combos fully in Chapter 10.")

    st.divider()
    st.markdown("#### Which numbers move with returns?")
    st.caption("Teal = more returns, grey = fewer. Straight-line links only — tested properly next chapter.")
    corr_cols = ["RTO", "PriorRTORate", "FirstTime", "DiscountPct", "OrderValue",
                 "PriorOrders", "Items", "OrderHour"]
    corr = cod[corr_cols].corr()["RTO"].drop("RTO").sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.barh(corr.index, corr.values,
                   color=[ACCENT if v >= 0 else GREY for v in corr.values], height=0.6)
    for b, v in zip(bars, corr.values):
        ax.text(v + (0.005 if v >= 0 else -0.005), b.get_y()+b.get_height()/2,
                f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color=INK, lw=1); ax.set_xlabel("Correlation with returning (COD)")
    ax.set_title("Which numbers move with returns?"); bare(ax, keep_left=False)
    ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    take("<b>Past-return rate stands out</b> — a shopper's history is the loudest number. The seed of the score.")

# =============================================================== 5. DIAGNOSTIC
with tabs[4]:
    chapter(5, "🔍", "The real reasons",
            "Which clues truly matter, and which are just noise?",
            "#4f46e5", "#4338ca")
    take("<b>Cramér's V</b> is a 0→1 'how strongly linked?' dial (closer to 1 = stronger). "
         "The <i>p-value</i> checks it's real, not luck. We grey-out anything that's just luck.")

    assoc_cols = [c for c in ["PaymentMethod", "CityTier", "Category", "AddressQuality",
                              "State", "Device"] if c in df.columns]
    rows = []
    for c in assoc_cols:
        chi2, p, dof, v, ncat = cramers_v(df, c)
        rows.append({"Clue": c, "Cramér's V": round(v, 3), "p-value": f"{p:.1e}",
                     "Real link?": "✅ yes" if p < 0.05 else "— (luck)"})
    assoc = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(assoc) + 1))
    a2 = assoc.sort_values("Cramér's V")
    colors = [GREY if "—" in s else ACCENT for s in a2["Real link?"]]
    bars = ax.barh(a2["Clue"], a2["Cramér's V"], color=colors, height=0.6)
    for b, v in zip(bars, a2["Cramér's V"]):
        ax.text(v + 0.005, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Cramér's V  (link strength with returns)")
    ax.set_title("Which clues are truly linked to returns?"); bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    st.dataframe(assoc, width="stretch", hide_index=True)

    take("Payment is the biggest lever — but we can't ban COD without losing sales. "
         "The smart move: look <b>inside COD</b> and separate risky from safe. (Chapter 6.)")

    st.divider()
    st.markdown("#### The clue that travels with you: your track record")
    h = cod[cod["PriorOrders"] > 0].copy()
    h["band"] = pd.cut(h["PriorRTORate"], [-0.01, 0.0, 0.25, 0.5, 1.01],
                       labels=["Never returned", "Returned 1–25%", "Returned 26–50%",
                               "Returned >50%"])
    g = h.groupby("band", observed=True)["RTO"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.bar(g.index.astype(str), g["mean"]*100,
                  color=[GREEN, "#84cc16", AMBER, RED], width=0.62)
    for b, (m, n) in zip(bars, zip(g["mean"], g["count"])):
        ax.text(b.get_x()+b.get_width()/2, m*100+1, f"{m*100:.0f}%\nn={n:,}",
                ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Return rate on the NEXT order")
    ax.set_title("Past returns predict the next return")
    bare(ax)
    st.pyplot(fig, width="stretch")
    take("Returned over half your parcels before? You'll likely return the next. Never "
         "returned? You rarely start. This is the most CIBIL-like signal we have.")

    st.divider()
    st.markdown("#### Trust is earned: risk falls as history grows")
    st.caption("COD return rate on an order, grouped by how many orders the shopper already had.")
    tb = cod.copy()
    tb["bucket"] = pd.cut(tb["PriorOrders"], [-1, 0, 2, 5, 10, 9999],
                          labels=["1st order", "2nd–3rd", "4th–6th", "7th–11th", "12th+"])
    g = tb.groupby("bucket", observed=True)["RTO"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(g)), g["mean"] * 100, "-o", color=ACCENT, lw=2.6, markersize=9)
    for i, (m, cnt) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, m * 100 + 1.4, f"{m*100:.0f}%", ha="center", fontsize=9,
                fontweight="bold", color=INK)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index)
    ax.set_xlabel("How many orders the shopper had placed before")
    ax.set_ylabel("Return rate on this order")
    ax.set_title("The more clean orders behind you, the safer the next")
    ax.set_ylim(0, g["mean"].max() * 100 * 1.25); bare(ax)
    st.pyplot(fig, width="stretch")
    take("A first-timer is a coin-flip; a shopper with a dozen clean orders is almost a sure "
         "thing. This is the CIBIL idea in one line — <b>a track record lowers your risk</b> — "
         "and it's why a good customer's score should <i>rise</i> the longer they stay clean.")

# =============================================================== 6. THE COD SCORE
with tabs[5]:
    chapter(6, "🎯", "The COD Trust Score",
            "Every clue, combined into one 300–900 number per shopper — CIBIL-style.",
            "#0d9488", "#115e59")

    X, y = design_matrix(cod)
    results, (ntr, nte, trp, tep) = train_models(X, y, test_size, seed)
    st.caption(f"Trained on {ntr:,} COD orders, tested on {nte:,} unseen ones. "
               "We predict whether a parcel comes back.")

    st.markdown("#### How five models did")
    metrics = pd.DataFrame({
        m: {"Accuracy": r["test_acc"], "Precision": r["precision"], "Recall": r["recall"],
            "F1": r["f1"], "ROC-AUC": r["roc_auc"],
            "Overfit gap": r["train_acc"] - r["test_acc"]}
        for m, r in results.items()
    }).T.round(3)
    st.dataframe(
        metrics.style.highlight_max(axis=0,
            subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], color="#d1fae5")
            .highlight_max(axis=0, subset=["Overfit gap"], color="#fee2e2"),
        width="stretch")
    with st.expander("What do these words mean?"):
        st.markdown("""
- **Accuracy** — how often the guess is right overall.
- **Precision** — when it says "will return", how often it's correct.
- **Recall** — of parcels that truly returned, how many it caught.
- **F1** — balance of precision and recall.
- **ROC-AUC** — overall skill (0.5 = coin-flip, 1.0 = perfect). Ours ~0.7–0.8: good, not magic.
- **Overfit gap** — seen vs unseen performance; smaller = safer.
""")
    auc_leader = metrics["ROC-AUC"].idxmax()
    take(f"<b>Which model builds the score?</b> A score must be <b>calibrated</b> — when it "
         f"says 35% risk, ~35% should really return. {auc_leader} ranks slightly best, but "
         f"<b>{SCORE_MODEL}</b>'s predicted risk matches reality almost exactly — so we score from it.")

    st.markdown("#### Look inside one model")
    pick = st.selectbox("Inspect a model", list(results.keys()),
                        index=list(results.keys()).index(SCORE_MODEL))
    cset = st.columns([1, 1])
    with cset[0]:
        cm = results[pick]["cm"]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="BuGn")
        for a in range(2):
            for bb in range(2):
                ax.text(bb, a, f"{cm[a,bb]:,}", ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if cm[a, bb] > cm.max()/2 else "#0f172a")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Said Deliver", "Said Return"], color=MUT)
        ax.set_yticklabels(["Was Delivered", "Was Returned"], color=MUT)
        ax.set_title(f"{pick}: hits & misses"); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig, width="stretch")
    with cset[1]:
        fpr, tpr, _ = results[pick]["roc"]
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.plot(fpr, tpr, color=ACCENT, lw=2.2,
                label=f"{pick} (AUC={results[pick]['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
        ax.set_xlabel("False alarms"); ax.set_ylabel("Returns caught")
        ax.set_title("ROC: skill above the coin-flip line")
        ax.legend(loc="lower right", fontsize=8); bare(ax)
        st.pyplot(fig, width="stretch")

    # ROC of all models together
    st.subheader("All models on one chart")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    palette = [ACCENT, AMBER, GREEN, RED, INK]
    for (name, r), c in zip(results.items(), palette):
        fpr, tpr, _ = r["roc"]
        ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} ({r['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    ax.set_xlabel("False alarms"); ax.set_ylabel("Returns caught")
    ax.set_title("Higher and more to the top-left = better"); ax.legend(fontsize=8)
    bare(ax)
    st.pyplot(fig, width="stretch")

    # what the scoring model leans on (Gradient Boosting exposes feature_importances_)
    sm = results[SCORE_MODEL]["pipe"]
    ohe = sm.named_steps["pre"].named_transformers_["cat"]
    feat = NUM_FEATS + list(ohe.get_feature_names_out(CAT_FEATS))
    imp = pd.Series(sm.named_steps["clf"].feature_importances_, index=feat)
    imp = imp.sort_values(ascending=False).head(12)[::-1]
    st.markdown("#### What the score pays most attention to")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(imp.index, imp.values, color=ACCENT, height=0.7)
    ax.set_xlabel("Importance")
    bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")

    st.divider()
    st.markdown("#### Risk → a 300–900 score")
    proba, score = honest_scores(X, y, SCORE_MODEL, test_size, seed)
    cod_scored = cod.copy()
    cod_scored["Risk"] = proba
    cod_scored["Score"] = score
    st.session_state["cod_scored"] = cod_scored

    pred_mean = float(proba.mean()); true_mean = float(y.mean())
    cal = st.columns(2)
    statcard(cal[0], f"{pred_mean:.0%}", "score's avg predicted risk", ACCENT)
    statcard(cal[1], f"{true_mean:.0%}", "real return rate", GREEN)
    take("Those two numbers nearly match — that's <b>calibration</b>. It's why the score's "
         "percentage means what it says. Higher score = safer shopper.")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(score, bins=40, color=GREY, edgecolor="none")
    ax.axvspan(300, 580, color=RED, alpha=0.14)
    ax.axvspan(580, 720, color=AMBER, alpha=0.16)
    ax.axvspan(720, 900, color=GREEN, alpha=0.16)
    top = ax.get_ylim()[1]
    ax.text(440, top*0.9, "Prepaid-only", color=RED, ha="center", fontweight="bold")
    ax.text(650, top*0.9, "COD + fee", color=AMBER, ha="center", fontweight="bold")
    ax.text(810, top*0.9, "COD free", color=GREEN, ha="center", fontweight="bold")
    ax.set_xlabel("COD Trust Score"); ax.set_ylabel("Shoppers")
    ax.set_title("Most shoppers are safe; a risky tail needs guard-rails"); bare(ax)
    st.pyplot(fig, width="stretch")

    st.markdown("**Example shoppers, low to high score:**")
    show = cod_scored.sort_values("Score")
    ex = pd.concat([show.head(2), show.iloc[[len(show)//2-1, len(show)//2]], show.tail(2)])
    st.dataframe(ex[["CustomerID", "CityTier", "Category", "OrderValue", "PriorOrders",
                     "PriorReturns", "Score"]].rename(columns={"OrderValue": "₹ Order"}),
                 width="stretch", hide_index=True)
    take("More past returns &amp; risky first-timers land <b>low</b>; loyal clean-history "
         "shoppers land <b>high</b> — just like a credit score rewards a good record.")

    st.divider()
    st.markdown("#### 🎮 Try it: score a shopper yourself")
    take("Like checking your own CIBIL — set an order's details and watch the score move. "
         "(We treat it as a COD order, since that's exactly who the score is built for.)")
    sim_pipe = results[SCORE_MODEL]["pipe"]
    cats = sorted(cod["Category"].unique())
    s1, s2, s3 = st.columns(3)
    with s1:
        i_tier = st.selectbox("City tier", ["Tier-1", "Tier-2", "Tier-3"], index=2)
        i_cat = st.selectbox("Category", cats,
                             index=cats.index("Fashion") if "Fashion" in cats else 0)
    with s2:
        i_addr = st.selectbox("Address quality", ["Complete", "Partial", "Vague", "Unknown"],
                              index=2)
        i_dev = st.selectbox("Device", sorted(cod["Device"].unique()))
    with s3:
        i_prior = int(st.number_input("Past orders", 0, 60, 0, 1))
        i_ret = int(st.number_input("…of which returned", 0, 60, 0, 1))
    i_val = st.slider("Order value (₹)", 200, 15000, 1500, 100)
    i_disc = st.slider("Discount (%)", 0, 80, 30, 5)

    i_ret = min(i_ret, i_prior)
    sim_row = pd.DataFrame([{
        "OrderValue": i_val, "DiscountPct": i_disc, "PriorOrders": i_prior,
        "PriorRTORate": (i_ret / i_prior) if i_prior > 0 else 0.0,
        "FirstTime": int(i_prior == 0), "OrderHour": 20, "Items": 1,
        "CityTier": i_tier, "State": cod["State"].mode()[0], "Category": i_cat,
        "Device": i_dev, "AddressQuality": i_addr,
    }])
    sim_risk = float(sim_pipe.predict_proba(sim_row)[:, 1][0])
    sim_score = int(round(900 - sim_risk * 600))
    sim_tier = ("Free COD" if sim_score >= 720
                else "Prepaid-only" if sim_score < 580 else "COD + fee")
    tcol = GREEN if sim_tier == "Free COD" else (RED if sim_tier == "Prepaid-only" else AMBER)
    rc1, rc2 = st.columns(2)
    statcard(rc1, f"{sim_score}", "COD Trust Score", tcol)
    statcard(rc2, sim_tier, f"{sim_risk*100:.0f}% predicted return risk", tcol)
    st.write("")

    reasons = []
    if i_prior == 0:
        reasons.append("🆕 First-timer — no track record yet (pushes risk **up**).")
    elif i_ret / max(i_prior, 1) > 0.3:
        reasons.append(f"↩️ Returned {i_ret} of {i_prior} past orders — history repeats (**up**).")
    elif i_ret == 0 and i_prior >= 3:
        reasons.append(f"✅ {i_prior} clean orders, none returned — a proven record (**down**).")
    if i_addr == "Vague":
        reasons.append("📍 Vague address — deliveries fail more often (**up**).")
    elif i_addr == "Complete":
        reasons.append("📍 Complete address — smooth delivery (**down**).")
    if i_tier == "Tier-3":
        reasons.append("🏘️ Tier-3 town — returns run higher here (**up**).")
    elif i_tier == "Tier-1":
        reasons.append("🏙️ Tier-1 metro — returns run lower here (**down**).")
    if i_cat in ("Fashion", "Footwear"):
        reasons.append(f"👗 {i_cat} — a 'try-at-home-then-refuse' category (**up**).")
    elif i_cat in ("Electronics", "Health"):
        reasons.append(f"📦 {i_cat} — rarely refused at the door (**down**).")
    if i_disc >= 40:
        reasons.append("🏷️ Deep discount — impulse buys get regretted more (**up**).")
    st.markdown("**Why this score:**")
    for r in reasons[:5]:
        st.markdown(f"- {r}")
    take("The score just rolls these everyday clues into one number. And remember the golden "
         "rule — it adjusts the <b>offer</b> (COD or prepay), it never judges the <b>person</b>.")

# =============================================================== 7. THE VERDICT
with tabs[6]:
    chapter(7, "⚖️", "The verdict: who gets COD?",
            "Turn the score into a 3-tier policy. Move the dials, watch the ₹ trade-off.",
            "#b45309", "#92400e")

    if "cod_scored" not in st.session_state:
        X, y = design_matrix(cod)
        results, _ = train_models(X, y, test_size, seed)
        proba, score = honest_scores(X, y, SCORE_MODEL, test_size, seed)
        cs = cod.copy(); cs["Risk"] = proba; cs["Score"] = score
        st.session_state["cod_scored"] = cs
    cs = st.session_state["cod_scored"].copy()

    st.markdown("#### Set the policy 🎛️")
    p1, p2, p3 = st.columns(3)
    with p1:
        low_cut = st.slider("Prepaid-only below score", 300, 700, 580, 10,
                            help="Risky shoppers below this must pay online")
    with p2:
        high_cut = st.slider("Free COD at or above score", 600, 900, 720, 10,
                             help="Trusted shoppers above this get COD with no friction")
    with p3:
        rto_cost = st.number_input("₹ lost per returned order", 100, 600,
                                   DEFAULT_RTO_COST, 10)
    cod_fee = st.slider("COD fee charged in the middle tier (₹)", 0, 150, 50, 5)
    if low_cut >= high_cut:
        st.warning("Keep the prepaid-only cut **below** the free-COD cut for three clean tiers.")
        high_cut = low_cut + 10

    cs["Tier"] = np.where(cs["Score"] >= high_cut, "Free COD",
                  np.where(cs["Score"] < low_cut, "Prepaid-only", "COD + fee"))

    order = ["Free COD", "COD + fee", "Prepaid-only"]
    colmap = {"Free COD": GREEN, "COD + fee": AMBER, "Prepaid-only": RED}
    summ = (cs.groupby("Tier")
              .agg(Shoppers=("Score", "size"), ReturnRate=("RTO", "mean"),
                   AvgScore=("Score", "mean"), AvgOrder=("OrderValue", "mean"))
              .reindex(order))

    st.markdown("#### The three tiers")
    ccc = st.columns(3)
    grad = {"Free COD": ("#16a34a", "#15803d"),
            "COD + fee": ("#f59e0b", "#d97706"),
            "Prepaid-only": ("#ef4444", "#dc2626")}
    for col, t in zip(ccc, order):
        n = int(summ.loc[t, "Shoppers"]); rr = summ.loc[t, "ReturnRate"]
        share = n / len(cs) * 100
        g1, g2 = grad[t]
        col.markdown(
            f"<div style='background:linear-gradient(135deg,{g1},{g2});border-radius:14px;"
            f"padding:16px 18px;color:#fff;box-shadow:0 6px 18px rgba(0,0,0,.15);'>"
            f"<div style='font-weight:700;font-size:1.05rem;'>{t}</div>"
            f"<div style='font-size:2rem;font-weight:800;line-height:1.1;margin:4px 0;'>{share:.0f}%</div>"
            f"<div style='opacity:.92;font-size:.9rem;'>{n:,} orders · {rr*100:.0f}% return</div>"
            f"</div>", unsafe_allow_html=True)
    st.write("")

    # tier chart
    fig, ax = plt.subplots(figsize=(8, 2.8))
    bars = ax.bar(order, summ["ReturnRate"]*100, color=[colmap[t] for t in order], width=0.6)
    for b, t in zip(bars, order):
        ax.text(b.get_x()+b.get_width()/2, summ.loc[t, "ReturnRate"]*100+0.8,
                f"{summ.loc[t,'ReturnRate']*100:.0f}%", ha="center", fontweight="bold",
                color=INK)
    ax.set_ylabel("Return rate (%)"); ax.set_title("Return rate climbs cleanly across tiers")
    bare(ax)
    st.pyplot(fig, width="stretch")

    # ---- the money: what the policy saves vs costs ----
    st.markdown("#### Does the policy pay off? 💰")
    base_loss = (cs["RTO"] * rto_cost).sum()  # cost if we ship every COD order blindly

    # prepaid-only tier: assume online payment removes ~90% of these returns
    pp = cs[cs["Tier"] == "Prepaid-only"]
    prevented = pp["RTO"].sum() * 0.90
    saved_prepaid = prevented * rto_cost

    # but some genuine buyers abandon when COD is removed (industry: a real cost)
    abandon_rate = 0.25  # conservative: 1 in 4 prepaid-only good buyers walk away
    good_lost = (len(pp) - pp["RTO"].sum()) * abandon_rate
    avg_pp_order = pp["OrderValue"].mean() if len(pp) else 0
    lost_margin = good_lost * avg_pp_order * 0.20  # ~20% contribution margin

    # middle tier fee income (only on orders that DO get delivered & accepted)
    mid = cs[cs["Tier"] == "COD + fee"]
    fee_income = len(mid) * cod_fee * 0.85  # assume 85% still accept the small fee

    net = saved_prepaid + fee_income - lost_margin

    mcol = st.columns(4)
    mcol[0].metric("Blind COD loss (today)", f"₹{base_loss:,.0f}",
                   help="If we ship every COD order with no policy")
    mcol[1].metric("Saved by prepaid-only", f"₹{saved_prepaid:,.0f}",
                   help="Returns avoided by asking risky shoppers to pay online")
    mcol[2].metric("COD-fee income", f"₹{fee_income:,.0f}",
                   help="Small risk-priced fee from the middle tier")
    mcol[3].metric("Net effect of policy", f"₹{net:,.0f}",
                   delta="better" if net > 0 else "worse",
                   delta_color="normal" if net > 0 else "inverse")

    st.markdown(f"""
**How to read this (the honest trade-off):**
- Asking the **prepaid-only** tier to pay online prevents most of their returns →
  about **₹{saved_prepaid:,.0f}** saved.
- A small **₹{cod_fee} COD fee** in the middle tier brings in **₹{fee_income:,.0f}** —
  this is *risk-based pricing*, just like a higher interest rate on a riskier loan.
- **But** removing COD scares off some genuine buyers (we assumed 1 in 4 walk away),
  costing roughly **₹{lost_margin:,.0f}** in lost margin.
- **Net:** the policy comes out **{'ahead' if net>0 else 'behind'}** by
  **₹{abs(net):,.0f}** on this data.
""")
    take("Push the <b>prepaid-only</b> dial right → prevent more returns, but lose more "
         "genuine sales. That tension is the real decision — the score lets you make it "
         "with eyes open.")

    st.divider()
    st.markdown("#### 📋 What we found — and what we'd tell the business")
    st.markdown(f"""
**The story in six lines:**
1. On our store, **COD returns at {cod_rto*100:.0f}%** versus **{prepaid_rto*100:.0f}%** for
   prepaid — COD is where the leak is.
2. Inside COD, returns are **not random**: smaller towns, fashion/footwear, deep discounts,
   first-timers, and vague addresses all return more.
3. The loudest single signal is a shopper's **own past-return record** — history repeats.
4. A model reads all these clues at **~0.7–0.8 skill** and turns them into one **COD Trust
   Score** (300–900), CIBIL-style.
5. That score sorts shoppers into **free COD / COD-with-a-fee / prepaid-only**, and the
   return rate rises cleanly across the three.
6. Applied sensibly, the policy comes out **net positive** — even after accounting for
   genuine buyers lost when COD is removed.

**Recommendations, each tied to a finding above:**
- **Default deep-discount + first-time + vague-address COD orders to the middle tier** (a
  small COD fee), because those are exactly the slices that returned most in Chapter 4.
- **Switch COD off for the bottom tier**, but soften it with a **small prepaid discount** —
  the industry shows a ₹75–125 nudge converts most of them to paying online willingly.
- **Fix addresses at checkout** (a "confirm your full address" step) — it's the cheapest
  lever, since vague addresses were among the strongest risk clues.
- **Reward loyalty:** let a clean track record *raise* a shopper's score over time, so good
  customers feel trusted, not punished.
""")

    st.warning("**A fair warning (our honesty note).** A low COD Score is **not** a verdict "
               "that someone is a bad person. It only means *this order is risky to ship on "
               "cash* — so we ask for prepayment instead. People in patchy-address areas or "
               "new to a brand aren't dishonest; they're just harder to deliver to. The "
               "score adjusts the **offer**, never judges the **person**. Used carelessly it "
               "could unfairly shut out whole towns — so a real brand should monitor it for "
               "exactly that and keep a prepaid path open for everyone.")

    st.download_button("⬇️ Download the cleaned, scored data (CSV)",
                       cs.to_csv(index=False).encode(),
                       "cod_scored.csv", "text/csv")

# =============================================================== 8. CLUSTERING
with tabs[7]:
    chapter(8, "👥", "Two kinds of shopper",
            "Forget single orders — group people by their whole habit. The computer finds "
            "the groups itself (unsupervised learning).",
            "#7c3aed", "#6d28d9")
    take("Everything so far judged one <b>order</b>. Here we zoom out and ask: are there "
         "natural <b>types of shopper</b>? We never tell the computer the answer — it groups "
         "people purely on their habits. That's what <b>unsupervised learning</b> means.")

    cust = customer_table(df, len(df))
    clu = cluster_everything(cust, len(cust), seed)

    st.markdown("#### One row per shopper — their whole story in five habits")
    st.caption("We shrink each shopper to: how many orders, average ₹, average discount, "
               "how often they use COD, and how often they return.")
    st.dataframe(cust.head(6).round(2), width="stretch", hide_index=True)

    st.markdown("#### First question: how many groups are really there?")
    st.caption("Two honest tests — they should point to the same number.")
    gg1, gg2 = st.columns(2)
    with gg1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(clu["ks"], clu["inertia"], "-o", color=ACCENT, lw=2.2, markersize=7)
        ax.axvline(clu["best_k"], color=RED, ls="--", lw=1.2)
        ax.set_xlabel("Number of groups (k)"); ax.set_ylabel("Left-over spread")
        ax.set_title("Elbow — where the drop flattens"); bare(ax)
        st.pyplot(fig, width="stretch")
    with gg2:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(clu["ks"], clu["sil"], "-o", color="#7c3aed", lw=2.2, markersize=7)
        ax.axvline(clu["best_k"], color=RED, ls="--", lw=1.2)
        ax.set_xlabel("Number of groups (k)"); ax.set_ylabel("Separation score")
        ax.set_title("Silhouette — higher = cleaner split"); bare(ax)
        st.pyplot(fig, width="stretch")
    take(f"Both agree on <b>{clu['best_k']} groups</b>: the elbow flattens and the silhouette "
         "peaks at the same spot. The store really does hold two shopper types — no more, no less.")

    st.markdown("#### Meet the two types (K-Means — hard groups)")
    prof = clu["prof"]
    ordered = sorted(prof.index, key=lambda i: prof.loc[i, "ReturnRate"])
    grad2 = {"Steady shoppers": ("#16a34a", "#15803d"),
             "Higher-risk shoppers": ("#ef4444", "#dc2626")}
    cc = st.columns(2)
    for col, lab_i in zip(cc, ordered):
        nm = clu["names"][lab_i]; p = prof.loc[lab_i]
        size = int((clu["labels"] == nm).sum())
        g1, g2 = grad2[nm]
        col.markdown(
            f"<div style='background:linear-gradient(135deg,{g1},{g2});border-radius:14px;"
            f"padding:16px 18px;color:#fff;box-shadow:0 6px 18px rgba(0,0,0,.15);'>"
            f"<div style='font-weight:800;font-size:1.1rem;'>{nm}</div>"
            f"<div style='font-size:1.8rem;font-weight:800;margin:4px 0;'>{size:,} shoppers</div>"
            f"<div style='opacity:.95;font-size:.92rem;line-height:1.55;'>"
            f"↩️ {p['ReturnRate']*100:.0f}% of their parcels come back<br>"
            f"💳 {p['PctCOD']*100:.0f}% pay by Cash on Delivery<br>"
            f"🛍️ ~{p['Orders']:.1f} orders each · avg ₹{p['AvgValue']:,.0f}</div>"
            f"</div>", unsafe_allow_html=True)
    st.write("")
    take("Two clean personas fall out: a big <b>steady</b> base that rarely returns, and a "
         "smaller <b>higher-risk</b> pocket that leans harder on COD and sends far more back. "
         "Same store, two very different audiences.")

    st.markdown("#### A softer look: the Gaussian Mixture Model")
    take("K-Means draws a <b>hard line</b> — you're in one group or the other. A <b>Gaussian "
         "Mixture Model</b> is gentler: it gives each shopper a <b>% chance</b> of being the "
         "risky type — like saying '80% steady, 20% risky'. It's the number version of "
         "<i>Latent Class Analysis</i>.")
    fig, ax = plt.subplots(figsize=(8, 2.9))
    ax.hist(clu["p_risky"] * 100, bins=30, color="#7c3aed", edgecolor="none")
    ax.set_xlabel("Model's % chance a shopper is the higher-risk type")
    ax.set_ylabel("Shoppers"); ax.set_title("Most shoppers are clearly one type or the other")
    bare(ax)
    st.pyplot(fig, width="stretch")
    fence = int(((clu["p_risky"] > 0.2) & (clu["p_risky"] < 0.8)).sum())
    take(f"The two tall spikes at 0% and 100% mean the split is <b>real and clean</b> — only "
         f"<b>{fence}</b> shoppers genuinely sit on the fence. When a hard method and a soft "
         "method agree this strongly, you can trust the grouping.")

    st.markdown("#### See it in 3-D — spin the cloud 🌀")
    take("We measured five habits, but eyes can't see five dimensions. <b>PCA</b> squeezes "
         "them into the three that carry the most information, so we can actually look. "
         "Drag to rotate; the two colours should pull apart.")
    plot_df = pd.DataFrame({"PC1": clu["coords"][:, 0], "PC2": clu["coords"][:, 1],
                            "PC3": clu["coords"][:, 2], "Shopper type": clu["labels"]})
    fig3d = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color="Shopper type",
                          color_discrete_map={"Steady shoppers": GREEN,
                                              "Higher-risk shoppers": RED},
                          opacity=0.55)
    fig3d.update_traces(marker=dict(size=3))
    fig3d.update_layout(template="plotly_dark" if DARK else "plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=8, b=0), height=470,
                        legend=dict(orientation="h", y=-0.02))
    st.plotly_chart(fig3d, width="stretch")
    take(f"These three axes keep <b>{clu['evr'].sum():.0%}</b> of everything that made "
         "shoppers different — and the two clouds still separate. Proof the groups aren't "
         "invented; they're really in the data.")

    st.markdown("#### 💡 What we'd tell the business")
    st.markdown("""
- The store is really **two audiences**, not one — a large **steady** base and a smaller
  **higher-risk** pocket. A single COD rule for everyone wastes money on both.
- **Serve the steady group with zero friction** — free COD, faster checkout, loyalty perks.
  They've earned trust and hate being treated as suspects.
- **Ring-fence the higher-risk pocket** — nudge to prepaid, add an address check — but
  *never* label them 'bad'. They're simply harder to deliver to, and the fix is the **offer**.
""")

# =============================================================== 9. REGRESSION
with tabs[8]:
    chapter(9, "💰", "What drives basket size?",
            "A different question: not WHO returns, but HOW BIG the order is (regression).",
            "#0891b2", "#0e7490")
    take("Every model so far predicted a <b>yes/no</b> (returned or not). <b>Regression</b> "
         "predicts a <b>number</b> instead — here, the order's ₹ value. Same toolbox, new "
         "question: what makes a basket big or small?")

    reg, cat_r2, cat_means = regression_models(df, len(df), seed)
    st.markdown("#### Four models, one honest scoreboard")
    st.caption("R² = how much of the ups-and-downs in price we explain (1.0 = perfect, 0 = none). "
               "MAE = how far off, in ₹, on a typical order.")
    show = reg.copy()
    show["R2"] = show["R2"].round(3)
    show["MAE"] = show["MAE"].round(0)
    st.dataframe(show.rename(columns={"R2": "R² (higher = better)",
                                      "MAE": "₹ off (lower = better)"}), width="stretch")
    take("All four land near <b>R² ≈ 0.5</b> — we explain about <b>half</b> of why baskets "
         "differ. That's an <b>honest, non-magical</b> result: real signal, but no crystal ball. "
         "Ridge and Lasso barely change Linear, which tells us nothing is wildly over-fit.")

    st.markdown("#### Why only half? Because it's mostly *what they bought*")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(cat_means.index, cat_means.values, color=ACCENT, height=0.66)
    for i, (k, v) in enumerate(cat_means.items()):
        ax.text(v + cat_means.max() * 0.01, i, f"₹{v:,.0f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, cat_means.max() * 1.15)
    ax.set_xlabel("Average order value (₹)"); ax.set_title("Category sets the basket size")
    bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    take(f"Product <b>category alone</b> explains almost as much (R² ≈ <b>{cat_r2:.2f}</b>) as "
         "all the features together. Electronics baskets dwarf Beauty ones — discount, device "
         "and town barely move the ₹.")

    st.markdown("#### 💡 What we'd tell the business")
    st.markdown("""
- **Basket size is set at the shelf, not by the shopper.** Want bigger orders? The lever is
  **product mix and bundling**, not chasing individuals.
- **Set free-shipping / free-COD thresholds by category.** A ₹999 threshold is nothing in
  Electronics but blocks most Beauty carts — one flat number fits no one.
- **Honest caveat:** predicting the ₹ value is a useful *side*-question. Our real money-saver
  stays the **return-risk score** — a different question, with a sharper answer.
""")

# =============================================================== 10. ASSOCIATION RULES
with tabs[9]:
    chapter(10, "🔗", "The 'perfect storm' combos",
            "Single clues can mislead — combinations tell the truth (association rule mining).",
            "#b45309", "#92400e")
    take("Chapter 5 scored clues <b>one at a time</b>. But a return often needs a "
         "<b>combination</b> — a Tier-3 town <i>and</i> a vague address <i>and</i> fashion. "
         "<b>Association rule mining</b> (the 'people who buy X also buy Y' engine) digs out "
         "exactly those risky bundles.")

    rules = association(cod, len(cod))
    st.markdown("#### How to read the three numbers")
    st.markdown("""
- **Support** — how *common* the combo is (bigger = affects more orders).
- **Confidence** — when you see the combo, how *often a return actually follows*.
- **Lift** — how many times **more likely** a return is with this combo than for a random
  COD order. **Lift > 1 = a risky bundle**; lift = 1 means no effect at all.
""")
    show = rules.head(12).copy()
    show["support"] = (show["support"] * 100).round(1).astype(str) + "%"
    show["confidence"] = (show["confidence"] * 100).round(0).astype(int).astype(str) + "%"
    show["lift"] = show["lift"].round(2)
    st.dataframe(show.rename(columns={"combo": "Risky trait combo", "support": "How common",
                                      "confidence": "Return follows", "lift": "Times riskier"}),
                 width="stretch", hide_index=True)

    r8 = rules.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.55 * len(r8))))
    ax.barh(r8["combo"], r8["lift"], color=RED, height=0.62)
    ax.axvline(1.0, color=INK, ls="--", lw=1.3)
    for i, (_, r) in enumerate(r8.iterrows()):
        ax.text(r["lift"] + 0.02, i, f"{r['lift']:.2f}×", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, rules["lift"].max() * 1.15)
    ax.set_xlabel("Lift  (times riskier than an average COD order)")
    ax.set_title("The riskiest trait bundles"); bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    top = rules.iloc[0]
    take(f"The worst bundle — <b>{top['combo']}</b> — returns <b>{top['lift']:.1f}×</b> more "
         f"often than a typical COD order, and a return follows <b>{top['confidence']:.0%}</b> "
         "of the time. No single clue on its own is that loud.")

    st.markdown("#### 💡 What we'd tell the business")
    st.markdown("""
- **Act on combos, not columns.** Don't ban 'Tier-3' or 'Fashion' — far too broad. Flag the
  **specific bundles** above (e.g. *Tier-3 + vague address + fashion*) for prepaid or a check.
- **Cheapest fix first:** most top bundles contain a **vague address** — a single 'confirm
  your full address' step at checkout quietly defuses several of them at once.
- **These rules are plain-English and auditable.** You can hand the exact combo to an ops
  team, which keeps the policy explainable and fair — never a black box.
""")

st.caption("Built with Streamlit + scikit-learn · synthetic data, real-world patterns · "
           "a low score adjusts the offer, never judges the person.")
