# 📦 The COD Trust Score

**A CIBIL-style risk score for Cash-on-Delivery (COD) orders in Indian e-commerce.**

A storytelling Streamlit dashboard that reads a synthetic-but-realistic dataset of online
orders and predicts which COD shoppers a brand can safely deliver to — and which should
be asked to pay online instead.

> Built for the **Data Analytics – MGB** project-based learning module.

---

## The idea in one paragraph

In India, about **60–65% of online orders are Cash on Delivery**, and roughly **1 in 3 COD
parcels comes back** — the brand pays shipping both ways and earns nothing
(*Return-to-Origin*, or RTO). Killing COD is not an option; it's how Indians trust new
brands. So instead, this project gives every shopper a **300–900 COD Trust Score** —
exactly like a CIBIL credit score, but read from return history instead of repayment
history — and sorts them into three tiers:

| Score band | Tier | What the brand does |
|---|---|---|
| **720–900** | Free COD | offer COD freely |
| **580–719** | COD + fee | offer COD with a small ₹50 fee (risk-priced, like a higher interest rate) |
| **300–579** | Prepaid-only | switch COD off, ask the shopper to pay online |

---

## How the dashboard reads — the 7-chapter story

1. **📦 The Problem** — what COD returns cost Indian D2C brands (with links to real industry blogs)
2. **🧾 Meet the Data** — the synthetic store: what one order looks like, why synthetic
3. **🧹 Cleaning** — the messy export, fixed step by step, every change logged
4. **📊 What Happened** — descriptive: who sends parcels back? (interactive lens)
5. **🔍 The Real Reasons** — diagnostic: Chi-square + Cramér's V on every clue
6. **🎯 The COD Score** — five models trained, the calibrated one becomes the score
7. **⚖️ The Verdict** — interactive policy sliders, ₹ trade-off math, recommendations

Chapters **1–5** are the *individual* phase deliverable. Chapters **6–7** are the *group*
phase extension. The whole dashboard runs as one app.

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Deploy free on Streamlit Community Cloud

1. Push this whole folder to a new GitHub repo (`app.py`, `requirements.txt`,
   `cod_orders.csv`, `generate_data.py`, `README.md`, and the `.streamlit/` folder).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo
   and branch, set the main file to `app.py`.
3. Deploy. Your live URL will be `https://<your-name>-<repo>.streamlit.app`.

---

## Files in this repo

| File | What it is |
|---|---|
| `app.py` | The full 7-chapter Streamlit dashboard |
| `cod_orders.csv` | The synthetic dataset (7,235 messy rows, ~2,420 shoppers) |
| `generate_data.py` | Regenerates the dataset; prints a validation report |
| `requirements.txt` | Python dependencies (pinned ranges for Streamlit Cloud) |
| `.streamlit/config.toml` | Light, minimalist teal theme |
| `README.md` | This file |

To regenerate the dataset with different parameters:

```bash
python generate_data.py
```

---

## Method notes (the honest version)

- **Data:** 7,200 orders across 2,600 customers, 18 states, 7 categories. Every shopper
  has a hidden "reliability" trait that drives both their past returns and their current
  RTO probability — this is what makes history a learnable signal (the CIBIL mechanic).
- **Messiness on purpose:** the raw CSV has 34 duplicate rows, `Rs`/`INR`/comma values,
  `%` discounts (some impossibly >100%), five case variants for each city tier, and 7%
  missing addresses — all caught and cleaned in Chapter 3.
- **Modelling:** Logistic Regression, KNN, Decision Tree, Random Forest, Gradient
  Boosting — all trained only on **COD orders** with features knowable *before* dispatch
  (no leakage). Test ROC-AUC sits in the **0.72–0.75** band — believable, not magical.
- **Why Gradient Boosting builds the score:** Logistic Regression with balanced class
  weights is the AUC leader, but its predicted risks run hot (mean predicted ≈ 47% vs
  actual 35%) — that mis-calibration would wrongly push ~39% of shoppers to prepaid-only.
  Gradient Boosting's predicted risk almost exactly matches the real rate. **A score must
  mean what it says**, so calibration wins over a tenth of a point of AUC.
- **Score:** `score = 900 − probability × 600`, computed via 5-fold cross-validated
  predictions so no order is scored by a model that already saw it.

---

## The honesty caveat

A low COD Score is **not** a verdict that a shopper is dishonest or "bad". It only means
this particular order is risky to ship on cash — so the brand asks for prepayment instead.
The score adjusts the **offer**, never judges the **person**. Used carelessly it could
unfairly shut out whole towns or income groups; a real deployment must monitor for that
and keep a prepaid path open for everyone.

---

## Built with

[Streamlit](https://streamlit.io) · [scikit-learn](https://scikit-learn.org) ·
[pandas](https://pandas.pydata.org) · [matplotlib](https://matplotlib.org) ·
[scipy](https://scipy.org)
