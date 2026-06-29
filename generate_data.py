"""
generate_data.py  --  Synthetic data for the COD Trust Score project
=====================================================================
Builds a realistic (and realistically MESSY) dataset of Indian e-commerce
orders so we can study which Cash-on-Delivery customers are likely to send
their parcel back (Return-to-Origin, "RTO").

Design idea (this is what makes a real "score" possible):
  Every customer has a HIDDEN trait we call `reliability`. A reliable shopper
  rarely refuses a parcel; a flaky one often does. That same hidden trait
  shows up in their PAST behaviour (how often they returned before). So a
  customer's history becomes a usable signal for their future risk -- exactly
  how a CIBIL credit score reads past repayment to judge a new loan.

Nothing here is hard-coded later: the dashboard recomputes every number from
the CSV this script writes.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_CUSTOMERS = 2600
N_ORDERS    = 7200

# ---------------------------------------------------------------- pools
TIERS      = ["Tier-1", "Tier-2", "Tier-3"]
TIER_PROB  = [0.42, 0.34, 0.24]

STATES = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Telangana",
          "Uttar Pradesh", "Gujarat", "West Bengal", "Rajasthan", "Kerala",
          "Punjab", "Haryana", "Madhya Pradesh", "Bihar", "Andhra Pradesh",
          "Odisha", "Assam", "Jharkhand"]

CATEGORIES = ["Fashion", "Footwear", "Accessories", "Beauty",
              "Home & Kitchen", "Health", "Electronics"]
CAT_PROB   = [0.26, 0.13, 0.11, 0.13, 0.14, 0.09, 0.14]
# typical price band (INR) per category: (low, high)
CAT_PRICE  = {"Fashion": (499, 2999), "Footwear": (699, 3499),
              "Accessories": (299, 1799), "Beauty": (249, 1599),
              "Home & Kitchen": (399, 4999), "Health": (349, 2499),
              "Electronics": (999, 14999)}
# category effect on return risk (log-odds nudge)
CAT_RISK   = {"Fashion": 0.62, "Footwear": 0.42, "Accessories": 0.30,
              "Beauty": 0.05, "Home & Kitchen": -0.10, "Health": -0.20,
              "Electronics": -0.45}

DEVICES    = ["Android App", "iOS App", "Mobile Web", "Desktop"]
DEV_PROB   = [0.55, 0.16, 0.18, 0.11]
DEV_RISK   = {"Android App": 0.10, "iOS App": -0.20,
              "Mobile Web": 0.20, "Desktop": -0.15}

ADDR       = ["Complete", "Partial", "Vague"]
ADDR_RISK  = {"Complete": -0.30, "Partial": 0.25, "Vague": 0.70}

TIER_RISK  = {"Tier-1": -0.45, "Tier-2": 0.10, "Tier-3": 0.65}

# ---------------------------------------------------------------- customers
# Mixture of mostly-reliable shoppers + a flaky minority -> clean segments.
is_risky_type = rng.random(N_CUSTOMERS) < 0.22
reliability = np.where(
    is_risky_type,
    rng.beta(2, 4, N_CUSTOMERS),   # flaky: mean ~0.33
    rng.beta(8, 2, N_CUSTOMERS),   # reliable: mean ~0.80
)
cust_tier  = rng.choice(TIERS, N_CUSTOMERS, p=TIER_PROB)
cust_state = rng.choice(STATES, N_CUSTOMERS)
# small, fixed per-state quirk (some regions a touch riskier)
state_effect = {s: e for s, e in zip(STATES, rng.normal(0, 0.18, len(STATES)))}

# ---------------------------------------------------------------- orders
# assign each order to a customer; reliable customers order a bit more often
cust_weight = 0.5 + reliability
cust_weight /= cust_weight.sum()
order_cust = rng.choice(N_CUSTOMERS, N_ORDERS, p=cust_weight)

# give each order a day, so we can compute "history before this order"
order_day = rng.integers(0, 365, N_ORDERS)

rows = []
for oid in range(N_ORDERS):
    c = order_cust[oid]
    cat = rng.choice(CATEGORIES, p=CAT_PROB)
    lo, hi = CAT_PRICE[cat]
    value = float(rng.uniform(lo, hi))
    discount = float(np.clip(rng.normal(22, 14), 0, 80))
    # tier-3 + flaky lean slightly more to COD
    cod_bias = 0.18 * (cust_tier[c] == "Tier-3") + 0.12 * (1 - reliability[c])
    pay = "COD" if rng.random() < (0.55 + cod_bias) else "Prepaid"
    device = rng.choice(DEVICES, p=DEV_PROB)
    # address quality correlated with tier (tier-3 worse on average)
    base_addr = {"Tier-1": [0.72, 0.22, 0.06],
                 "Tier-2": [0.55, 0.32, 0.13],
                 "Tier-3": [0.38, 0.38, 0.24]}[cust_tier[c]]
    addr = rng.choice(ADDR, p=base_addr)
    hour = int(rng.integers(0, 24))
    items = int(rng.choice([1, 1, 1, 2, 2, 3], ))
    rows.append([oid, c, order_day[oid], cat, value, discount, pay,
                 device, addr, hour, items])

cols = ["order_id", "cust", "day", "category", "value", "discount",
        "payment", "device", "address", "hour", "items"]
df = pd.DataFrame(rows, columns=cols)

# ---- decide RTO chronologically per customer so HISTORY can be a signal ----
df = df.sort_values(["cust", "day", "order_id"]).reset_index(drop=True)

prior_orders = np.zeros(len(df), dtype=int)
prior_returns = np.zeros(len(df), dtype=int)
rto = np.zeros(len(df), dtype=int)

# running tally per customer
seen = {}            # cust -> [orders_so_far, returns_so_far]
for i in range(len(df)):
    c = int(df.at[i, "cust"])
    po, pr = seen.get(c, (0, 0))
    prior_orders[i] = po
    prior_returns[i] = pr
    prior_rate = (pr / po) if po > 0 else 0.0

    # ---- build log-odds of THIS order being returned ----
    z = -1.95  # intercept (tuned so COD RTO ~ low 30s%, prepaid ~ mid-single %)
    z += 1.30 if df.at[i, "payment"] == "COD" else -1.45
    z += (0.5 - reliability[c]) * 3.6                 # hidden trait
    z += (prior_rate - 0.25) * 1.8 if po > 0 else 0.0  # observed history
    z += 0.45 if po == 0 else 0.0                      # first-timer
    z += TIER_RISK[cust_tier[c]]
    z += state_effect[cust_state[c]]
    z += CAT_RISK[df.at[i, "category"]]
    z += (df.at[i, "discount"] - 20) / 100 * 2.2       # deep discount -> impulse
    # high COD bill -> cold feet at the door
    z += ((df.at[i, "value"] - 1500) / 3000) * 0.35 * (df.at[i, "payment"] == "COD")
    z += ADDR_RISK[df.at[i, "address"]]
    z += DEV_RISK[df.at[i, "device"]]
    z += 0.30 if df.at[i, "hour"] in (0, 1, 2, 3, 4) else 0.0
    z += rng.normal(0, 0.55)                           # irreducible noise

    p = 1 / (1 + np.exp(-z))
    r = int(rng.random() < p)
    rto[i] = r
    seen[c] = (po + 1, pr + r)

df["prior_orders"] = prior_orders
df["prior_returns"] = prior_returns
df["rto"] = rto

# attach customer-level fields
df["city_tier"] = [cust_tier[c] for c in df["cust"]]
df["state"] = [cust_state[c] for c in df["cust"]]
df["_reliability_hidden"] = [reliability[c] for c in df["cust"]]  # not exported

# ---------------------------------------------------------------- assemble tidy frame
out = pd.DataFrame({
    "OrderID": "ORD" + (100000 + df["order_id"]).astype(str),
    "CustomerID": "CUST" + (10000 + df["cust"]).astype(str),
    "OrderDay": df["day"],
    "CityTier": df["city_tier"],
    "State": df["state"],
    "Category": df["category"],
    "OrderValue": df["value"].round(0).astype(int),
    "DiscountPct": df["discount"].round(0).astype(int),
    "PaymentMethod": df["payment"],
    "Device": df["device"],
    "AddressQuality": df["address"],
    "OrderHour": df["hour"],
    "Items": df["items"],
    "PriorOrders": df["prior_orders"],
    "PriorReturns": df["prior_returns"],
    "DeliveryStatus": np.where(df["rto"] == 1, "Returned", "Delivered"),
})

# ============================================================= INJECT MESSINESS
# (so the cleaning chapter has real, honest work to do -- mirrors a real export)
out = out.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle

# 1) OrderValue as messy strings: commas, "Rs"/"INR"/"Rs." prefixes sometimes
def messy_value(v):
    s = f"{v:,}"
    r = rng.random()
    if r < 0.18:   return "Rs " + s
    if r < 0.30:   return "INR " + s
    if r < 0.40:   return "Rs." + s
    return s
out["OrderValue"] = out["OrderValue"].map(messy_value)

# 2) DiscountPct sometimes "20%" strings, sometimes plain
out["DiscountPct"] = out["DiscountPct"].map(
    lambda v: (f"{v}%" if rng.random() < 0.45 else str(v)))

# 3) CityTier case / format variants
tier_variants = {"Tier-1": ["Tier-1", "Tier 1", "tier1", "T1", "TIER-1"],
                 "Tier-2": ["Tier-2", "Tier 2", "tier2", "T2", "TIER-2"],
                 "Tier-3": ["Tier-3", "Tier 3", "tier3", "T3", "TIER-3"]}
out["CityTier"] = out["CityTier"].map(lambda t: rng.choice(tier_variants[t]))

# 4) State case variants for a slice of rows
def messy_state(s):
    r = rng.random()
    if r < 0.10: return s.upper()
    if r < 0.18: return s.lower()
    if r < 0.22: return "  " + s + " "   # stray whitespace
    return s
out["State"] = out["State"].map(messy_state)

# 5) Missing values: AddressQuality (~7%) and Device (~4%)
amask = rng.random(len(out)) < 0.07
out.loc[amask, "AddressQuality"] = np.nan
dmask = rng.random(len(out)) < 0.04
out.loc[dmask, "Device"] = np.nan

# 6) A handful of duplicate rows (accidental double-export)
dups = out.sample(35, random_state=11)
out = pd.concat([out, dups], ignore_index=True)

# 7) A few impossible discounts (>100) to be caught and capped
bad = out.sample(12, random_state=13).index
out.loc[bad, "DiscountPct"] = rng.choice(["150%", "120", "999%"], size=12)

out = out.sample(frac=1, random_state=21).reset_index(drop=True)  # final shuffle
out.to_csv("cod_orders.csv", index=False)

# ============================================================= VALIDATION REPORT
print("rows written :", len(out))
print("unique custs :", out['CustomerID'].nunique())
print("columns      :", list(out.columns))
print("duplicates   :", out.duplicated().sum())

# quick clean copy just to validate the embedded signal is realistic
chk = out.drop_duplicates().copy()
chk["val"] = (chk["OrderValue"].str.replace(r"[^0-9]", "", regex=True)
              .replace("", np.nan).astype(float))
chk["disc"] = (chk["DiscountPct"].astype(str).str.replace("%", "", regex=False)
               .astype(float).clip(0, 100))
chk["tier"] = (chk["CityTier"].str.upper().str.replace(r"[^0-9]", "", regex=True)
               .map({"1": "Tier-1", "2": "Tier-2", "3": "Tier-3"}))
chk["y"] = (chk["DeliveryStatus"] == "Returned").astype(int)

print("\n--- overall RTO  : {:.1%}".format(chk['y'].mean()))
print("--- by payment :"); print(chk.groupby('PaymentMethod')['y'].mean().round(3))
print("--- COD only RTO : {:.1%}".format(chk[chk.PaymentMethod=='COD']['y'].mean()))
print("--- by tier    :"); print(chk.groupby('tier')['y'].mean().round(3))
print("--- by category:"); print(chk.groupby('Category')['y'].mean().round(3).sort_values())
cod = chk[chk.PaymentMethod == "COD"]
print("--- first-timer vs repeat (COD):")
print(cod.assign(first=cod.PriorOrders.eq(0)).groupby('first')['y'].mean().round(3))

# does history predict? mean RTO by prior-return-rate bucket (COD, has history)
h = cod[cod.PriorOrders > 0].copy()
h["prate"] = (h.PriorReturns / h.PriorOrders)
h["pb"] = pd.cut(h["prate"], [-0.01, 0.0, 0.25, 0.5, 1.01],
                 labels=["0%", "1-25%", "26-50%", ">50%"])
print("--- RTO by PRIOR return-rate bucket (the 'history' signal):")
print(h.groupby('pb', observed=True)['y'].mean().round(3))

# ---- learnability check: is the signal realistic (not too easy)? ----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
codm = cod.copy()
codm["prate"] = np.where(codm.PriorOrders > 0, codm.PriorReturns / codm.PriorOrders, 0.0)
codm["first"] = codm.PriorOrders.eq(0).astype(int)
num = ["val", "disc", "prate", "first", "PriorOrders"]
cat = ["tier", "Category", "AddressQuality", "Device"]
codm = codm.dropna(subset=["val"])
Xc = codm[num + cat].fillna("NA")
yc = codm["y"]
pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat)],
                        remainder="passthrough")
pipe = Pipeline([("p", pre), ("m", LogisticRegression(max_iter=1000))])
auc = cross_val_score(pipe, Xc, yc, cv=4, scoring="roc_auc")
print("--- COD model 4-fold ROC-AUC: {:.3f} (+/- {:.3f})".format(auc.mean(), auc.std()))
