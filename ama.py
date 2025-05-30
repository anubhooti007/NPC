# parlay_risk_dashboard.py
#
# Streamlit “one-file” dashboard: shows 3 distinct rejection reasons
# ─ Kelly-cap, No-edge, CVaR-limit ─ at default sidebar settings.

import streamlit as st
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────
# 1) Page config & CSS tweaks
# ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Parlay Risk Dashboard", layout="wide")
st.title("📊 Parlay Risk Evaluation Dashboard")

st.markdown(
    """
    <style>
      div.stDataFrame table {table-layout: fixed !important; width: 100% !important;}
      div.stDataFrame th, div.stDataFrame td {white-space: normal !important; word-wrap: break-word;}
      div.stDataFrame thead th {background:#f0f0f0!important;color:#333!important;font-weight:600!important;border-bottom:2px solid #ddd!important;}
    </style>
    """,
    unsafe_allow_html=True,
)

pd.set_option("display.max_colwidth", None)

# ────────────────────────────────────────────────────────────
# 2) Sidebar – risk knobs
# ────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Risk Parameter Controls")
BANKROLL       = st.sidebar.number_input("Platform Bankroll", 0, None, 100_000, 1_000, format="%d")
KELLY_FRACTION = st.sidebar.slider("Kelly Fraction", 0.0, 1.0, 1.0, 0.05)
MAX_PAYOUT     = st.sidebar.number_input("Max Payout (stake×odds)", 0, None, 2_000, 100)
CVAR_LIMIT     = st.sidebar.number_input("CVaR Limit", 0, None, 300, 50)          # 300 -> lets row 9 trip CVaR
VAR_ALPHA      = st.sidebar.slider("VaR Confidence α", 0.90, 0.99, 0.95, 0.01)
N_TRIALS       = st.sidebar.number_input("Simulation Trials", 1_000, None, 20_000, 1_000)

np.random.seed(0)

# ────────────────────────────────────────────────────────────
# 3) Helpers
# ────────────────────────────────────────────────────────────
def odds_tier_cap(o: float) -> float:
    if   o <=   5: return BANKROLL
    elif o <=  20: return BANKROLL * 0.20
    elif o <= 100: return BANKROLL * 0.05
    else:          return BANKROLL * 0.01

def simulate_losses(stakes, p_odds):
    losses = np.zeros((len(stakes), N_TRIALS))
    for i, stake in enumerate(stakes):
        p, o = p_odds[i]
        wins  = np.random.rand(N_TRIALS) < p
        losses[i] = np.where(wins, stake * (o - 1), 0)
    return losses.sum(axis=0)

def var_cvar(total_losses):
    VaR = np.percentile(total_losses, VAR_ALPHA * 100)
    tail = total_losses[total_losses >= VaR]
    CVaR = tail.mean() if tail.size else VaR
    return VaR, CVaR

# ────────────────────────────────────────────────────────────
# 4) Core evaluation
# ────────────────────────────────────────────────────────────
def evaluate_parlays(parlays):
    accepted_s, accepted_po, rows = [], [], []

    for i, pl in enumerate(parlays, 1):
        p    = np.prod([leg[0] for leg in pl["legs"]])
        odds = np.prod([leg[1] for leg in pl["legs"]])

        # Kelly sizing
        edge = p * (odds - 1) - (1 - p)
        kf   = max(edge / (odds - 1), 0)
        stake_kelly = BANKROLL * kf * KELLY_FRACTION

        # Caps
        stake_tier = odds_tier_cap(odds)
        stake_max  = MAX_PAYOUT / odds
        allowed    = min(pl["requested"], stake_kelly, stake_tier, stake_max)

        # Decision tree
        if allowed < 1e-6:
            decision, reason = "Rejected", "No positive edge"
        else:
            # provisional accept → test portfolio CVaR
            losses = simulate_losses(accepted_s + [allowed], accepted_po + [(p, odds)])
            _, cvar_val = var_cvar(losses)

            if cvar_val > CVAR_LIMIT:
                decision, reason = "Rejected", "CVaR exceeded portfolio limit"
            elif np.isclose(allowed, stake_kelly, atol=1e-6) and allowed < pl["requested"]:
                decision, reason = "Rejected", "Stake capped by Kelly"
            else:
                decision, reason = "Accepted", ""
                accepted_s.append(allowed)
                accepted_po.append((p, odds))

        rows.append(
            dict(
                **{"Parlay #": i,
                   "Description": pl["display"],
                   "Win Prob (p)": round(p, 3),
                   "Decimal Odds": round(odds, 2),
                   "Requested": pl["requested"],
                   "Allowed": round(allowed, 2),
                   "Decision": decision,
                   "Reason": reason}
            )
        )

    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────
# 5) Demo sample parlays (7 pass, 3 fail for 3 distinct reasons)
# ────────────────────────────────────────────────────────────
sample_parlays = [
    # Accepted examples
    {"display":"NFL Double 1.62×1.45",  "legs":[(0.62,1.62),(0.69,1.45)],             "requested":120},
    {"display":"NBA Triple 1.80/1.50/1.70", "legs":[(0.56,1.80),(0.67,1.50),(0.59,1.70)], "requested":80},
    {"display":"EPL Double 1.75×1.83",  "legs":[(0.57,1.75),(0.55,1.83)],              "requested":60},
    {"display":"Low-Variance 1.50×1.50", "legs":[(0.70,1.50),(0.70,1.50)],              "requested":50},
    {"display":"UFC 1.55 & O2.5@1.80",  "legs":[(0.65,1.55),(0.56,1.80)],              "requested":25},
    {"display":"Soccer BTTS/Over 1.55×1.55", "legs":[(0.65,1.55),(0.65,1.55)],          "requested":40},
    {"display":"Tennis 1.45×1.35",      "legs":[(0.70,1.45),(0.75,1.35)],              "requested":90},

    # Rejected – Kelly cap
    {"display":"MLB Treble 1.60/1.70/1.80", "legs":[(0.63,1.60),(0.59,1.70),(0.56,1.80)], "requested":300},

    # Rejected – No edge
    {"display":"Futures Double 4.0×4.0",  "legs":[(0.25,4.0),(0.25,4.0)],               "requested":100},

    # Rejected – CVaR exceed
    {"display":"Big-Favourite High-Stake 1.40×1.40",
     "legs":[(0.72,1.40),(0.72,1.40)],                                                   "requested":1_500},
]

# ────────────────────────────────────────────────────────────
# 6) Run, display, allow CSV download
# ────────────────────────────────────────────────────────────
df = evaluate_parlays(sample_parlays)

st.subheader("Parlay Evaluations")
st.markdown("Accepted → **green**, Rejected → **red**")

styled = (
    df.style
      .applymap(
          lambda v: ("background:#d4edda;color:#155724;font-weight:600"
                     if v=="Accepted" else
                     ("background:#f8d7da;color:#721c24;font-weight:600"
                      if v=="Rejected" else "")),
          subset=["Decision"]
      )
)

st.dataframe(styled, use_container_width=True, height=430)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "parlay_results.csv", "text/csv")
