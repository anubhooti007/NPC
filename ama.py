# parlay_risk_dashboard.py
#
# Streamlit “one-file” dashboard: shows 6 distinct rejection reasons
# ─ Kelly-cap, No-edge, Expected-loss-limit, Stop-Loss, CVaR-limit, Exposure-limit.

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
# 2) Sidebar – risk knobs + new Exposure, Expected Loss, Stop-Loss
# ────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Risk Parameter Controls")

BANKROLL              = st.sidebar.number_input("Platform Bankroll", 0, None, 100_000, 1_000, format="%d")
KELLY_FRACTION        = st.sidebar.slider("Kelly Fraction", 0.0, 1.0, 1.0, 0.05)
MAX_PAYOUT            = st.sidebar.number_input("Max Payout (stake×odds)", 0, None, 2_000, 100)
CVAR_LIMIT            = st.sidebar.number_input("CVaR Limit", 0, None, 3000, 50)          # 300 → lets row 9 trip CVaR
VAR_ALPHA             = st.sidebar.slider("VaR Confidence α", 0.90, 0.99, 0.95, 0.01)
N_TRIALS              = st.sidebar.number_input("Simulation Trials", 1_000, None, 20_000, 1_000)

# New: maximum exposure per individual leg/outcome
default_exposure      = int(BANKROLL * 0.10)
EXPOSURE_LIMIT        = st.sidebar.number_input(
    "Max Exposure per Outcome",
    0,
    None,
    default_exposure,
    step=100,
    format="%d"
)

# New: per‐parlay expected‐loss limit (p × liability = p × stake × (odds - 1))
default_exp_loss      = int(BANKROLL * 0.01)
EXPECTED_LOSS_LIMIT   = st.sidebar.number_input(
    "Parlay Expected Loss Limit",
    0,
    None,
    default_exp_loss,
    step=100,
    format="%d"
)

# New: cumulative stop‐loss threshold (total expected loss across accepted parlays)
default_stop_loss     = int(BANKROLL * 0.05)
STOP_LOSS_LIMIT       = st.sidebar.number_input(
    "Cumulative Stop-Loss Limit",
    0,
    None,
    default_stop_loss,
    step=100,
    format="%d"
)

np.random.seed(0)

# ────────────────────────────────────────────────────────────
# 3) Helpers
# ────────────────────────────────────────────────────────────
def odds_tier_cap(o: float) -> float:
    """
    Tiered cap on stake (in $) based on parlay odds.
    """
    if   o <=   5: return BANKROLL
    elif o <=  20: return BANKROLL * 0.20
    elif o <= 100: return BANKROLL * 0.05
    else:          return BANKROLL * 0.01

def simulate_losses(stakes, p_odds):
    """
    Given a list of stakes and corresponding (p, odds),
    run N_TRIALS Monte Carlo draws to compute profit/loss.
    Returns an array of total losses across all accepted parlays.
    """
    losses = np.zeros((len(stakes), N_TRIALS))
    for i, stake in enumerate(stakes):
        p, o = p_odds[i]
        wins  = np.random.rand(N_TRIALS) < p
        # If wins → payout = stake * (o - 1); else 0.
        losses[i] = np.where(wins, stake * (o - 1), 0)
    return losses.sum(axis=0)

def var_cvar(total_losses):
    """
    Compute VaR and CVaR at the VAR_ALPHA confidence level.
    """
    VaR = np.percentile(total_losses, VAR_ALPHA * 100)
    tail = total_losses[total_losses >= VaR]
    CVaR = tail.mean() if tail.size else VaR
    return VaR, CVaR

# ────────────────────────────────────────────────────────────
# 4) Core evaluation (with Dynamic Exposure, Expected Loss, Stop-Loss)
# ────────────────────────────────────────────────────────────
def evaluate_parlays(parlays, exposure_limit, exp_loss_limit, stop_loss_limit):
    """
    Evaluate each parlay in 'parlays' against:
      1) No‐edge check (Kelly < tiny)
      2) Kelly sizing (fractional Kelly)
      3) Tier cap (odds-based)
      4) Max payout cap
      5) Per-parlay Expected Loss check (p × liability)
      6) Cumulative Stop-Loss (total expected-loss)
      7) Dynamic exposure per-leg
      8) Portfolio CVaR check

    Returns a DataFrame with:
      - Parlay #, Description, p, odds, requested, allowed,
        Decision, Reason,
        Max Users on Leg, Exposure (Max Leg)
    """
    accepted_stakes         = []
    accepted_po             = []
    exposure_map            = {}   # leg_key → cumulative potential payout
    count_map               = {}   # leg_key → number of accepted parlays containing this leg
    cum_expected_loss       = 0.0   # cumulative expected loss across accepted parlays
    stop_loss_active        = False

    rows = []
    for i, pl in enumerate(parlays, 1):
        # 1) Compute combined win‐prob & combined odds
        p    = np.prod([leg[0] for leg in pl["legs"]])
        odds = np.prod([leg[1] for leg in pl["legs"]])

        # 2) Kelly sizing
        edge = p * (odds - 1) - (1 - p)                  # user edge per $1
        kf   = max(edge / (odds - 1), 0)                 # fraction of Kelly
        stake_kelly = BANKROLL * kf * KELLY_FRACTION

        # 3) Tier cap & 4) Max payout cap
        stake_tier = odds_tier_cap(odds)
        stake_max  = MAX_PAYOUT / odds
        allowed    = min(pl["requested"], stake_kelly, stake_tier, stake_max)

        # For reporting: pre‐existing leg stats
        leg_keys           = [tuple(leg) for leg in pl["legs"]]
        existing_counts    = [count_map.get(lk, 0) for lk in leg_keys]
        existing_exposure  = [exposure_map.get(lk, 0) for lk in leg_keys]
        max_count_before   = max(existing_counts) if existing_counts else 0
        max_expo_before    = max(existing_exposure) if existing_exposure else 0

        # DECISION LOGIC:
        if allowed < 1e-6:
            decision, reason = "Rejected", "No positive edge"
        else:
            # 5) Check if Stop-Loss already active:
            if stop_loss_active:
                decision, reason = "Rejected", "Stop-Loss Active"
            else:
                # 6) Per‐parlay Expected Loss check
                # platform_expected_loss = max(edge * allowed, 0)
                platform_expected_loss = max(edge * allowed, 0)
                if platform_expected_loss > exp_loss_limit:
                    decision, reason = "Rejected", "Expected-Loss Limit Exceeded"
                else:
                    # 7) Cumulative Stop-Loss check (if adding this parlay)
                    if cum_expected_loss + platform_expected_loss > stop_loss_limit:
                        decision = "Rejected"
                        reason   = "Stop-Loss Triggered"
                        stop_loss_active = True
                    else:
                        # 8) Dynamic Exposure Check
                        #   - For each leg, potential_payout = allowed * (odds - 1)
                        potential_payout = allowed * (odds - 1)
                        exposure_violation_leg = None
                        for lk in leg_keys:
                            if exposure_map.get(lk, 0) + potential_payout > exposure_limit:
                                exposure_violation_leg = lk
                                break

                        if exposure_violation_leg is not None:
                            # Identify the leg description for messaging:
                            leg_desc = f"p={exposure_violation_leg[0]:.3f}, o={exposure_violation_leg[1]:.2f}"
                            decision = "Rejected"
                            reason   = f"Exposure limit exceeded for [{leg_desc}]"
                        else:
                            # 9) Tentatively accept → compute portfolio CVaR including this parlay
                            losses = simulate_losses(
                                accepted_stakes + [allowed],
                                accepted_po     + [(p, odds)]
                            )
                            _, cvar_val = var_cvar(losses)

                            if cvar_val > CVAR_LIMIT:
                                decision, reason = "Rejected", "CVaR Exceeded Portfolio Limit"
                            elif np.isclose(allowed, stake_kelly, atol=1e-6) and allowed < pl["requested"]:
                                decision, reason = "Rejected", "Stake Capped by Kelly"
                            else:
                                # ACCEPT this parlay
                                decision, reason = "Accepted", ""
                                accepted_stakes.append(allowed)
                                accepted_po.append((p, odds))

                                # Update cumulative expected loss
                                cum_expected_loss += platform_expected_loss

                                # Update exposure_map & count_map for each leg
                                for lk in leg_keys:
                                    exposure_map[lk] = exposure_map.get(lk, 0) + potential_payout
                                    count_map[lk]    = count_map.get(lk, 0) + 1

        rows.append(
            dict(
                **{
                    "Parlay #": i,
                    "Description": pl["display"],
                    "Win Prob (p)": round(p, 3),
                    "Decimal Odds": round(odds, 2),
                    "Requested": pl["requested"],
                    "Allowed": round(allowed, 2),
                    "Decision": decision,
                    "Reason": reason,
                    "Max Users on Leg": max_count_before,
                    "Exposure (Max Leg)": round(max_expo_before, 2),
                }
            )
        )

    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────
# 5) Demo sample parlays (7 pass, 6 fail for distinct reasons)
# ────────────────────────────────────────────────────────────
sample_parlays = [
    # Accepted examples
    {"display":"NFL Double 1.62×1.45",       "legs":[(0.62,1.62),(0.69,1.45)], "requested":120},
    {"display":"NBA Triple 1.80/1.50/1.70",  "legs":[(0.56,1.80),(0.67,1.50),(0.59,1.70)], "requested":80},
    {"display":"EPL Double 1.75×1.83",       "legs":[(0.57,1.75),(0.55,1.83)], "requested":60},
    {"display":"Low-Var 1.50×1.50",          "legs":[(0.70,1.50),(0.70,1.50)], "requested":50},
    {"display":"UFC 1.55 & O2.5@1.80",       "legs":[(0.65,1.55),(0.56,1.80)], "requested":25},
    {"display":"Soccer BTTS/Over 1.55×1.55", "legs":[(0.65,1.55),(0.65,1.55)], "requested":40},
    {"display":"Tennis 1.45×1.35",           "legs":[(0.70,1.45),(0.75,1.35)], "requested":90},

    # Rejected – Kelly-cap
    {"display":"MLB Treble 1.60/1.70/1.80",  "legs":[(0.63,1.60),(0.59,1.70),(0.56,1.80)], "requested":300},

    # Rejected – No edge
    {"display":"Futures Double 4.0×4.0",     "legs":[(0.25,4.0),(0.25,4.0)],               "requested":100},

    # Rejected – Expected-Loss exceed
    {"display":"High-Expensive Parlay 2.00×2.00",
     "legs":[(0.90,2.00),(0.90,2.00)],
     "requested":5_000},  # pushes per‐parlay expected loss beyond limit

    # Rejected – Stop-Loss cumulative (if wcześniejsze parlays push cum loss)
    {"display":"Additional Risky Parlay 1.80×1.80",
     "legs":[(0.85,1.80),(0.85,1.80)],
     "requested":3_000},

    # Rejected – CVaR exceed
    {"display":"Big-Favourite High-Stake 1.40×1.40",
     "legs":[(0.72,1.40),(0.72,1.40)],
     "requested":1_500},

    # 🔄 NEW ⇒ Rejected – Exposure-limit exceed
    {"display":"NFL Double Whale 1.62×1.45",
     "legs":[(0.62,1.62),(0.69,1.45)],
     "requested":5_000},
]

# ────────────────────────────────────────────────────────────
# 6) Run, display, allow CSV download
# ────────────────────────────────────────────────────────────
df = evaluate_parlays(sample_parlays, EXPOSURE_LIMIT, EXPECTED_LOSS_LIMIT, STOP_LOSS_LIMIT)

st.subheader("Parlay Evaluations")
st.markdown("Accepted → **green**, Rejected → **red**")

styled = (
    df.style
      .applymap(
          lambda v: (
              "background:#d4edda;color:#155724;font-weight:600"
              if v == "Accepted" else
              ("background:#f8d7da;color:#721c24;font-weight:600" if v == "Rejected" else "")
          ),
          subset=["Decision"]
      )
)

st.dataframe(styled, use_container_width=True, height=550)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, "parlay_results.csv", "text/csv")


