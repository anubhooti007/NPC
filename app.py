# Streamlit Dashboard for Competi – Parlay Risk Management (Single-File)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Competi Parlay Risk Dashboard", layout="wide")

# -- Monte Carlo Simulation for P&L --
@st.cache_data
def run_monte_carlo(p, odds, stake, n_trials=100000):
    outcomes = np.random.rand(n_trials) < p
    pnl = np.where(outcomes, (odds - 1) * stake, -stake)
    return pnl

# -- Drawdown & Ruin Simulation --
def simulate_bankroll_runs(p, odds, stake, capital, n_seq=50, n_runs=10000):
    ruin_count = 0
    max_drawdowns = []
    for _ in range(n_runs):
        bank, peak = capital, capital
        seq_drawdowns = []
        for _ in range(n_seq):
            outcome = (odds - 1) * stake if np.random.rand() < p else -stake
            bank += outcome
            peak = max(peak, bank)
            seq_drawdowns.append((peak - bank) / peak)
            if bank <= 0:
                ruin_count += 1
                break
        if seq_drawdowns:
            max_drawdowns.append(max(seq_drawdowns))
    prob_ruin = ruin_count / n_runs
    max_dd = np.max(max_drawdowns) if max_drawdowns else 0
    return prob_ruin, max_dd

# -- Sidebar Navigation --
page = st.sidebar.radio("Navigate to", [
    "CVaR & Buffer Sizing",
    "Drawdown & Ruin",
    "Tranche Allocation",
    "Limits & Controls",
    "Hedging Strategies",
    "Summary"
])

# -- Common Inputs --
stake = st.sidebar.number_input("Stake per Parlay ($)", min_value=1.0, value=1.0)
p1 = st.sidebar.slider("Leg1 Win%", 0, 100, 60) / 100
p2 = st.sidebar.slider("Leg2 Win%", 0, 100, 55) / 100
p3 = st.sidebar.slider("Leg3 Win%", 0, 100, 50) / 100
o1 = st.sidebar.slider("Leg1 Odds", 1.01, 5.0, 1.8)
o2 = st.sidebar.slider("Leg2 Odds", 1.01, 5.0, 2.0)
o3 = st.sidebar.slider("Leg3 Odds", 1.01, 5.0, 2.5)
capital = st.sidebar.number_input("LP Capital ($)", min_value=1_000.0, value=100_000.0)
gamma = st.sidebar.slider("Exposure Cap γ (%)", 0.1, 10.0, 1.0) / 100

# Derived parlay parameters
odds_parlay = o1 * o2 * o3 * 0.97  # net of 3% fee
p_parlay = p1 * p2 * p3
liability = (odds_parlay - 1) * stake

# Page logic
if page == "CVaR & Buffer Sizing":
    st.header("CVaR Method & Insurance Buffer Sizing")
    st.markdown("We simulate **100,000** parlays to estimate the 99% VaR and CVaR.")
    pnl = run_monte_carlo(p_parlay, odds_parlay, stake)
    var99 = np.percentile(pnl, 1)
    cvar99 = pnl[pnl <= var99].mean()
    var_abs = var99 * stake
    cvar_abs = cvar99 * stake
    col1, col2, col3 = st.columns(3)
    col1.metric("99% VaR (per $ stake)", f"${-var_abs:,.2f}")
    col2.metric("99% CVaR (per $ stake)", f"${-cvar_abs:,.2f}")
    col3.metric("Buffer Required (per $ stake)", f"${-cvar_abs:,.2f}")
    fig = px.histogram(pnl, x=pnl, nbins=100, title="P&L Distribution (100k trials)")
    fig.add_vline(x=var99, line_color="red", annotation_text="VaR 99%", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Bet Acceptance Gate:** Ensure buffer ≤ capital × γ = ${capital*gamma:.2f}.")
    if -cvar_abs > capital * gamma:
        st.error("⚠️ CVaR buffer exceeds cap. Throttle new parlays.")
    else:
        st.success("✅ CVaR buffer is within cap.")
elif page == "Drawdown & Ruin":
    st.header("Drawdown Rule & Risk-of-Ruin")
    st.markdown("Simulate sequences to estimate max drawdown and ruin probability.")
    n_seq = st.number_input("Sequence length (# parlays)", min_value=10, value=50)
    n_runs = st.number_input("Simulation runs", min_value=1000, value=10000)
    prob_ruin, max_dd = simulate_bankroll_runs(p_parlay, odds_parlay, stake, capital, n_seq, n_runs)
    col1, col2 = st.columns(2)
    col1.metric("Risk of Ruin", f"{prob_ruin:.2%}")
    col2.metric("Max Drawdown", f"{max_dd:.2%}")
    if prob_ruin > 0.001:
        st.warning("⚠️ High ruin risk! Tighten parameters.")
    if max_dd > 0.25:
        st.warning("⚠️ Drawdown > 25%. Halve Kelly, lower γ.")
elif page == "Tranche Allocation":
    st.header("Tranche-Based Capital Allocation")
    senior_pct = st.slider("Senior Tranche % of LP", 0, 100, 70) / 100
    senior_cap = senior_pct * capital
    junior_cap = capital - senior_cap
    st.metric("Senior Tranche", f"${senior_cap:,.2f}")
    st.metric("Junior Tranche", f"${junior_cap:,.2f}")
    payout_demand = st.number_input("Payout Demand ($)", min_value=0.0, value=250000.0)
    loss_junior = min(payout_demand, junior_cap)
    loss_senior = max(0, payout_demand - junior_cap)
    junior_end = junior_cap - loss_junior
    senior_end = senior_cap - loss_senior
    st.write(f"After ${payout_demand:,.2f} payout: Junior=${junior_end:,.2f}, Senior=${senior_end:,.2f}")
elif page == "Limits & Controls":
    st.header("Limits & Controls")
    st.subheader("Maximum Payout Limits")
    max_payout_cap = st.number_input("Payout Cap ($)", value=1_000_000.0)
    st.write(f"Payouts capped at ${max_payout_cap:,.2f} regardless of odds.")
    st.subheader("Bet Size Restrictions Based on Odds")
    df_tiers = pd.DataFrame(
        [(100,10000),(500,5000),(np.inf,1000)],
        columns=["Max Odds","Max Stake"]
    )
    st.table(df_tiers)
    st.subheader("Dynamic Exposure Limits")
    exp_thresh = st.number_input("Exposure Threshold per Outcome ($)", value=2_000_000.0)
    st.write(f"Stops new bets if exposure > ${exp_thresh:,.2f} on any outcome.")
    st.subheader("Stop-Loss Mechanisms")
    loss_thresh = st.number_input("Hourly Loss Threshold ($)", value=500_000.0)
    st.write(f"Halts bets for 1h if losses > ${loss_thresh:,.2f} in an hour.")
    st.subheader("Expected Loss (p × L)")
    exp_loss = p_parlay * liability
    cap_exp = gamma * capital
    st.write(f"p×L = ${exp_loss:,.2f}, cap = ${cap_exp:,.2f}")
    if exp_loss > cap_exp:
        st.error("⚠️ p×L exceeds cap; hedge excess.")
    else:
        st.success("✅ p×L within cap.")
elif page == "Hedging Strategies":
    st.header("Hedging Strategies")
    st.markdown("Explore parlay hedges via single-leg and multi-leg approaches.")
    # Delta-Neutral Replication
    st.subheader("1. Delta-Neutral Replication")
    col1, col2 = st.columns(2)
    with col1:
        delta = st.slider("Leg 1 Probability Shift Δ", -0.2, 0.2, 0.0, key="hedge_delta")
        p_other = p2 * p3
        hedge_adjust = p_other * liability * delta
        st.metric("Required Hedge Adjustment", f"${hedge_adjust:,.2f}")
        st.caption(f"Liability: ${liability:,.2f}  |  LP Capital: ${capital:,.2f}")
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hedge_adjust,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Hedge ($)"},
            gauge={'axis': {'range': [min(0, hedge_adjust*1.5), max(0, hedge_adjust*1.5)]}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig)
    st.markdown("---")
    # Martingale Hedging
    st.subheader("2. Martingale Hedging")
    st.markdown("""
Place incremental hedges after each successful leg using a simplified martingale approach:
1. Start with a base hedge equal to original stake.
2. After each winning leg, double the hedge on the next leg’s opposite outcome to lock in your original stake.
3. Caps ensure you never exceed a predefined max hedge size.
""")
    base_hedge = st.number_input("Base Hedge Amount ($)", min_value=1.0, value=stake)
    max_hedge = st.number_input("Max Hedge Cap ($)", min_value=base_hedge, value=base_hedge*8)
    legs_won = st.slider("Number of consecutive legs won", 0, 3, 0)
    hedge_martingale = min(base_hedge * (2 ** legs_won), max_hedge)
    st.metric("Martingale Hedge for Next Leg", f"${hedge_martingale:,.2f}")
    st.caption(f"After {legs_won} wins: hedge = min({base_hedge}×2^{legs_won}, cap {max_hedge})")
    st.markdown(f"**Action:** If you’ve won {legs_won} legs, place **${hedge_martingale:,.2f}** opposite the next leg to recover losses and secure your base stake.")
    st.markdown("---")
    # Quantile Hedging (Föllmer–Leukert)
    st.subheader("4. Quantile Hedging (Föllmer–Leukert)")
    st.markdown(
        "Identify the two legs with the highest payout volatility (p·(1-p)), then allocate a fixed hedge budget across them to maximize the chance of covering shortfalls."
    )
    q_budget = 2000.0
    q_target = st.slider("Success Probability (1 - ε)", 0.90, 1.00, 0.995, step=0.005)
    legs = pd.DataFrame({
        "Leg": ["Leg1", "Leg2", "Leg3"],
        "WinProb": [p1, p2, p3],
        "Odds": [o1, o2, o3]
    })
    legs["Volatility"] = legs.WinProb * (1 - legs.WinProb)
    legs = legs.sort_values("Volatility", ascending=False).reset_index(drop=True)
    top = legs.head(2)
    total_vol = top.Volatility.sum()
    top["HedgeAmount"] = (top.Volatility / total_vol) * q_budget
    legs = legs.merge(top[["Leg","HedgeAmount"]], on="Leg", how="left").fillna(0)
    st.table(legs[["Leg","WinProb","Odds","Volatility","HedgeAmount"]]
             .rename(columns={"WinProb":"Win Prob","Odds":"Odds","HedgeAmount":"$ Hedge"}))
    est_success = 1 - np.prod(top.WinProb.values)
    st.metric("Estimated Success P", f"{est_success:.2%}")
    if est_success < q_target:
        st.warning("⚠️ Budget insufficient for target coverage – enlarge budget or relax target.")
    else:
        st.success("✅ Hedge budget meets coverage target.")
elif page == "Summary":
    st.title("Summary & Next Steps")
    st.write("- **CVaR**: worst-tail buffer sizing.")
    st.write("- **Drawdown/Ruin**: sequence risk controls.")
    st.write("- **Tranches**: hierarchical capital protection.")
    st.write("- **Limits**: payout, stake, exposure, stop-loss.")
    st.write("- **Hedging**: delta-neutral, martingale, quantile methodologies integrated.")
    st.write("**Combine these modules** to build a robust parlay risk system.")
