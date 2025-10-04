import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path

st.set_page_config(page_title="PhonePe Unified Analytics", layout="wide")

DATA_DIR = Path("/Users/kishore_kumar/PhonePe/pulse/csv_out")
with open(Path(__file__).parent / "states_centroids.json") as f:
    STATE_CENTROIDS = json.load(f)
@st.cache_data
def load_data():
    dfs = {}
    names = [
        "aggregated_transactions","map_transactions","top_transactions_state_combined",
        "top_transaction_state_districts","top_transaction_state_pincodes",
        "aggregated_users","map_users","aggregated_users_by_device",
        "aggregated_insurance","map_insurance",
        "top_insurance_state_districts","top_insurance_state_pincodes","top_insurance_state_combined"
    ]
    for name in names:
        dfs[name] = pd.read_csv(DATA_DIR / f"{name}.csv")
    for k in ["aggregated_transactions","map_transactions","aggregated_users","map_users",
              "aggregated_insurance","map_insurance"]:
        if "Year" in dfs[k].columns: dfs[k]["Year"] = dfs[k]["Year"].astype(int)
        if "Quarter" in dfs[k].columns: dfs[k]["Quarter"] = dfs[k]["Quarter"].astype(int)
    return dfs

dfs = load_data()

def kpi(value, label, delta=None):
    st.metric(label, value, delta=delta)

def get_years_quarters(df):
    years = sorted(df["Year"].unique().tolist())
    quarters = [1,2,3,4]
    return years, quarters

def apply_filters(df, state, year, quarter, col_type=None, val=None):
    d = df.copy()
    if state and state != "All India" and "State" in d.columns: d = d[d["State"]==state]
    if year and year != "All" and "Year" in d.columns: d = d[d["Year"]==int(year)]
    if quarter and quarter != "All" and "Quarter" in d.columns: d = d[d["Quarter"]==int(quarter)]
    if col_type and val and val != "All" and col_type in d.columns: d = d[d[col_type]==val]
    return d

def map_state_scatter(df_state_amt, value_col, title):
    df = df_state_amt.copy()
    df["key"] = df["State"].str.lower()
    df["lat"] = df["key"].map(lambda s: STATE_CENTROIDS.get(s, (22.0,79.0))[0])
    df["lon"] = df["key"].map(lambda s: STATE_CENTROIDS.get(s, (22.0,79.0))[1])
    fig = px.scatter_geo(df, lat="lat", lon="lon", scope="asia",
                         size=value_col, hover_name="State",
                         hover_data={value_col:":.0f","lat":False,"lon":False})
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#444")
    fig.update_layout(title=title, margin=dict(l=0,r=0,t=40,b=0))
    return fig

def yoy_qoq(df, measure_col, year, quarter):
    """
    Compute YoY and QoQ growth for the selected filters.
    - If both year and quarter are specified -> use that (y, q).
    - If quarter == "All" with a specific year -> use the latest q in that year.
    - If year == "All" (with any quarter selection) -> use the latest (y, q) present.
    Returns (yoy, qoq) as floats (e.g., 0.123 for +12.3%) or None if not computable.
    """
    if df.empty or measure_col not in df.columns:
        return None, None

    df = df.copy().sort_values(["Year", "Quarter"])

    # Determine the anchor (y, q) to compute growth for
    if year != "All" and quarter != "All":
        y, q = int(year), int(quarter)
    elif year != "All" and quarter == "All":
        y = int(year)
        # latest quarter available in that year
        if (df["Year"] == y).any():
            q = int(df.loc[df["Year"] == y, "Quarter"].max())
        else:
            return None, None
    else:
        # year == "All": pick the latest (Year, Quarter) pair in df
        y = int(df["Year"].max())
        q = int(df.loc[df["Year"] == y, "Quarter"].max())

    # Current value
    this_val = df.query("Year==@y and Quarter==@q")[measure_col].sum()

    # QoQ: previous quarter (wrap to previous year if q==1)
    pq, py = (q-1, y) if q > 1 else (4, y-1)
    prev_q_val = df.query("Year==@py and Quarter==@pq")[measure_col].sum()
    qoq = (this_val/prev_q_val - 1) if prev_q_val > 0 else None

    # YoY: same quarter in previous year
    prev_y_val = df.query("Year==@y-1 and Quarter==@q")[measure_col].sum()
    yoy = (this_val/prev_y_val - 1) if prev_y_val > 0 else None

    return yoy, qoq
    
    # Case 2: "All Quarters" → Compare latest year vs previous year
    if year != "All" and quarter == "All":
        y = int(year)
        this_val = df[df["Year"]==y][measure_col].sum()
        prev_val = df[df["Year"]==y-1][measure_col].sum()
        yoy = (this_val/prev_val - 1) if prev_val>0 else None
        return yoy, None
    
    # Case 3: "All Years" → Compare last year vs previous year
    if year == "All":
        last_y = df["Year"].max()
        prev_y = last_y - 1
        this_val = df[df["Year"]==last_y][measure_col].sum()
        prev_val = df[df["Year"]==prev_y][measure_col].sum()
        yoy = (this_val/prev_val - 1) if prev_val>0 else None
        return yoy, None
    
    return None, None

# Sidebar filters
st.sidebar.header("Filters")
years_t, _ = get_years_quarters(dfs["aggregated_transactions"])
state_list = ["All India"] + sorted(dfs["aggregated_transactions"]["State"].unique().tolist())
tx_types = ["All"] + sorted(dfs["aggregated_transactions"]["Transaction_type"].unique().tolist())
device_brands = ["All"] + sorted(dfs["aggregated_users_by_device"]["Brand"].unique().tolist()) if "Brand" in dfs["aggregated_users_by_device"].columns else ["All"]

sel_state = st.sidebar.selectbox("State", state_list, index=0)
sel_year   = st.sidebar.selectbox("Year", ["All"] + [str(y) for y in years_t], index=0)
sel_quarter= st.sidebar.selectbox("Quarter", ["All","1","2","3","4"], index=0)
sel_tx_type= st.sidebar.selectbox("Transaction Type", tx_types, index=0)
sel_device = st.sidebar.selectbox("Device Brand", device_brands, index=0)

st.title("📊 PhonePe Unified Analytics (Streamlit)")
st.caption("Covers Transactions, Users, Devices, and Insurance — with India bubble maps (state centroids), charts, and leaderboards.")

tabs = st.tabs([
    "1) Transaction Dynamics",
    "2) Device Dominance & Engagement",
    "3) Insurance Penetration & Growth",
    "4) Market Expansion (Transactions)",
    "5) User Engagement & Growth",
    "6) Transaction Leaders (State/District/Pincode)",
    "7) User Registration Leaders",
    "8) Insurance Transaction Leaders"
])

# Tab 1
with tabs[0]:
    st.subheader("Transaction Dynamics")
    df = apply_filters(dfs["aggregated_transactions"], sel_state, sel_year, sel_quarter, "Transaction_type", sel_tx_type)
    k1 = int(df["Transaction_count"].sum())
    k2 = float(df["Transaction_amount"].sum())
    yoy, qoq = yoy_qoq(apply_filters(dfs["aggregated_transactions"], sel_state, sel_year, sel_quarter, "Transaction_amount", None), "Transaction_amount", sel_year, sel_quarter)
    c1,c2,c3 = st.columns(3)
    with c1: kpi(f"{k1:,}", "Total Transactions")
    with c2: kpi(f"₹{k2:,.0f}", "Total Amount")
    

with c3:
    # Show YoY and QoQ as two stacked KPI metrics in the same column (no nested columns)
    if yoy is None:
        st.metric("YoY", "—")
    else:
        pct = f"{yoy*100:.1f}%"
        st.metric("YoY", pct, delta=pct)

    if qoq is None:
        st.metric("QoQ", "—")
    else:
        pct = f"{qoq*100:.1f}%"
        st.metric("QoQ", pct, delta=pct)


    trend = apply_filters(dfs["aggregated_transactions"], sel_state, "All", "All", "Transaction_type", sel_tx_type)\
            .groupby(["Year","Quarter"], as_index=False).agg(Amount=("Transaction_amount","sum"), Count=("Transaction_count","sum"))
    trend["YearQ"] = trend["Year"].astype(str) + " Q" + trend["Quarter"].astype(str)
    st.plotly_chart(px.line(trend, x="YearQ", y="Amount", markers=True, title="Quarterly Transaction Amount"), use_container_width=True)

    latest = apply_filters(dfs["aggregated_transactions"], "All India", sel_year, sel_quarter, "Transaction_type", sel_tx_type)
    if sel_year == "All" or sel_quarter == "All":
        idx = latest[["Year","Quarter"]].drop_duplicates().sort_values(["Year","Quarter"]).tail(1)
        if len(idx):
            y, q = int(idx.iloc[0]["Year"]), int(idx.iloc[0]["Quarter"])
            latest = latest.query("Year==@y and Quarter==@q")
    state_amt = latest.groupby("State", as_index=False).agg(Amount=("Transaction_amount","sum"))
    st.plotly_chart(map_state_scatter(state_amt, "Amount", "State-wise Transactions (bubble size = amount)"), use_container_width=True)

    heat = apply_filters(dfs["aggregated_transactions"], "All India", "All", "All", "Transaction_type", sel_tx_type)\
            .groupby(["State","Year","Quarter"], as_index=False).agg(Amount=("Transaction_amount","sum"))
    heat["YearQ"] = heat["Year"].astype(str)+" Q"+heat["Quarter"].astype(str)
    pivot = heat.pivot_table(index="State", columns="YearQ", values="Amount", fill_value=0)
    st.plotly_chart(px.imshow(pivot, aspect="auto", title="State × Quarter Heatmap (Amount)"), use_container_width=True)

# Tab 2
with tabs[1]:
    st.subheader("Device Dominance & Engagement")
    agg_users = apply_filters(dfs["aggregated_users"], sel_state, sel_year, sel_quarter)
    ucol = "Registered_users" if "Registered_users" in agg_users.columns else "Registered_Users" if "Registered_Users" in agg_users.columns else None
    ocol = "App_opens" if "App_opens" in agg_users.columns else "App_Opens" if "App_Opens" in agg_users.columns else None
    total_users = int(agg_users[ucol].sum()) if ucol else 0
    total_opens = int(agg_users[ocol].sum()) if ocol else 0
    engagement = (total_opens/total_users) if total_users>0 else 0
    c1,c2,c3 = st.columns(3)
    with c1: kpi(f"{total_users:,}", "Registered Users")
    with c2: kpi(f"{total_opens:,}", "App Opens")
    with c3: kpi(f"{engagement:.2f}", "Engagement (opens/user)")

    dev = dfs["aggregated_users_by_device"].copy()
    if sel_state!="All India" and "State" in dev.columns: dev = dev[dev["State"]==sel_state]
    if sel_year!="All" and "Year" in dev.columns: dev = dev[dev["Year"]==int(sel_year)]
    if sel_quarter!="All" and "Quarter" in dev.columns: dev = dev[dev["Quarter"]==int(sel_quarter)]
    if sel_device!="All" and "Brand" in dev.columns: dev = dev[dev["Brand"]==sel_device]
    if "Count" in dev.columns:
        dev_share = dev.groupby("Brand", as_index=False).agg(Users=("Count","sum"))
        st.plotly_chart(px.bar(dev_share.sort_values("Users",ascending=False).head(10), x="Brand", y="Users", title="Device Market Share (Registered Users)"), use_container_width=True)
        st.dataframe(dev_share.sort_values("Users", ascending=False).head(20))

# Tab 3
with tabs[2]:
    st.subheader("Insurance Penetration & Growth")
    ai = apply_filters(dfs["aggregated_insurance"], sel_state, sel_year, sel_quarter)
    pol_col = "Insurance_count" if "Insurance_count" in ai.columns else "Transaction_count" if "Transaction_count" in ai.columns else None
    amt_col = "Insurance_amount" if "Insurance_amount" in ai.columns else "Transaction_amount" if "Transaction_amount" in ai.columns else None
    total_policies = int(ai[pol_col].sum()) if pol_col else 0
    total_premium = float(ai[amt_col].sum()) if amt_col else 0.0
    avg_premium = (total_premium/total_policies) if total_policies>0 else 0.0
    c1,c2,c3 = st.columns(3)
    with c1: kpi(f"{total_policies:,}", "Total Policies")
    with c2: kpi(f"₹{total_premium:,.0f}", "Total Premium Value")
    with c3: kpi(f"₹{avg_premium:,.0f}", "Avg Premium / Policy")

    tr = apply_filters(dfs["aggregated_insurance"], sel_state, "All", "All")\
            .groupby(["Year","Quarter"], as_index=False).agg(Premium=(amt_col,"sum"))
    tr["YearQ"] = tr["Year"].astype(str)+" Q"+tr["Quarter"].astype(str)
    st.plotly_chart(px.line(tr, x="YearQ", y="Premium", markers=True, title="Insurance Premium Trajectory"), use_container_width=True)

    latest_i = apply_filters(dfs["aggregated_insurance"], "All India", sel_year, sel_quarter)
    if sel_year == "All" or sel_quarter == "All":
        idx = latest_i[["Year","Quarter"]].drop_duplicates().sort_values(["Year","Quarter"]).tail(1)
        if len(idx):
            y, q = int(idx.iloc[0]["Year"]), int(idx.iloc[0]["Quarter"])
            latest_i = latest_i.query("Year==@y and Quarter==@q")
    state_prem = latest_i.groupby("State", as_index=False).agg(Premium=(amt_col,"sum"))
    st.plotly_chart(map_state_scatter(state_prem, "Premium", "State-wise Insurance Premium (bubble map)"), use_container_width=True)

    di = dfs["map_insurance"].copy()
    if sel_state!="All India": di = di[di["State"]==sel_state]
    if sel_year!="All": di = di[di["Year"]==int(sel_year)]
    if sel_quarter!="All": di = di[di["Quarter"]==int(sel_quarter)]
    di_top = di.groupby(["State","Name"], as_index=False).agg(Premium=("Amount","sum"), Policies=("Count","sum")).sort_values("Premium", ascending=False).head(10)
    st.dataframe(di_top)

# Tab 4
with tabs[3]:
    st.subheader("Market Expansion (Transactions)")
    tx = dfs["aggregated_transactions"].copy()
    users = dfs["map_users"].copy()
    latest_tx = tx.sort_values(["Year","Quarter"]).groupby(["State"], as_index=False).tail(1)\
                  .groupby("State", as_index=False).agg(Amount=("Transaction_amount","sum"), Count=("Transaction_count","sum"))
    # Users latest per state
    if "Registered_users" in users.columns:
        users_col = "Registered_users"
    else:
        users_col = "Registered_Users"
    latest_users = users.sort_values(["Year","Quarter"]).groupby(["State"], as_index=False).tail(1)\
                      .groupby("State", as_index=False).agg(Users=(users_col,"sum"))
    # Growth YoY
    growth_df = tx.groupby(["State","Year","Quarter"], as_index=False).agg(Amount=("Transaction_amount","sum"))
    last_y, last_q = int(growth_df["Year"].max()), int(growth_df[growth_df["Year"].eq(growth_df["Year"].max())]["Quarter"].max())
    prev = growth_df[growth_df["Year"].eq(last_y-1) & growth_df["Quarter"].eq(last_q)].rename(columns={"Amount":"Amount_prev"})
    curr = growth_df[growth_df["Year"].eq(last_y) & growth_df["Quarter"].eq(last_q)].rename(columns={"Amount":"Amount_curr"})
    g = pd.merge(curr[["State","Amount_curr"]], prev[["State","Amount_prev"]], on="State", how="left")
    g["GrowthRate"] = (g["Amount_curr"]/g["Amount_prev"] - 1.0).replace([np.inf,-np.inf], np.nan)
    base = pd.merge(g, latest_users, on="State", how="left")
    base["PenetrationProxy"] = base["Amount_curr"] / base["Users"]
    base["MEI"] = (base["GrowthRate"].fillna(0)) * (base["Users"].fillna(0)) / (base["PenetrationProxy"].replace(0,np.nan))
    base = base.replace([np.inf,-np.inf], np.nan)
    st.plotly_chart(px.scatter(base, x="PenetrationProxy", y="GrowthRate", size="Users", hover_name="State",
                               title="Growth vs Penetration (bubble size = Users)"), use_container_width=True)
    st.dataframe(base.sort_values("MEI", ascending=False).head(15)[["State","MEI","GrowthRate","Users","PenetrationProxy"]])
    st.plotly_chart(map_state_scatter(base.fillna(0).rename(columns={"MEI":"Value"})[["State","Value"]], "Value", "Market Expansion Index (MEI) by State"), use_container_width=True)

# Tab 5
with tabs[4]:
    st.subheader("User Engagement & Growth")
    mu = apply_filters(dfs["map_users"], sel_state, sel_year, sel_quarter)
    if "Registered_users" in mu.columns and "App_opens" in mu.columns:
        by_state = mu.groupby("State", as_index=False).agg(Users=("Registered_users","sum"), Opens=("App_opens","sum"))
    else:
        by_state = mu.groupby("State", as_index=False).agg(Users=("Registered_Users","sum"), Opens=("App_Opens","sum"))
    by_state["Engagement"] = by_state["Opens"] / by_state["Users"]
    st.plotly_chart(px.scatter(by_state, x="Users", y="Engagement", size="Opens", hover_name="State",
                               title="Users vs Engagement (opens/user)"), use_container_width=True)
    st.plotly_chart(map_state_scatter(by_state.rename(columns={"Engagement":"Value"})[["State","Value"]], "Value", "Engagement by State (opens per user)"), use_container_width=True)
    st.dataframe(by_state.sort_values("Engagement", ascending=False).head(15))

# Tab 6
with tabs[5]:
    st.subheader("Transaction Leaders (State / District / Pincode)")
    tx_latest = apply_filters(dfs["aggregated_transactions"], "All India", sel_year, sel_quarter, "Transaction_type", sel_tx_type)
    top_states_amt = tx_latest.groupby("State", as_index=False).agg(Amount=("Transaction_amount","sum"), Count=("Transaction_count","sum")).sort_values("Amount", ascending=False).head(10)
    st.plotly_chart(px.bar(top_states_amt, x="State", y="Amount", title="Top 10 States by Amount", text_auto=True), use_container_width=True)
    dmap = apply_filters(dfs["map_transactions"], "All India", sel_year, sel_quarter)
    top_districts = dmap.groupby(["State","Name"], as_index=False).agg(Amount=("Amount","sum"), Count=("Count","sum")).sort_values("Amount", ascending=False).head(10)
    st.dataframe(top_districts)
    pmap = apply_filters(dfs["top_transaction_state_pincodes"], "All India", sel_year, sel_quarter)
    pin_col = "Pincode" if "Pincode" in pmap.columns else ("Name" if "Name" in pmap.columns else None)
    if pin_col:
        top_pins = pmap.groupby(["State", pin_col], as_index=False).agg(Amount=("Amount","sum"), Count=("Count","sum")).sort_values("Amount", ascending=False).head(10)
        st.dataframe(top_pins)

# Tab 7
with tabs[6]:
    st.subheader("User Registration Leaders (select Year/Quarter in sidebar)")
    umap = apply_filters(dfs["map_users"], "All India", sel_year, sel_quarter)
    reg_col = "Registered_users" if "Registered_users" in umap.columns else "Registered_Users"
    top_states_reg = umap.groupby("State", as_index=False).agg(Registered=(reg_col,"sum")).sort_values("Registered", ascending=False).head(10)
    st.plotly_chart(px.bar(top_states_reg, x="State", y="Registered", title="Top States by Registrations"), use_container_width=True)
    top_dist_reg = umap.groupby(["State","Name"], as_index=False).agg(Registered=(reg_col,"sum")).sort_values("Registered", ascending=False).head(10)
    st.dataframe(top_dist_reg)

# Tab 8
with tabs[7]:
    st.subheader("Insurance Transaction Leaders (select Year/Quarter)")
    imap = apply_filters(dfs["map_insurance"], "All India", sel_year, sel_quarter)
    top_states_ins = imap.groupby("State", as_index=False).agg(Premium=("Amount","sum"), Policies=("Count","sum")).sort_values("Premium", ascending=False).head(10)
    st.plotly_chart(px.bar(top_states_ins, x="State", y="Premium", title="Top States by Insurance Premium"), use_container_width=True)
    top_dist_ins = imap.groupby(["State","Name"], as_index=False).agg(Premium=("Amount","sum"), Policies=("Count","sum")).sort_values("Premium", ascending=False).head(10)
    st.dataframe(top_dist_ins)

st.write("---")
st.caption("Built with Streamlit + Plotly | Bubble maps use state centroids (approximate) for offline-friendly visuals.")
