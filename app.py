import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Big Data Dashboard", layout="wide")

# ---------- DARK MODE TOGGLE ----------
mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
font_color = "#111827" if not mode else "white"

# ---------- UI ----------
if not mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
    }

    section[data-testid="stSidebar"] {
        background: #eef2ff;
    }

    .card {
        padding: 25px;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
        transition: 0.3s;
    }

    .card:hover {
        transform: translateY(-5px);
    }

    h1 {
        color: #111827;
    }

    h2, h3, p {
        color: #374151;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #0f172a, #020617);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #1e293b, #334155);
    }

    .card {
        padding: 25px;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
        transition: 0.3s;
    }

    .card:hover {
        transform: translateY(-5px);
    }

    h1, h2, h3, p {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore_20000.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    big_data = []

    for i in range(200):
        temp = df.copy()
        temp["Order Date"] += pd.Timedelta(days=i*3)

        trend = 1 + (i / 200) * 0.5
        season = 1 + 0.1 * np.sin(i / 10)

        temp["Sales"] *= trend * season * np.random.uniform(0.9, 1.1, len(temp))
        temp["Profit"] *= trend * np.random.uniform(0.85, 1.15, len(temp))

        big_data.append(temp)

    return pd.concat(big_data, ignore_index=True)

with st.spinner("Loading dashboard..."):
    df = load_data()

# ---------- SIDEBAR ----------
st.sidebar.header("🔎 Filters")

region = st.sidebar.multiselect("Region", df["Region"].unique(), default=df["Region"].unique())
category = st.sidebar.multiselect("Category", df["Category"].unique(), default=df["Category"].unique())

date_range = st.sidebar.date_input(
    "Date Range",
    [df["Order Date"].min(), df["Order Date"].max()]
)

# ---------- FILTER ----------
filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Order Date"] >= pd.to_datetime(date_range[0])) &
    (df["Order Date"] <= pd.to_datetime(date_range[1]))
].copy()

# ---------- TITLE ----------
st.markdown("<h1>📊 InsightFlow Dashboard</h1>", unsafe_allow_html=True)
st.write(f"📦 Dataset Size: **{len(df):,} records**")

# ---------- KPI ----------
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
orders = filtered_df.shape[0]

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="card" style="background: linear-gradient(135deg, #2563eb, #1d4ed8);">
<h5 style="color:white;">Total Revenue</h5>
<h2 style="color:white;">₹{total_sales:,.0f}</h2>
<p style="color:white;">📈 Growth trend</p>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="card" style="background: linear-gradient(135deg, #10b981, #059669);">
<h5 style="color:white;">Total Profit</h5>
<h2 style="color:white;">₹{total_profit:,.0f}</h2>
<p style="color:white;">💰 Profitability</p>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="card" style="background: linear-gradient(135deg, #f59e0b, #d97706);">
<h5 style="color:white;">Total Orders</h5>
<h2 style="color:white;">{orders}</h2>
<p style="color:white;">📦 Transactions</p>
</div>
""", unsafe_allow_html=True)

# ---------- REVENUE TREND ----------
st.subheader("📈 Revenue Trend (₹)")

sales_trend = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]
    .sum()
    .reset_index()
)

if len(sales_trend) > 1:
    sales_trend = sales_trend.iloc[:-1]

fig1 = px.area(sales_trend, x="Order Date", y="Sales")

fig1.update_traces(
    line=dict(color="#2563eb", width=4),
    fill='tozeroy',
    fillcolor="rgba(37, 99, 235, 0.25)"
)

fig1.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    font=dict(color=font_color)
)

fig1.update_yaxes(tickprefix="₹")

st.plotly_chart(fig1, use_container_width=True)

# ---------- REGION ----------
st.subheader("🌍 Region Distribution")

fig2 = px.pie(filtered_df, names="Region", values="Sales", hole=0.6)
fig2.update_layout(font=dict(color=font_color))
st.plotly_chart(fig2, use_container_width=True)

# ---------- CATEGORY ----------
st.subheader("📦 Category Sales")

fig3 = px.bar(filtered_df, x="Category", y="Sales", color="Category")
fig3.update_layout(font=dict(color=font_color))
st.plotly_chart(fig3, use_container_width=True)

# ---------- ML ----------
st.subheader("🤖 Sales Prediction")

trend = sales_trend.copy()
trend["Day"] = np.arange(len(trend))

model = LinearRegression()
model.fit(trend[["Day"]], trend["Sales"])

future_days = np.arange(len(trend), len(trend) + 10).reshape(-1, 1)
predictions = model.predict(future_days)

future_dates = pd.date_range(
    start=trend["Order Date"].max() + pd.Timedelta(days=30),
    periods=10,
    freq="M"
)

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": predictions
})

fig4 = px.line(pred_df, x="Date", y="Predicted Sales")
fig4.update_layout(font=dict(color=font_color))
fig4.update_yaxes(tickprefix="₹")

st.plotly_chart(fig4, use_container_width=True)

# ---------- SEPARATOR ----------
st.markdown("---")

# ---------- DOWNLOAD + TABLE ----------
st.subheader("📥 Download & Data Table")

st.download_button(
    "📥 Download Filtered Data",
    filtered_df.to_csv(index=False),
    "filtered_data.csv"
)

rows_to_show = st.slider(
    "Select number of rows to display",
    10,
    min(20000, len(filtered_df)),
    100
)

st.dataframe(filtered_df.head(rows_to_show), use_container_width=True)