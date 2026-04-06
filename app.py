import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="InsightFlow Dashboard", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
}

section[data-testid="stSidebar"] {
    background: #eef2ff;
}

.card {
    padding: 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
}

h1 {
    color: #111827;
    font-size: 28px;
}

h2, h3, p {
    color: #374151;
}

@media (max-width: 768px) {

    .block-container {
        padding: 1rem !important;
    }

    h1 {
        font-size: 22px !important;
        text-align: left !important;
        word-break: break-word;
    }

    .card {
        padding: 18px;
        margin-bottom: 15px;
    }
}
</style>
""", unsafe_allow_html=True)

font_color = "#111827"

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

df = load_data()

st.sidebar.header("Filters")

region = st.sidebar.multiselect("Region", df["Region"].unique(), default=df["Region"].unique())
category = st.sidebar.multiselect("Category", df["Category"].unique(), default=df["Category"].unique())
date_range = st.sidebar.date_input("Date Range", [df["Order Date"].min(), df["Order Date"].max()])

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Order Date"] >= pd.to_datetime(date_range[0])) &
    (df["Order Date"] <= pd.to_datetime(date_range[1]))
]

st.markdown("<h1>📊 InsightFlow Dashboard</h1>", unsafe_allow_html=True)
st.write(f"📦 Dataset Size: **{len(df):,} records**")

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
orders = filtered_df.shape[0]

col1, col2, col3 = st.columns([1,1,1], gap="medium")

with col1:
    st.markdown(f"""
    <div class="card" style="background: linear-gradient(135deg, #2563eb, #1d4ed8);">
    <h5>Total Revenue</h5>
    <h2>₹{total_sales:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card" style="background: linear-gradient(135deg, #10b981, #059669);">
    <h5>Total Profit</h5>
    <h2>₹{total_profit:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card" style="background: linear-gradient(135deg, #f59e0b, #d97706);">
    <h5>Total Orders</h5>
    <h2>{orders}</h2>
    </div>
    """, unsafe_allow_html=True)

st.subheader("📈 Revenue Trend")

sales_trend = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]
    .sum()
    .reset_index()
)

fig1 = px.area(sales_trend, x="Order Date", y="Sales")
fig1.update_layout(font=dict(color=font_color))
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🌍 Region Distribution")

fig2 = px.pie(filtered_df, names="Region", values="Sales", hole=0.6)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📦 Category Sales")

fig3 = px.bar(filtered_df, x="Category", y="Sales", color="Category")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🤖 Sales Prediction")

trend = sales_trend.copy()
trend["Day"] = np.arange(len(trend))

model = LinearRegression()
model.fit(trend[["Day"]], trend["Sales"])

future_days = np.arange(len(trend), len(trend)+10).reshape(-1,1)
predictions = model.predict(future_days)

future_dates = pd.date_range(
    start=trend["Order Date"].max(),
    periods=10,
    freq="MS"
)

pred_df = pd.DataFrame({"Date": future_dates, "Predicted Sales": predictions})

fig4 = px.line(pred_df, x="Date", y="Predicted Sales")
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.subheader("📥 Data")

st.download_button("Download Data", filtered_df.to_csv(index=False), "data.csv")
st.dataframe(filtered_df.head(100), use_container_width=True)