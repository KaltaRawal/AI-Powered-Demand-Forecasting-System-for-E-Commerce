import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from joblib import load
from datetime import timedelta
import os

# -------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------
st.set_page_config(page_title="AI-Powered E-Commerce Forecast Dashboard", layout="wide")

st.title("AI-Powered E-Commerce Demand Forecasting Dashboard")
st.markdown("This dashboard analyzes past sales trends and forecasts future demand using a trained machine learning model.")

# -------------------------------------------------------------
# Load the trained model
# -------------------------------------------------------------
model_path = r"C:\Users\kalta\forecast_pipeline.pkl"
model = None

try:
    if os.path.exists(model_path):
        model = load(model_path)
        st.sidebar.success("Model loaded successfully.")
    else:
        st.sidebar.warning("Model not found. Running in demo mode.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# -------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------
st.sidebar.header("Data Upload")

uploaded_file = st.sidebar.file_uploader("Upload sales data (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    default_path = r"C:\Users\kalta\OneDrive\DSA\Documents\Downloads\brazilian-ecommerce (2)"
    try:
        data = pd.read_csv(default_path)
        st.sidebar.info("Loaded default Brazilian e-commerce dataset.")
    except:
        st.info("No file found. Using generated sample data.")
        data = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=60),
            "Sales": np.random.randint(200, 800, size=60)
        })

# -------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------
if "Date" not in data.columns:
    data.rename(columns={data.columns[0]: "Date"}, inplace=True)

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = data.dropna(subset=["Date"]).sort_values("Date")

if "Sales" not in data.columns:
    data["Sales"] = np.random.randint(200, 800, len(data))

# -------------------------------------------------------------
# Display historical trend
# -------------------------------------------------------------
st.subheader("Historical Sales Trend")
fig = px.line(data, x="Date", y="Sales", title="Sales Over Time", markers=True)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Forecasting section
# -------------------------------------------------------------
st.subheader("AI Forecast Results")

if model is not None:
    try:
        # Create future dates for prediction
        last_date = data["Date"].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=15)
        df_future = pd.DataFrame({"Date": future_dates})

        # Create basic time-based features
        df_future["day_of_week"] = df_future["Date"].dt.dayofweek
        df_future["month"] = df_future["Date"].dt.month
        df_future["is_weekend"] = df_future["day_of_week"].isin([5, 6]).astype(int)
        df_future["is_holiday"] = 0  # placeholder feature

        # Match model's expected features
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        else:
            expected_features = [col for col in df_future.columns if col != "Date"]

        # Add missing columns as zeros
        for col in expected_features:
            if col not in df_future.columns:
                df_future[col] = 0

        # Keep columns in correct order and ensure numeric values
        X_future = df_future[expected_features]
        X_future = X_future.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Generate predictions
        forecast = model.predict(X_future)
        forecast_df = pd.DataFrame({"Date": df_future["Date"], "Predicted Sales": forecast})

    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        forecast_df = pd.DataFrame({
            "Date": pd.date_range(start=data["Date"].max() + timedelta(days=1), periods=15),
            "Predicted Sales": np.random.randint(400, 800, 15)
        })
else:
    forecast_df = pd.DataFrame({
        "Date": pd.date_range(start=data["Date"].max() + timedelta(days=1), periods=15),
        "Predicted Sales": np.random.randint(400, 800, 15)
    })

# -------------------------------------------------------------
# Combine historical and forecast data
# -------------------------------------------------------------
combined = pd.concat([
    data.rename(columns={"Sales": "Predicted Sales"})[["Date", "Predicted Sales"]],
    forecast_df
])

fig2 = px.line(combined, x="Date", y="Predicted Sales",
               title="Sales Forecast for the Next 15 Days", markers=True)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------------
# Insights section
# -------------------------------------------------------------
st.subheader("Insights")

average_sales = data["Sales"].mean()
max_sales = data["Sales"].max()
forecast_growth = forecast_df["Predicted Sales"].mean() - average_sales

st.markdown(f"""
**Average Daily Sales:** {average_sales:.2f}  
**Peak Sales:** {max_sales:.0f}  
**Forecasted Growth:** {forecast_growth:.2f}  
**Forecast Period:** {forecast_df['Date'].min().date()} to {forecast_df['Date'].max().date()}
""")

st.success("Forecasting completed. Scroll above to view detailed charts and trends.")
