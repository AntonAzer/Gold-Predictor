import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="AI Gold Price Predictor", page_icon="🪙", layout="centered")

st.title("🪙 AI-Powered Gold Price Predictor")
st.markdown("""
This application uses **Meta's Prophet AI model** to forecast global gold prices. 
It automatically fetches real-time market data, updates itself daily, and calculates prices for different gold karats.
""")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ App Settings")
# Optional: Let users adjust local currency exchange rate (default USD to EGP)
exchange_rate = st.sidebar.number_input("USD to Local Currency Exchange Rate (e.g., EGP)", min_value=1.0, value=1.0, step=0.1)
currency_symbol = st.sidebar.text_input("Currency Symbol", value="USD")

# --- Step 1: Load Live Data Automatically ---
@st.cache_data(ttl=86400) # Cache data for 24 hours to keep the app fast
def load_live_data():
    # Fetch historical gold spot price (GC=F) up to today
    gold_data = yf.download("GC=F", start="2016-01-01", end=datetime.date.today().strftime("%Y-%m-%d"))
    df = gold_data[['Close']].copy()
    df.columns = ['y'] # Prophet expects target column to be 'y'
    df['ds'] = df.index.tz_localize(None) # Prophet expects datetime column to be 'ds'
    return df

with st.spinner("Fetching live market data and updating the AI model..."):
    df_clean = load_live_data()

# --- Step 2: Train Meta Prophet Model ---
@st.cache_resource
def train_prophet_model(df):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    return model

model = train_prophet_model(df_clean)

# --- Step 3: User Date Input for Prediction ---
st.subheader("📅 Predict Future Gold Price")
min_date = datetime.date.today()
max_date = min_date + datetime.timedelta(days=365) # Limit prediction to 1 year ahead for accuracy

target_date = st.date_input(
    "Select a date to view predicted prices:",
    value=min_date + datetime.timedelta(days=2), # Default to 2 days ahead (e.g., July 19)
    min_value=min_date,
    max_value=max_date
)

# --- Step 4: Make the Prediction ---
# Generate future dataframe up to target date
days_to_predict = (target_date - min_date).days

if days_to_predict >= 0:
    # We create a dataframe containing all historical dates + future dates up to target_date
    future = model.make_future_dataframe(periods=days_to_predict + 5) # Pad a few days
    forecast = model.predict(future)
    
    # Filter the exact prediction for the selected target_date
    prediction_row = forecast[forecast['ds'].dt.date == target_date]
    
    if not prediction_row.empty:
        predicted_ounce_usd = prediction_row['yhat'].values[0]
        
        # Ounce to Gram 24K Conversion
        gram_24k_usd = predicted_ounce_usd / 31.1034768
        
        # Karat Calculations in local currency
        price_24k = gram_24k_usd * exchange_rate
        price_22k = (gram_24k_usd * (22 / 24)) * exchange_rate
        price_21k = (gram_24k_usd * (21 / 24)) * exchange_rate
        price_18k = (gram_24k_usd * (18 / 24)) * exchange_rate

        # --- Step 5: Display Results ---
        st.success(f"### Predicted Prices for: **{target_date.strftime('%B %d, %Y')}**")
        
        # Display as cards/metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🏆 Gold 24K (per gram)", value=f"{price_24k:.2f} {currency_symbol}")
            st.metric(label="✨ Gold 21K (per gram)", value=f"{price_21k:.2f} {currency_symbol}")
        with col2:
            st.metric(label="🌟 Gold 22K (per gram)", value=f"{price_22k:.2f} {currency_symbol}")
            st.metric(label="💍 Gold 18K (per gram)", value=f"{price_18k:.2f} {currency_symbol}")
            
        st.info(f"💡 Based on a predicted Global Spot Price of **${predicted_ounce_usd:.2f} USD** per Troy Ounce.")
    else:
        st.error("Could not generate a prediction for this specific date. Please try another date.")

# --- Disclaimer ---
st.markdown("---")
st.caption("""
⚠️ **Financial Disclaimer:** This application is for educational and informational purposes only. 
Gold markets are highly volatile and influenced by geopolitical events. 
Do not use these predictions as absolute financial or investment advice.
""")