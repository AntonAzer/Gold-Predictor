import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

#Loading the data
gold_ticker = yf.Ticker("GC=F")
df = gold_ticker.history(start="2016-01-01")
df_clean = df[['Close']].copy()
df_clean.columns = ['y']
df_clean['ds'] = df_clean.index.tz_localize(None)

# Training the model
model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
model.fit(df_clean)

#Create one full year excpectations
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)


fig1 = model.plot(forecast)
plt.title("Gold Price Forecast Overview")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.show()

fig2 = model.plot_components(forecast)
plt.show()
