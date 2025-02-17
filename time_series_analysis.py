# 📥 Step 1: Import Libraries with Error Handling
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
except ModuleNotFoundError as e:
    print(f"Module error: {e}. Please run: pip install pandas matplotlib statsmodels")
    exit()

# 🟠 Step 2: Load Data with Safety Checks
try:
    df = pd.read_csv('stock_prices.csv', parse_dates=['Date'], index_col='Date')
except FileNotFoundError:
    print("Error: stock_prices.csv not found. Please ensure the file is in the correct folder.")
    exit()

# 🟢 Step 3: Clean Data (Drop Missing Values)
df.dropna(inplace=True)

# 📊 Step 4: Explore Data
print("Basic Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# 📈 Step 5: Plot Historical Prices
plt.figure(figsize=(10, 5))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 📊 Step 6: Analyze Trends (Seasonal Decomposition)
result = seasonal_decompose(df['Close'], model='multiplicative', period=30)
result.plot()
plt.show()

# 🧪 Step 7: Test for Stationarity (ADF Test)
adf_test = adfuller(df['Close'])
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')
if adf_test[1] > 0.05:
    print("The data is not stationary. Proceeding with differencing.")
else:
    print("The data is stationary. No differencing needed.")

# 🔄 Step 8: Differencing (If Needed)
df['Close_diff'] = df['Close'].diff().dropna()

plt.figure(figsize=(10, 5))
plt.plot(df['Close_diff'], label='Differenced Prices', color='green')
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Price Difference')
plt.legend()
plt.grid(True)
plt.show()

# ⚙️ Step 9: Build ARIMA Model
model = ARIMA(df['Close_diff'], order=(5, 1, 0))
model_fit = model.fit()

# 📊 Step 10: Display ARIMA Model Summary
print(model_fit.summary())

# 🔮 Step 11: Forecast Next 30 Days
forecast = model_fit.forecast(steps=30)

# 🗓️ Create Proper Dates for Forecast
forecast_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')

# 📉 Step 12: Plot Forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label='Historical Prices', color='blue')
plt.plot(forecast_dates, forecast, label='30-Day Forecast', color='red', linestyle='--')
plt.title('Stock Price Forecast (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 📝 Final Notes:
# - The script includes safety checks for missing files and modules.
# - It handles missing values before modeling.
# - Forecasts are correctly plotted with dates.
