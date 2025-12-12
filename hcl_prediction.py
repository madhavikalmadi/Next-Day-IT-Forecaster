company_name = "HCL Technologies Ltd"
ticker = "HCLTECH.NS"
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Use a substantial historical period for training the model
# 5 years is a decent starting point for basic trend identification
end_date_download = datetime.now()
start_date_download = end_date_download - timedelta(days=5 * 365) # Approximately 5 years ago

print(f"--- Predicting Next Day Close Price for {company_name} ({ticker}) ---")
print(f"Historical data range: {start_date_download.strftime('%Y-%m-%d')} to {end_date_download.strftime('%Y-%m-%d')}")

# --- Step 1: Download Historical Data ---
try:
    df = yf.download(ticker, start=start_date_download, end=end_date_download, progress=False)
    
    # Ensure index is datetime and drop any rows with missing values
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"No historical data found for {company_name} ({ticker}). Please check the ticker symbol or date range.")
    if 'Close' not in df.columns:
        raise ValueError(f"'Close' price column not found in data for {company_name}. Available columns: {df.columns.tolist()}")

except Exception as e:
    print(f"Error downloading data: {e}")
    print("Exiting script.")
    exit()

print(f"\nSuccessfully downloaded {len(df)} historical data points.")
print("Latest historical data point:")
print(df.tail(1))


# --- Step 2: Prepare Data for Linear Regression ---
# data (X)
# Close (y).
df['Days'] = (df.index - df.index[0]).days

X = df[['Days']] # Independent variable (feature) 
y = df['Close']  # Dependent variable (target)


# --- Step 3: Train the Linear Regression Model ---
model = LinearRegression() 
model.fit(X, y) #using linear regression the model learns the relationship between 'Days' and 'Close' prices
print("\nLinear Regression model trained.")


# --- Step 4: Predict the Next Day's Closing Price ---
next_day_value_for_prediction = df['Days'].iloc[-1] + 1
predicted_next_close_price = model.predict([[next_day_value_for_prediction]])[0].item()


# --- Step 5: Display Results ---
latest_close_price = df['Close'].iloc[-1].item() #retirve actual closing price
print(f"\n--- Prediction Results for {company_name} ---")
print(f"Last Closing Price: ₹{latest_close_price:.2f}")
print(f"Predicted Closing Price: ₹{predicted_next_close_price:.2f}")


# --- Step 6: Visualize Historical Data and Prediction ---
plt.figure(figsize=(12, 7))
plt.plot(df.index, df['Close'], label='Historical Close Price', color='blue')
plt.plot(df.index, model.predict(X), color='red', linestyle='--', label='Linear Regression Trend')
predicted_date = df.index[-1] + timedelta(days=1)
plt.scatter(predicted_date, predicted_next_close_price, color='green', marker='o', s=100, zorder=5, label='Predicted Next Close Price')
plt.annotate(f'₹{predicted_next_close_price:.2f}', (predicted_date, predicted_next_close_price),
             textcoords="offset points", xytext=(0,10), ha='center', color='green', weight='bold')

plt.title(f'{company_name} Historical Close Price with Next Day Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price (INR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()