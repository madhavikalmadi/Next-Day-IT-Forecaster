company_name = "Tata Consultancy Services Ltd"
ticker = "TCS.NS"
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
# We'll use the number of days since the first data point as our feature (X)
# and the 'Close' price as our target (y).
df['Days'] = (df.index - df.index[0]).days

X = df[['Days']] # Independent variable (feature) - must be 2D
y = df['Close']  # Dependent variable (target)
# --- Step 3: Train the Linear Regression Model ---
model = LinearRegression()
model.fit(X, y)

print("\nLinear Regression model trained.")
# --- Step 4: Predict the Next Day's Closing Price ---
# The 'next day' will be one day after the last day in our historical data
# We add 1 to the last 'Days' value to get the 'Days' value for the next prediction point.
next_day_value_for_prediction = df['Days'].iloc[-1] + 1

# Predict the price. model.predict returns a NumPy array, so we use .item() to get the scalar.
predicted_next_close_price = model.predict([[next_day_value_for_prediction]])[0].item()

# --- Step 5: Display Results (Modified to remove dates) ---
latest_close_price = df['Close'].iloc[-1].item()
# previous_day_date = df.index[-1].strftime('%Y-%m-%d') # No longer used for display
# predicted_day_date = (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d') # No longer used for display

print(f"\n--- Prediction Results for {company_name} ---")
# Changed the print statements to remove the date formatting
print(f"Last Closing Price : ₹{latest_close_price:.2f}")
print(f"Predicted Closing Price : ₹{predicted_next_close_price:.2f}")

# --- Step 6: Visualize Historical Data and Prediction ---
plt.figure(figsize=(12, 7))

# Plot historical close prices
plt.plot(df.index, df['Close'], label='Historical Close Price', color='blue')

# Plot the linear regression line
# For the regression line, we'll plot it over the entire historical range
plt.plot(df.index, model.predict(X), color='red', linestyle='--', label='Linear Regression Trend')

# Plot the predicted point
# Create a date for the predicted point
predicted_date = df.index[-1] + timedelta(days=1) # This is a conceptual date, might not be a trading day
# For plotting, convert the single predicted_date to a list to match the predicted_price format
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