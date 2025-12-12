import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting (essential for server-side)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime, timedelta
import os # Import os for directory creation and path manipulation

# Define company names and their respective Yahoo Finance tickers
companies = {
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Wipro Ltd": "WIPRO.NS",
    'HCL': 'HCLTECH.NS',
    'L&T': 'LTTS.NS'  # L&T Technology Services
}

# Define the historical date range for data download
end_date_data_download = datetime.now()
start_date_data_download = end_date_data_download - timedelta(days=5 * 365) # Approx 5 years prior

results = []

# Function to evaluate stock performance metrics (CAGR, Volatility, Max Drawdown)
def evaluate_stock(data):
    data = data.copy()
    data.index = pd.to_datetime(data.index)

    price_series_name = None
    if 'Close' in data.columns:
        price_series_name = 'Close'
    elif 'Adj Close' in data.columns: 
        price_series_name = 'Adj Close'
    else:
        print("Warning: Neither 'Close' nor 'Adj Close' column found in data for evaluation.")
        return np.nan, np.nan, np.nan

    price_data_series = data[price_series_name].squeeze()

    if not isinstance(price_data_series, pd.Series):
        print(f"CRITICAL ERROR: After selection and squeeze, price data is not a Series. It's a {type(price_data_series)}. Cannot proceed.")
        return np.nan, np.nan, np.nan

    price_data_series = pd.to_numeric(price_data_series, errors='coerce').dropna()

    if len(price_data_series) < 2: 
        return np.nan, np.nan, np.nan

    data['Return'] = price_data_series.pct_change()

    if data['Return'].isnull().all() or data['Return'].empty:
        return np.nan, np.nan, np.nan

    volatility = data['Return'].std() * np.sqrt(252) #volatility 

    #CAGR calculation
    initial_price = price_data_series.iloc[0]  
    final_price = price_data_series.iloc[-1]    

    years = (price_data_series.index[-1] - price_data_series.index[0]).days / 365.25

    if years <= 0 or initial_price <= 0:
        cagr = np.nan
    else:
        cagr = (final_price / initial_price) ** (1 / years) - 1

    # Max Drawdown calculation
    cum_returns = (1 + data['Return']).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    return float(cagr * 100), float(volatility), float(abs(max_drawdown))

print(f"--- Starting Overall Stock Analysis (Data range: {start_date_data_download.strftime('%Y-%m-%d')} to {end_date_data_download.strftime('%Y-%m-%d')}) ---")

# Main loop to download data and evaluate stocks for each company
for name, ticker in companies.items():
    print(f"\nDownloading data for {name} ({ticker})...")
    df = yf.download(ticker, start=start_date_data_download, end=end_date_data_download, progress=False, auto_adjust=True)
    df.dropna(inplace=True) # Drop rows with any NaN values after download

    if not df.empty:
        cagr, vol, mdd = evaluate_stock(df)

        if not pd.isna(cagr) and not pd.isna(vol) and not pd.isna(mdd):
            is_safe = '✅ Yes' if cagr > 10 and vol < 0.35 and mdd < 0.3 else '❌ No'

            results.append({
                'Company': name,
                'Ticker': ticker, # <--- CRITICAL: Include the Ticker here for app.py
                'CAGR (%)': round(cagr, 2),
                'Volatility': round(vol, 3),
                'Max Drawdown': round(mdd, 3),
                'Safe to Invest?': is_safe
            })
            print(f"    Metrics calculated: CAGR={cagr:.2f}%, Vol={vol:.3f}, MDD={mdd:.3f}. Safe: {is_safe}")
        else:
            print(f"    Skipping {name} due to insufficient or invalid data for metric calculation after download.")
    else:
        print(f"    No data downloaded for {name} ({ticker}). Skipping.")

# Convert the list of results to a DataFrame
df_result = pd.DataFrame(results)

if df_result.empty:
    print("\n--- No Stock Analysis Results Generated ---")
    print("No data could be processed for any company. Cannot train model or save results.")
else:
    print("\n--- Stock Analysis Results Before LR Prediction ---")
    print(df_result[['Company', 'Ticker', 'CAGR (%)', 'Volatility', 'Max Drawdown', 'Safe to Invest?']].to_string(index=False))

    # --- Logistic Regression Model for 'Safe to Invest?' Prediction ---
    log_reg_model = None 
    df_result['Is_Safe_Numerical'] = df_result['Safe to Invest?'].apply(lambda x: 1 if x == '✅ Yes' else 0)

    # Define features (X) and target (y) for the model
    X = df_result[['CAGR (%)', 'Volatility', 'Max Drawdown']]
    y = df_result['Is_Safe_Numerical']

    # Check if there's enough data to train the model
    if len(X) > 1 and len(y.unique()) > 1:
        print("\nTraining Logistic Regression Model...")
        log_reg_model = LogisticRegression(random_state=42, solver='liblinear') 
        log_reg_model.fit(X, y)

        # Make predictions and get probabilities
        y_pred_lr = log_reg_model.predict(X)
        y_proba_lr = log_reg_model.predict_proba(X)[:, 1] # Probability of being '1' (Safe)

        print("\n--- Logistic Regression Model Performance ---")
        print(f"Accuracy: {accuracy_score(y, y_pred_lr):.2f}")
        print("Classification Report:\n", classification_report(y, y_pred_lr, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred_lr))

        # Add model predictions and confidence to the results DataFrame
        df_result['LR_Prediction'] = y_pred_lr.tolist()
        df_result['LR_Prediction_Label'] = df_result['LR_Prediction'].apply(lambda x: '✅ Yes' if x == 1 else '❌ No')
        df_result['LR_Confidence'] = [round(p, 2) for p in y_proba_lr] # Store confidence rounded

        print("\n--- Full Results with Logistic Regression Predictions (Final DataFrame) ---")
        print(df_result[['Company', 'Ticker', 'CAGR (%)', 'Volatility', 'Max Drawdown', 'Safe to Invest?', 'LR_Prediction_Label', 'LR_Confidence']].to_string(index=False))

        # Save the trained Logistic Regression model to a .pkl file
        model_filename = 'logistic_regression_model.pkl'
        joblib.dump(log_reg_model, model_filename)
        print(f"\nLogistic Regression model saved to {model_filename}")
    else:
        print("\nSkipping Logistic Regression model training and prediction: Not enough data points or only one class present.")
        # Ensure these columns exist even if model wasn't trained, to prevent KeyError in Flask app
        df_result['LR_Prediction'] = np.nan
        df_result['LR_Prediction_Label'] = "N/A"
        df_result['LR_Confidence'] = np.nan

    # Save the df_result DataFrame to a .pkl file (THIS MUST HAPPEN AFTER ALL COLUMNS ARE ADDED)
    df_result_filename = 'stock_analysis_results.pkl'
    joblib.dump(df_result, df_result_filename)
    print(f"Stock analysis results saved to {df_result_filename}")


# --- Overall Plotting Part (will only show if df_result is not empty) ---
if not df_result.empty:
    # Create 'overall_plots' directory if it doesn't exist
    overall_plots_dir = 'overall_plots'
    if not os.path.exists(overall_plots_dir):
        os.makedirs(overall_plots_dir)

    # --- Plotting settings for overall plots (even more aggressive) ---
    plot_title_fontsize = 60
    plot_label_fontsize = 50
    plot_tick_fontsize = 40
    plot_figsize = (30, 20) # Very large figure size
    plot_dpi = 300 # High DPI

    # Plot 1: CAGR (%)
    fig_cagr, ax_cagr = plt.subplots(figsize=plot_figsize, dpi=plot_dpi)
    ax_cagr.bar(df_result['Company'], df_result['CAGR (%)'], color='green')
    ax_cagr.set_title('CAGR (%) - All Companies', fontsize=plot_title_fontsize, fontweight='bold', pad=40)
    ax_cagr.set_ylabel('Growth %', fontsize=plot_label_fontsize, fontweight='bold')
    ax_cagr.set_xlabel('Company', fontsize=plot_label_fontsize, fontweight='bold')
    ax_cagr.tick_params(axis='x', rotation=45, ha='right', labelsize=plot_tick_fontsize)
    ax_cagr.tick_params(axis='y', labelsize=plot_tick_fontsize)
    ax_cagr.set_ylim(bottom=0)
    ax_cagr.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2.0)
    cagr_plot_path = os.path.join(overall_plots_dir, 'cagr_overall.png')
    plt.savefig(cagr_plot_path, bbox_inches='tight', pad_inches=0.7, dpi=plot_dpi)
    plt.close(fig_cagr)
    print(f"Saved CAGR plot to {cagr_plot_path}")

    # Plot 2: Volatility
    fig_vol, ax_vol = plt.subplots(figsize=plot_figsize, dpi=plot_dpi)
    ax_vol.bar(df_result['Company'], df_result['Volatility'], color='orange')
    ax_vol.set_title('Volatility - All Companies', fontsize=plot_title_fontsize, fontweight='bold', pad=40)
    ax_vol.set_ylabel('Std Dev', fontsize=plot_label_fontsize, fontweight='bold')
    ax_vol.set_xlabel('Company', fontsize=plot_label_fontsize, fontweight='bold')
    ax_vol.tick_params(axis='x', rotation=45, ha='right', labelsize=plot_tick_fontsize)
    ax_vol.tick_params(axis='y', labelsize=plot_tick_fontsize)
    ax_vol.set_ylim(bottom=0)
    ax_vol.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2.0)
    vol_plot_path = os.path.join(overall_plots_dir, 'volatility_overall.png')
    plt.savefig(vol_plot_path, bbox_inches='tight', pad_inches=0.7, dpi=plot_dpi)
    plt.close(fig_vol)
    print(f"Saved Volatility plot to {vol_plot_path}")

    # Plot 3: Max Drawdown
    fig_mdd, ax_mdd = plt.subplots(figsize=plot_figsize, dpi=plot_dpi)
    ax_mdd.bar(df_result['Company'], df_result['Max Drawdown'], color='red')
    ax_mdd.set_title('Max Drawdown - All Companies', fontsize=plot_title_fontsize, fontweight='bold', pad=40)
    ax_mdd.set_ylabel('Drawdown %', fontsize=plot_label_fontsize, fontweight='bold')
    ax_mdd.set_xlabel('Company', fontsize=plot_label_fontsize, fontweight='bold')
    ax_mdd.tick_params(axis='x', rotation=45, ha='right', labelsize=plot_tick_fontsize)
    ax_mdd.tick_params(axis='y', labelsize=plot_tick_fontsize)
    ax_mdd.set_ylim(bottom=0)
    ax_mdd.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2.0)
    mdd_plot_path = os.path.join(overall_plots_dir, 'mdd_overall.png')
    plt.savefig(mdd_plot_path, bbox_inches='tight', pad_inches=0.7, dpi=plot_dpi)
    plt.close(fig_mdd)
    print(f"Saved Max Drawdown plot to {mdd_plot_path}")

else:
    print("\nNo data to plot. df_result is empty.")

print("\n--- Overall Analysis Script Finished ---")