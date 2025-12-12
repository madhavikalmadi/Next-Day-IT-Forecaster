from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression, LinearRegression
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os # Import os for file path manipulation

app = Flask(__name__)

# --- GLOBAL MATPLOTLIB SETTINGS (Even More Aggressive Defaults) ---
# These act as a fallback, but we'll set explicit sizes in functions.
plt.rcParams.update({
    'font.size': 20,           # Increased global base font size
    'axes.titlesize': 28,      # Increased global title font size
    'axes.labelsize': 24,      # Increased global axis label font size
    'xtick.labelsize': 20,     # Increased global X-axis tick label font size
    'ytick.labelsize': 20,     # Increased global Y-axis tick label font size
    'legend.fontsize': 22,     # Increased global legend font size
    'figure.autolayout': False, # Disable autolayout globally; use tight_layout explicitly
    'savefig.dpi': 300,        # Default savefig DPI to high resolution
})


# --- Load Logistic Regression Model and Stock Analysis Data when the app starts ---
lr_model = None
df_results_full = pd.DataFrame()

print("--- Flask App Startup: Loading Models and Data ---")
try:
    lr_model = joblib.load('logistic_regression_model.pkl')
    print("Logistic Regression model loaded successfully by app.py.")

    df_results_full = joblib.load('stock_analysis_results.pkl')
    print("Stock analysis results DataFrame loaded successfully by app.py.")
    print(f"Columns in df_results_full after loading: {df_results_full.columns.tolist()}")
    print(f"First few rows of df_results_full:\n{df_results_full.head().to_string()}")

    # Ensure 'Is_Safe_Numerical' column exists for consistent use, if needed for other parts
    if 'Is_Safe_Numerical' not in df_results_full.columns and 'Safe to Invest?' in df_results_full.columns:
        df_results_full['Is_Safe_Numerical'] = df_results_full['Safe to Invest?'].apply(lambda x: 1 if x == '✅ Yes' else 0)
    elif 'Is_Safe_Numerical' not in df_results_full.columns:
            print("Warning: 'Safe to Invest?' or 'Is_Safe_Numerical' column not found in stock_analysis_results.pkl. Some features (like LR confidence) might not work.")

except FileNotFoundError as e:
    print(f"Error: Required model or data file not found during app startup: {e}. Make sure 'overall_analysis.py' was run successfully.")
    lr_model = None
    df_results_full = pd.DataFrame()
except Exception as e:
    print(f"Error loading files in app.py during startup: {e}")
    lr_model = None
    df_results_full = pd.DataFrame()

print("--- Flask App Startup: Initialization Complete ---")


# --- Function to generate general plots (CAGR, Volatility, MDD) - STRICTLY FOR MULTIPLE COMPANIES ---
def generate_plot(df_data, metric, color, title, ylabel):
    # This function is ONLY for "Overall Analysis" where df_data will have multiple rows/companies.
    # It should NOT be called for single company analysis.
    if df_data.empty or metric not in df_data.columns or len(df_data) <= 1:
        print(f"Warning: generate_plot called with insufficient data for bar chart. Metric: {metric}, Data Length: {len(df_data)}")
        return None

    # --- EXTREME FONT SIZES AND FIGURE SIZE FOR OVERALL PLOTS ---
    # These override any global rcParams for this specific plot type
    title_fontsize = 72 # Maximize title size
    label_fontsize = 60 # Maximize label size
    tick_fontsize = 50  # Maximize tick label size
    company_name_rotation = 45 # Rotation for x-axis company names

    # Maximized figure size and DPI - pushing limits
    fig, ax = plt.subplots(figsize=(40, 25), dpi=400) # EXTREMELY LARGE FIGURE SIZE AND HIGHEST DPI
    
    ax.bar(df_data['Company'], df_data[metric], color=color)
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=50) # Explicit font size and padding
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold') # Explicit font size
    ax.set_xlabel('Company', fontsize=label_fontsize, fontweight='bold') # Explicit font size
    
    # Explicit tick parameters
    ax.tick_params(axis='x', rotation=company_name_rotation, labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    # Ensure x-tick labels are correctly set with rotation
    plt.xticks(rotation=company_name_rotation, ha='right') # Explicitly set rotation for x-ticks
    ax.set_ylim(bottom=0)

    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add grid for better readability

    plt.tight_layout(pad=2.5) # Use tight_layout with MORE padding to prevent labels clipping
    
    img_bytes = io.BytesIO()
    
    # IMPORTANT: Save to PNG with higher resolution for in-memory byte stream
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.7, dpi=400) # Ensure highest DPI for output
    
    # --- DEBUGGING STEP: Save a copy to local file for direct inspection ---
    # Create a 'debug_plots' directory if it doesn't exist
    debug_dir = 'debug_plots'
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    filename_debug = os.path.join(debug_dir, f"{metric.replace(' ', '_').replace('%', '')}_overall_debug.png")
    plt.savefig(filename_debug, format='png', bbox_inches='tight', pad_inches=0.7, dpi=400)
    print(f"DEBUG: Saved overall analysis plot to {filename_debug}") # Print path to confirm

    plt.close(fig) # Explicitly close the figure
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


# --- Function to generate Next Day Close Price Prediction Plot and Text Output ---
def generate_next_day_prediction_plot_and_text(ticker, company_name):
    
    end_date_download = datetime.now() # Use current date
    start_date_download = end_date_download - timedelta(days=5 * 365) # 5 years historical data

    try:
        print(f"--- Predicting Next Day Close Price for {company_name} ({ticker}) ---")
        print(f"Historical data range: {start_date_download.strftime('%Y-%m-%d')} to {end_date_download.strftime('%Y-%m-%d')}")
        
        # Download data
        df = yf.download(ticker, start=start_date_download, end=end_date_download, progress=False, auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)

        if df.empty or 'Close' not in df.columns:
            raise ValueError("No sufficient historical data found for next day prediction. Please check the ticker symbol or date range.")

        # Prepare data for Linear Regression
        df['Days'] = (df.index - df.index[0]).days
        X = df[['Days']]
        y = df['Close']

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict the next day's closing price
        next_day_value_for_prediction = df['Days'].iloc[-1] + 1
        predicted_next_close_price = model.predict([[next_day_value_for_prediction]])[0].item()

        latest_close_price = df['Close'].iloc[-1].item()
        # previous_day_date = df.index[-1].strftime('%Y-%m-%d') # No longer needed for text output
        
        # Determine the actual next trading day, skipping weekends
        predicted_date_for_plot = df.index[-1] + timedelta(days=1)
        while predicted_date_for_plot.weekday() >= 5: # Monday is 0, Sunday is 6
            predicted_date_for_plot += timedelta(days=1)
        
        # predicted_day_date = predicted_date_for_plot.strftime('%Y-%m-%d') # No longer needed for text output

        # --- CONCISE TEXT OUTPUT (MODIFIED TO REMOVE DATES) ---
        return_text = (
            f"Last Close Price : ₹{latest_close_price:.2f}\n" # Removed date
            f"Predicted Close Price : ₹{predicted_next_close_price:.2f}" # Removed date
        )

        # --- GRAPH ENHANCEMENTS FOR PREDICTION PLOT (Similar extreme sizing) ---
        title_fontsize = 72
        label_fontsize = 60
        tick_fontsize = 50
        legend_fontsize = 40 # Increased legend font size
        annotation_fontsize = 45 # Increased annotation font size (for predicted price value)

        fig, ax = plt.subplots(figsize=(40, 22), dpi=400) # EXTREME SIZE, highest DPI
        ax.plot(df.index, df['Close'], label='Historical Close Price', color='blue', linewidth=8) # Thicker line
        ax.plot(df.index, model.predict(X), color='red', linestyle='--', label='Linear Regression Trend', linewidth=8) # Thicker line
        
        # Use the corrected predicted_date_for_plot for the scatter point
        ax.scatter(predicted_date_for_plot, predicted_next_close_price, color='green', marker='o', s=700, zorder=5, label='Predicted Next Close Price', edgecolor='black', linewidth=5) # Larger marker, added edge
        ax.annotate(f'₹{predicted_next_close_price:.2f}', (predicted_date_for_plot, predicted_next_close_price),
                             textcoords="offset points", xytext=(0,50), ha='center', color='green', weight='bold', fontsize=annotation_fontsize,
                             bbox=dict(boxstyle="round,pad=0.7", fc="yellow", ec="k", lw=2, alpha=0.9)) # Add a background box to annotation

        ax.set_title(f'{company_name} Historical Close Price with Next Trading Day Prediction', fontsize=title_fontsize, fontweight='bold', pad=50)
        ax.set_xlabel('Date', fontsize=label_fontsize, fontweight='bold')
        ax.set_ylabel('Close Price (INR)', fontsize=label_fontsize, fontweight='bold')
        
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        
        ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=legend_fontsize) # Use explicit legend size, add frame and shadow
        ax.grid(True, linestyle='--', alpha=0.7) # Enhance grid
        
        plt.tight_layout(pad=2.5) # Use tight_layout with more padding
        
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0.7, dpi=400) # Ensure highest DPI for output
        
        # --- DEBUGGING STEP: Save a copy to local file for direct inspection ---
        debug_dir = 'debug_plots'
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        filename_debug = os.path.join(debug_dir, f"{company_name.replace(' ', '_')}_prediction_debug.png")
        plt.savefig(filename_debug, format='png', bbox_inches='tight', pad_inches=0.7, dpi=400)
        print(f"DEBUG: Saved prediction plot to {filename_debug}") # Print path to confirm

        plt.close(fig) # Explicitly close the figure
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

        return f"data:image/png;base64,{img_base64}", return_text, None

    except Exception as e:
        error_message_for_log = f"Error generating next day prediction for {company_name} ({ticker}): {e}"
        print(error_message_for_log, file=sys.stderr)
        return None, None, error_message_for_log # Return error message to display on UI


@app.route('/')
def home():
    companies_list = ['Overall Analysis']
    if not df_results_full.empty:
        companies_list.extend(df_results_full['Company'].tolist())
    
    prediction_info = {}
    plots = {}

    return render_template('index.html', 
                            companies=companies_list,
                            error_message=None if not df_results_full.empty else "Application data (model/results) not found. Please run 'overall_analysis.py' first to generate them.",
                            selected_option="",
                            prediction_info=prediction_info,
                            plots=plots
                           )


@app.route('/analyze', methods=['POST'])
def analyze():
    selected_option = request.form['company_selection']

    plots = {}
    prediction_info = {}
    error_message = None # Initialize error_message here for all paths

    if df_results_full.empty:
        error_message = "Application data not loaded. Please ensure model and data files exist by running 'overall_analysis.py'."
        companies_list = ['Overall Analysis']
        if not df_results_full.empty:
            companies_list.extend(df_results_full['Company'].tolist())
        return render_template('index.html',
                               companies=companies_list,
                               error_message=error_message,
                               selected_option=selected_option,
                               prediction_info=prediction_info,
                               plots=plots
                               )

    companies_list = ['Overall Analysis'] + df_results_full['Company'].tolist()

    if selected_option == 'Overall Analysis':
        df_display = df_results_full.copy()

        # CRITICAL FIX: Only call generate_plot if there's more than one company
        # as it's designed for overall comparison bar charts.
        if len(df_display) > 1:
            plots['cagr_plot'] = generate_plot(df_display, 'CAGR (%)', 'green', 'CAGR (%) - All Companies', 'Growth %')
            plots['volatility_plot'] = generate_plot(df_display, 'Volatility', 'orange', 'Volatility - All Companies', 'Std Dev')
            plots['mdd_plot'] = generate_plot(df_display, 'Max Drawdown', 'red', 'Max Drawdown - All Companies', 'Drawdown %')
        else:
            error_message = (error_message + "\n" if error_message else "") + "Insufficient data (only one company) for 'Overall Analysis' bar charts. Displaying table only."
            plots = {} # Ensure no plots are passed if data is insufficient

        prediction_info['type'] = 'table'
        # Columns to display for the overall table (without LR columns)
        columns_to_display = ['Company', 'CAGR (%)', 'Volatility', 'Max Drawdown', 'Safe to Invest?']
        
        prediction_info['data_html'] = df_display[columns_to_display].to_html(index=False, classes='table table-striped')
        
        if not lr_model:
            error_message = (error_message + "\n" if error_message else "") + "Logistic Regression model not loaded. Overall predictions (LR_Prediction_Label, LR_Confidence) might be missing for other uses."

    else: # Individual company selected
        df_display = df_results_full[df_results_full['Company'] == selected_option].copy()
        
        selected_ticker = None
        if not df_display.empty and 'Ticker' in df_display.columns and pd.notna(df_display['Ticker'].iloc[0]):
            selected_ticker = df_display['Ticker'].iloc[0]
        else:
            # Fixed the 'error_msg' NameError here by using the 'error_message' variable.
            error_message = (error_message + "\n" if error_message else "") + f"Ticker not found for {selected_option} in loaded data. Next day prediction unavailable."
            print(error_message, file=sys.stderr) 

        if df_display.empty:
            error_message = (error_message + "\n" if error_message else "") + f"No detailed analysis data found for {selected_option}."
        
        plots = {} # Clear plots for individual company view to ensure no bar charts appear
        if selected_ticker:
            next_day_plot_img, prediction_text_output, next_day_error = generate_next_day_prediction_plot_and_text(selected_ticker, selected_option)
            if next_day_plot_img:
                plots['next_day_prediction_plot'] = next_day_plot_img
                prediction_info['next_day_prediction_text'] = prediction_text_output 
            if next_day_error:
                error_message = (error_message + "\n" + next_day_error) if error_message else next_day_error
        else:
            plots['next_day_prediction_plot'] = None
            prediction_info['next_day_prediction_text'] = f"Next day prediction unavailable for {selected_option} (ticker information is missing)."

        prediction_info['type'] = 'single_company' 
        
    return render_template('index.html',
                            companies=companies_list,
                            selected_option=selected_option,
                            plots=plots,
                            prediction_info=prediction_info,
                            error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='127.0.0.1')