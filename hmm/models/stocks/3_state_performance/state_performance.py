import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
exchange = ccxt.binance()
from datetime import datetime,timezone
import pandas_ta as ta
from tqdm import tqdm


def analyze_state_returns(data, states, state_names=None, min_samples=10):
    """Analyze returns by state, filtering low-sample states."""
    df = data.copy()
    df['State'] = states
    
    # Filter rare states
    state_counts = df['State'].value_counts()
    valid_states = state_counts[state_counts >= min_samples].index
    df = df[df['State'].isin(valid_states)]
    
    # Calculate metrics
    returns_df = df.groupby('State')['Returns'].agg(
        mean_return='mean',
        cum_return=lambda x: (1 + x).prod() - 1,
        win_rate=lambda x: (x > 0).mean(),
        volatility='std'
    ).sort_values('mean_return', ascending=False)
    
    # Add Sharpe ratio
    returns_df['sharpe'] = returns_df['mean_return'] / returns_df['volatility']
    
    # Auto-name states
    if state_names is None:
        rank_names = ["Strong Bull", "Bull", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Bear", "Strong Bear"]
        returns_df['auto_name'] = rank_names[:len(returns_df)]
    else:
        returns_df['auto_name'] = state_names
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    returns_df['mean_return'].plot.bar(ax=ax1, color=np.where(returns_df['mean_return']>0, 'g', 'r'))
    ax1.set_title('Average Returns by State (Filtered)')
    ax1.axhline(0, color='black', linestyle='--')
    
    for state in returns_df.index:
        (1 + df[df['State']==state]['Returns']).cumprod().plot(ax=ax2, label=returns_df.loc[state, 'auto_name'])
    ax2.set_title('Cumulative Returns by State')
    ax2.legend()
    
    plt.tight_layout()
    return returns_df

def generate_trading_rules(returns_df, quantile_threshold=0.7):
    """Generate rules based on top/bottom quantile returns."""
    buy_thresh = returns_df['mean_return'].quantile(quantile_threshold)
    sell_thresh = returns_df['mean_return'].quantile(1 - quantile_threshold)
    
    rules = {}
    for state, row in returns_df.iterrows():
        if row['mean_return'] >= buy_thresh:
            rules[state] = 'Buy'
        elif row['mean_return'] <= sell_thresh:
            rules[state] = 'Sell'
        else:
            rules[state] = 'Hold'
    return rules

def load_and_preprocess_data(file_path, years_of_data=10):
    print(f"Loading S&P 500 data from {file_path}...")

    df = pd.read_csv(file_path, skiprows=2, header=None)

    # Set headers manually
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df.iloc[1:] # Remove the potentially problematic first row after header assignment

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    print("Original DataFrame head after loading all data:")
    print(df.head())
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Original data range: {df.index.min()} to {df.index.max()}")

    # --- Filter for the last 'years_of_data' years ---
    if not df.empty and years_of_data is not None:
        most_recent_date = df.index.max()
        start_date_filter = most_recent_date - pd.DateOffset(years=years_of_data)
        
        df = df[df.index >= start_date_filter]
        print(f"\nFiltered DataFrame for the last {years_of_data} years.")
        print(f"Data now ranges from: {df.index.min()} to {df.index.max()}")
        print(f"Filtered DataFrame shape: {df.shape}")
    elif df.empty:
        print("Warning: DataFrame is empty after initial loading. Cannot filter by date.")
        return df # Return empty df if it's already empty
    # ----------------------------------------------------

    # Ensure 'Close', 'High', 'Low', 'Open', 'Volume' are numeric, coercing errors
    cols_to_convert = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in cols_to_convert:
        # Replace non-numeric strings like '-' or empty strings with NaN before conversion
        df[col] = df[col].replace({'-': np.nan, '': np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where key financial data might be NaN after conversion
    # Especially 'Close' is critical for many calculations
    df.dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)
    
    if df.empty:
        print("Warning: DataFrame is empty after NaN removal post numeric conversion. Check data quality.")
        return df

    print("Adjusted DataFrame head after filtering and numeric conversion:")
    print(df.head())
    
    
    print("Calculating daily returns and volatility...")
    # Daily returns with safeguards (using Close price)
    
    print("Calculating daily returns and volatility...")
    # Daily returns with safeguards (using Close price)
    df['Returns'] = df['Close'].pct_change()
    df['Returns'] = df['Returns'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility using 20-day rolling window
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    print("Calculating volume metrics...")
    # Volume change with safeguards
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], np.nan)
    df['Volatility_Scaled'] = np.log(df['Volatility'])  # More continuous representation
    # Volume z-score using 20-day window
    df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

    print("Calculating technical indicators...")
    # Bollinger Bands (20-day)
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
    
    # RSI (14-day)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # MACD (12/26/9)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']


    # ATR (14-day)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['MACD_Strength'] = df['MACD_Hist'] / df['ATR']  # Normalized by volatility

    # EMA Deviation (20-day)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['Price_EMA_Diff'] = (df['Close'] - df['EMA_20']) / df['EMA_20']

    # ADX (14-day)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['Trend_Strength'] = df['ADX']

    df['Returns_t'] = df['Close'].pct_change()  # r_t (same as existing 'Returns')
    df['Returns_t_1'] = df['Returns_t'].shift(1)  # r_{t-1}
    df['Returns_t_2'] = df['Returns_t'].shift(2)  # r_{t-2} 
    print("Creating derived features...")
    # Volatility regime (top quartile = high volatility)
    df['Vol_Regime'] = np.where(df['Volatility'] > df['Volatility'].quantile(0.75), 1, 0)
        
    # RSI Z-score
    df['RSI_Zscore'] = (df['RSI'] - 50) / 20
    
    # Fisher Transform of RSI
    df['Fisher_RSI'] = 0.5 * np.log((1 + df['RSI']/100)/(1 - df['RSI']/100))
    df['Fisher_RSI_Smoothed'] = df['Fisher_RSI'].rolling(5).mean()
    df['Fisher_RSI_Scaled'] = np.tanh(df['Fisher_RSI_Smoothed'])
    df['Trend_Direction'] = np.sign(df['Close'].diff(5))
    df['Enhanced_Trend'] = df['ADX'] * df['Trend_Direction']
    df['Vol_Volume'] = df['Volume_Zscore'] * df['Volatility_Scaled']
    df['Vol_Regime_Score'] = df['Volatility_Scaled'] * np.sign(df['Volatility'].pct_change(5))
    df['Enhanced_Trend_v2'] = df['ADX'] * (df['RSI_Zscore'].clip(-1, 1))

    
    # Handle extreme values
    df['Volume_Change'] = df['Volume_Change'].clip(-3, 3)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Dropping NaN values...")
    df.dropna(inplace=True)

    print(f"S&P 500 data preprocessed. Shape: {df.shape}")
    print("\nPreview of processed data:")
    print("Available columns in training data:")
    print(df.head())
    return df


def save_state_analysis(state_returns, output_dir='hmm_results'):
    """
    Saves state performance and trading rules to CSV files
    """
    import os
    from datetime import datetime
    
    # Create directory if needed
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    rounded_returns = state_returns.copy()
    numeric_cols = state_returns.select_dtypes(include=[np.number]).columns
    rounded_returns[numeric_cols] = rounded_returns[numeric_cols].round(4)
    
    # Save state returns
    returns_path = os.path.join(output_dir, f'state_returns_{timestamp}.csv')
    rounded_returns.to_csv(returns_path)
    print(f"Saved state returns to: {returns_path}")



# Load state transitions
file_path = r'C:\Users\joshd\OneDrive - Monash University\Projects\Python\Trading\Intro_using_AI\data\sp500_daily_all.csv'
#hmm\7_regimes\volar_frsi_ts_volz\bitcoin_state_changes.csv
df = load_and_preprocess_data(file_path)

state_df = pd.read_csv(
    'hmm/models/stocks/2_selected_models/12_volr_frsi_ts_volz/sp500_state_changes.csv',
    parse_dates=['Start Time', 'End Time'],  # Now matches the stripped names
    skipinitialspace=True  # This removes leading spaces from column names
)

# Create a new column for state and initialize with NaN
df['State'] = np.nan

# Use tqdm for progress bar if many rows
for idx, row in tqdm(state_df.iterrows(), total=len(state_df)):
    mask = (df.index >= row['Start Time']) & (df.index <= row['End Time'])
    df.loc[mask, 'State'] = row['State']

state_returns = analyze_state_returns(
    data=df,
    states=df['State'],  # Pass the Series, not DataFrame
    min_samples=20
)
print(state_returns)

trading_rules = generate_trading_rules(state_returns, quantile_threshold=0.7)
print(pd.Series(trading_rules).to_frame('Action'))
save_state_analysis(state_returns)


