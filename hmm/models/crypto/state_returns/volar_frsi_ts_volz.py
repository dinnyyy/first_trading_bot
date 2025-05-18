'''
State
4.0       0.000331    0.187388  0.530357    0.006894  0.047948  Strong Bull
6.0       0.000158    1.895742  0.514524    0.004422  0.035634         Bull
2.0       0.000102    0.247858  0.505455    0.008401  0.012184      Neutral
1.0       0.000054    0.014582  0.502041    0.009876  0.005483         Bear
0.0      -0.000009   -0.170576  0.504358    0.003379 -0.002734  Strong Bear
'''

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
        rank_names = ["Strong Bull", "Bull", "Neutral", "Bear", "Strong Bear"]
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

def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    
    # Read CSV with custom column names, no header
    df = pd.read_csv(file_path)

    print("Creating datetime index...")
    # Current time in milliseconds
    now_ms = exchange.milliseconds()

    # Subtract 5 years in milliseconds
    five_years_ago_ms = now_ms - int(5 * 365.25 * 24 * 60 * 60 * 1000)

    # Convert to datetime
    five_years_ago_dt = datetime.fromtimestamp(five_years_ago_ms / 1000, tz=timezone.utc)
    # Format as yyyy-mm-dd
    formatted_date = five_years_ago_dt.strftime('%Y-%m-%d')

    df.index = pd.date_range(start=formatted_date, periods=len(df), freq='h')
    
    print("Calculating returns and volatility...")
    # Calculate returns with safeguards
    df['Returns'] = df['Close'].pct_change()
    df['Returns'] = df['Returns'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate volatility
    df['volatility'] = df['Returns'].rolling(window=24).std()

    print("Calculating volume change...")
    # Calculate volume change with safeguards
    df['volume_change'] = df['Volume'].pct_change()

    print("Calculating Bollinger Bands...")
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']  # Middle band
    
    # === RSI ===
    df['rsi'] = ta.rsi(df['Close'], length=14)

    # === MACD ===
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    # === ATR ===
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # === EMA Deviation ===
    df['ema_20'] = ta.ema(df['Close'], length=20)
    df['price_ema_diff'] = (df['Close'] - df['ema_20']) / df['ema_20']

    df['vol_regime'] = np.where(df['volatility'] > df['volatility'].quantile(0.75), 1, 0)
    df['trend_strength'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['scaled_vol'] = np.log1p(df['volatility'])  # Reduce magnitude
    df['rsi_zscore'] = (ta.rsi(df['Close']) - 50) / 20  # More sensitive

    df['fisher_rsi'] = 0.5 * np.log((1 + df['rsi']/100)/(1 - df['rsi']/100))  # Fisher Transform
    df['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(24).mean()) / df['Volume'].rolling(24).std()
    df['adx'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    
    # Handle extreme volume changes
    volume_change = df['volume_change']
    valid_changes = volume_change[~np.isinf(volume_change)]
    
    # Calculate reasonable bounds (e.g., 3 standard deviations)
    if len(valid_changes) > 0:
        mean_change = valid_changes.mean()
        std_change = valid_changes.std()
        upper_bound = mean_change + 3 * std_change
        lower_bound = mean_change - 3 * std_change
        
        # Clip volume changes to these bounds
        df['volume_change'] = df['volume_change'].clip(lower_bound, upper_bound)
    else:
        # If no valid changes, replace inf values with 0
        df['volume_change'] = df['volume_change'].replace([np.inf, -np.inf], 0)

    print("Dropping NaN values...")
    df.dropna(inplace=True)

    print(f"Data preprocessed. Shape: {df.shape}")
    return df

# Load state transitions
file_path = r'C:\Users\joshd\OneDrive - Monash University\Projects\Python\Trading\Intro_using_AI\data\btc_1h_3years.csv'
#hmm\7_regimes\volar_frsi_ts_volz\bitcoin_state_changes.csv
df = load_and_preprocess_data(file_path)

state_df = pd.read_csv(
    'hmm/7_regimes/volar_frsi_ts_volz/bitcoin_state_changes.csv',
    parse_dates=['Start Time', 'End Time'],  # Now matches the stripped names
    skipinitialspace=True  # This removes leading spaces from column names
)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a new column for state and initialize with NaN
df['State'] = np.nan

# Use tqdm for progress bar if many rows
for idx, row in tqdm(state_df.iterrows(), total=len(state_df)):
    mask = (df['timestamp'] >= row['Start Time']) & (df['timestamp'] <= row['End Time'])
    df.loc[mask, 'State'] = row['State']

state_returns = analyze_state_returns(
    data=df,
    states=df['State'],  # Pass the Series, not DataFrame
    min_samples=20
)
print(state_returns)

trading_rules = generate_trading_rules(state_returns, quantile_threshold=0.7)
print(pd.Series(trading_rules).to_frame('Action'))

