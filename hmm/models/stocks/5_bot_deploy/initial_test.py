import alpaca_trade_api as tradeapi
import joblib
import pandas as pd
import numpy as np
import pandas_ta as ta # Or your TA library
import time
from datetime import datetime, timedelta
import os # For API keys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# --- CONFIGURATION ---
API_KEY = os.environ.get('APCA_API_KEY_ID') # Best practice: use environment variables
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets' # e.g., 'https://paper-api.alpaca.markets'
# For live trading, change to 'https://api.alpaca.markets'

SYMBOL = 'SPY' # Or whatever S&P 500 ETF/futures you're trading
N_HISTORICAL_BARS = 100 # Number of bars needed for your indicators + HMM (e.g., longest rolling window)
BAR_TIMEFRAME = '1Day' # Or '1Min', '5Min', '1Hour' - must match your strategy's timeframe
# Ensure this timeframe provides enough data for your indicators (e.g., 20-day rolling std needs >20 days)

MODEL_PATH = 'model.joblib'
SCALER_PATH = 'scaler.joblib'
FEATURES = ['Vol_Regime', 'Fisher_RSI', 'Trend_Strength', 'Volume_Zscore'] # Your HMM features

# --- Initialize Alpaca API ---
try:
    api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
    account = api.get_account()
    print(f"Connected to Alpaca. Account Status: {account.status}")
except Exception as e:
    print(f"Error connecting to Alpaca: {e}")
    exit()

# --- Load HMM Model and Scaler ---
try:
    hmm_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("HMM model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit()

def get_latest_market_data(symbol, timeframe, limit):
    """Fetches the latest market data bars from Alpaca."""
    try:
        # Adjust 'limit' to ensure you have enough data for all lookback periods
        # For daily data, 'limit' is number of trading days.
        # For intraday, Alpaca might have limitations on how far back you can go with high frequency.
        end_dt = pd.Timestamp.now(tz='America/New_York') # Alpaca uses America/New_York
        start_dt = end_dt - pd.Timedelta(days=limit * 1.5) # Fetch a bit more to be safe for trading days for daily
                                                        # For intraday, adjust accordingly (e.g., limit * 2 * timeframe_minutes)

        # Use get_bars for wider historical range if needed
        bars_df = api.get_bars(symbol, timeframe,
                               start=start_dt.strftime('%Y-%m-%dT%H:%M:%S-04:00'), # Explicit timezone
                               end=end_dt.strftime('%Y-%m-%dT%H:%M:%S-04:00'),
                               adjustment='raw').df # Get as pandas DataFrame

        # Ensure data is sorted by time and has the correct columns
        bars_df = bars_df[['open', 'high', 'low', 'close', 'volume']]
        bars_df.index = pd.to_datetime(bars_df.index) # Ensure index is datetime
        bars_df = bars_df[~bars_df.index.duplicated(keep='last')] # Remove duplicates if any
        bars_df = bars_df.sort_index()

        # Select only the required number of bars from the end
        if len(bars_df) >= limit:
            return bars_df.iloc[-limit:] # Return the most recent 'limit' bars
        else:
            print(f"Warning: Fetched {len(bars_df)} bars, less than required {limit}.")
            return bars_df # Or handle error if not enough data

    except Exception as e:
        print(f"Error fetching market data for {symbol}: {e}")
        return pd.DataFrame()


def calculate_features(df):
    """Calculates all necessary features for the HMM model."""
    # --- THIS SECTION MUST EXACTLY MIRROR YOUR `load_and_preprocess_data` FEATURE ENGINEERING ---
    # --- Only calculate features for the *last* row if that's what your HMM expects for live prediction ---
    # --- Or, if your HMM uses features derived from a sequence, ensure df has enough history ---
    # --- Be careful about lookahead bias: use .iloc[:-1] for calculations if predicting for current open bar ---

    # Example: (ensure these match your training preprocessing)
    # It's crucial that these calculations are identical to your training phase
    
    # Ensure columns are correctly named if Alpaca returns different names (e.g., 'Close' vs 'close')
    # The api.get_bars().df usually returns 'open', 'high', 'low', 'close', 'volume'

    temp_df = df.copy() # Work on a copy

    # Make sure your column names here match the output of get_latest_market_data
    # e.g., if get_latest_market_data returns 'close', use 'close' not 'Close'
    # It's good practice to standardize column names early.
    # For this example, I'll assume they are 'Close', 'High', 'Low', 'Volume'
    # You might need to rename them:
    temp_df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)


    temp_df['Returns'] = temp_df['Close'].pct_change()
    temp_df['Volatility'] = temp_df['Returns'].rolling(window=20).std() # Example window

    temp_df['Volume_Change'] = temp_df['Volume'].pct_change()
    temp_df['Volume_Zscore'] = (temp_df['Volume'] - temp_df['Volume'].rolling(20).mean()) / temp_df['Volume'].rolling(20).std()

    bb = ta.bbands(temp_df['Close'], length=20, std=2)
    if bb is not None and not bb.empty:
        temp_df['BB_Width'] = (bb[f'BBU_{20}_{2.0}'] - bb[f'BBL_{20}_{2.0}']) / bb[f'BBM_{20}_{2.0}']
    else:
        temp_df['BB_Width'] = np.nan

    temp_df['RSI'] = ta.rsi(temp_df['Close'], length=14)
    
    adx = ta.adx(temp_df['High'], temp_df['Low'], temp_df['Close'], length=14)
    if adx is not None and not adx.empty:
        temp_df['ADX'] = adx['ADX_14']
        temp_df['Trend_Strength'] = temp_df['ADX'] # Assuming Trend_Strength is ADX
    else:
        temp_df['ADX'] = np.nan
        temp_df['Trend_Strength'] = np.nan # Ensure this is handled if ADX fails

    # Derived features (as in your training)
    temp_df['Vol_Regime'] = np.where(temp_df['Volatility'] > temp_df['Volatility'].quantile(0.75), 1, 0) # Quantile might need adjustment if data distribution changes
    temp_df['Fisher_RSI'] = 0.5 * np.log((1 + temp_df['RSI']/100)/(1 - temp_df['RSI']/100))
    
    temp_df['Volume_Change'] = temp_df['Volume_Change'].clip(-3, 3)
    temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Return only the features needed for the HMM for the latest complete bar
    # The HMM model expects a 2D array for prediction, even if it's for one sample
    latest_features = temp_df[FEATURES].iloc[-1:].copy() # Get the last row as a DataFrame

    # Handle any NaNs in the latest_features (e.g., fill with a neutral value or skip prediction)
    # This is critical. If NaNs are present, scaling or prediction will fail.
    if latest_features.isnull().values.any():
        print(f"Warning: NaN values found in latest calculated features for {temp_df.index[-1]}:")
        print(latest_features[latest_features.isnull().any(axis=1)])
        return pd.DataFrame() # Indicate failure

    return latest_features


def predict_hmm_state(features_df):
    """Predicts HMM state using the loaded model and scaler."""
    if features_df.empty:
        return None
    try:
        scaled_features = scaler.transform(features_df)
        predicted_state = hmm_model.predict(scaled_features)
        return predicted_state[0] # Return the single predicted state
    except Exception as e:
        print(f"Error predicting HMM state: {e}")
        print(f"Features causing error: \n{features_df}")
        return None

def execute_trade_strategy(current_state, symbol):
    """Executes trades based on the HMM state and strategy rules."""
    # --- THIS IS WHERE YOUR `HMMstatestrategy.html` LOGIC IS IMPLEMENTED ---
    # Example:
    # optimal_states_for_buy = [0, 3, 5] # From your backtest
    # optimal_states_for_sell_or_exit = [2, 4] # From your backtest

    print(f"Current predicted HMM State: {current_state}")
    
    # Placeholder for your specific strategy logic from HMMstatestrategy.html
    # This will be highly custom based on how your HTML translates to rules.
    # For example, if your HTML implies "State X means go long, State Y means go short/flat":
    
    # Get current position
    try:
        position = api.get_position(symbol)
        current_qty = int(position.qty)
        print(f"Current position in {symbol}: {current_qty} shares.")
    except tradeapi.rest.APIError as e: # No position exists
        if e.status_code == 404:
            current_qty = 0
            print(f"No current position in {symbol}.")
        else:
            print(f"Error getting position: {e}")
            return

    # Define desired position based on state (EXAMPLE LOGIC)
    # You need to translate your HMMstatestrategy.html rules here
    desired_qty = 0
    if current_state == 0: # Replace with your "bullish" state from backtest
        print("Strategy: Bullish state detected. Target: Long position.")
        desired_qty = 10 # Example: buy 10 shares
    elif current_state == 1: # Replace with your "bearish" state
        print("Strategy: Bearish state detected. Target: Short or flat position.")
        desired_qty = -10 # Example: short 10 shares (if allowed and desired) or 0 for flat
    elif current_state == 2: # Replace with your "neutral/exit" state
        print("Strategy: Neutral/Exit state detected. Target: Flat position.")
        desired_qty = 0
    # Add more states and logic as per your strategy...

    qty_to_trade = desired_qty - current_qty

    if qty_to_trade > 0:
        side = 'buy'
        print(f"Placing BUY order for {abs(qty_to_trade)} shares of {symbol}.")
    elif qty_to_trade < 0:
        side = 'sell'
        print(f"Placing SELL order for {abs(qty_to_trade)} shares of {symbol}.")
    else:
        print(f"No trade needed. Current position matches desired position for state {current_state}.")
        return

    if qty_to_trade != 0:
        try:
            api.submit_order(
                symbol=symbol,
                qty=abs(qty_to_trade),
                side=side,
                type='market', # Or 'limit' with a price
                time_in_force='day' # Or 'gtc'
            )
            print(f"Order submitted: {side} {abs(qty_to_trade)} {symbol}")
        except Exception as e:
            print(f"Error submitting order: {e}")

def run_bot():
    """Main bot execution logic."""
    print(f"\n--- Running Bot Cycle: {datetime.now()} ---")
    
    # 1. Get Data
    # Adjust N_HISTORICAL_BARS to be slightly more than the maximum lookback period
    # of your indicators (e.g., if longest is 20-day rolling, get at least 50-60 bars
    # to ensure stable calculation for the most recent bar)
    market_data_df = get_latest_market_data(SYMBOL, BAR_TIMEFRAME, N_HISTORICAL_BARS + 40) # +40 for buffer
    if market_data_df.empty or len(market_data_df) < N_HISTORICAL_BARS : # Need enough for features
        print(f"Insufficient market data to proceed. Fetched: {len(market_data_df)}")
        return

    # 2. Calculate Features
    # We need to ensure the features are calculated for the *latest complete bar*
    # If BAR_TIMEFRAME is '1Day', this usually means yesterday's close.
    # If BAR_TIMEFRAME is '1Min', this means the close of the last minute's bar.
    # The HMM was trained on features from *complete* bars.
    
    # For daily, usually predict based on previous day's close data to trade at today's open/during day
    # If your strategy is based on the *close* of the current bar to trade on the *next bar's open*,
    # then the features should be calculated on all data *up to and including the latest closed bar*.
    # The `market_data_df.iloc[-1:]` in `calculate_features` assumes this.
    
    features_for_prediction = calculate_features(market_data_df) # market_data_df should contain all bars needed for lookbacks

    if features_for_prediction.empty:
        print("Could not calculate features for prediction (likely due to NaNs or insufficient data). Skipping cycle.")
        return

    # 3. Predict State
    predicted_state = predict_hmm_state(features_for_prediction)
    if predicted_state is None:
        print("Could not predict HMM state. Skipping cycle.")
        return

    # 4. Execute Strategy
    if not api.get_clock().is_open:
        print(f"Market is currently closed. Predicted state {predicted_state}, but no trading actions taken.")
        # You might still want to log the state or update desired positions for next open
    else:
        execute_trade_strategy(predicted_state, SYMBOL)
    
    print("--- Bot Cycle Complete ---")

# --- Main Loop / Scheduler ---
if __name__ == "__main__":
    # For a simple test, run once:
    run_bot() 

    # For continuous operation:
    # Use a scheduler like 'schedule' or APScheduler
    # # Example with 'schedule' library (pip install schedule)
    # import schedule
    
    # if BAR_TIMEFRAME == '1Day':
    #     # Example: Run once a day after market open (adjust time as needed)
    #     # Ensure your data fetching gets data *up to yesterday's close* if trading at today's open
    #     schedule.every().day.at("09:35", "America/New_York").do(run_bot) # Example: 5 mins after open
    #     print("Scheduled daily run at 09:35 New York time.")
    # elif 'Min' in BAR_TIMEFRAME or 'Hour' in BAR_TIMEFRAME:
    #     interval = int(BAR_TIMEFRAME.replace('Min','').replace('Hour',''))
    #     if 'Min' in BAR_TIMEFRAME:
    #         schedule.every(interval).minutes.do(run_bot)
    #         print(f"Scheduled run every {interval} minutes.")
    #     elif 'Hour' in BAR_TIMEFRAME:
    #         schedule.every(interval).hours.do(run_bot)
    #         print(f"Scheduled run every {interval} hours.")
    # else:
    #     print(f"Unsupported BAR_TIMEFRAME for scheduling: {BAR_TIMEFRAME}. Running once for test.")
    #     run_bot()
    #     exit()

    # print(f"Bot started. Waiting for scheduled runs... Press Ctrl+C to exit.")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)