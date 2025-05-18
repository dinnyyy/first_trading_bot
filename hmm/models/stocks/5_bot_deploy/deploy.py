import alpaca_trade_api as tradeapi
import joblib
import pandas as pd
import numpy as np
import pandas_ta as ta # Or your TA library
import time
from datetime import datetime, timedelta
import os # For API keys
from dotenv import load_dotenv # 1. Import the load_dotenv function

load_dotenv()

API_KEY = os.environ.get('APCA_API_KEY_ID')
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')
BASE_URL = os.environ.get('APCA_PAPER_URL')

SYMBOL = 'SPY'
N_HISTORICAL_BARS = 34
BAR_TIMEFRAME = '1Day' 

MODEL_PATH = 'hmm/models/stocks/2_selected_models/5_returnsl_rsi_macd_vola/model.joblib'
SCALER_PATH = 'hmm/models/stocks/2_selected_models/5_returnsl_rsi_macd_vola/scaler.joblib'
FEATURES = [
    'Returns_t',       # Current return
    'Returns_t_1',     # 1-period lagged return
    'Returns_t_2',     # 2-period lagged return
    'RSI',             # RSI(14)
    'MACD_Signal',     # MACD signal line
    'Volatility'       # 20-day rolling std
]

TRADING_RULES_HMM = {
    3.0: 'Sell',  
    0.0: 'Sell',  
    1.0: 'Hold',  
    2.0: 'Buy',    
    4.0: 'Buy'      
}

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
        end_dt = pd.Timestamp.now(tz='America/New_York') # Alpaca uses America/New_York
        start_dt = end_dt - pd.Timedelta(days=limit * 1.5)
        bars_df = api.get_bars(symbol, timeframe,
                           start=start_dt.strftime('%Y-%m-%dT%H:%M:%S-04:00'), # Explicit timezone
                           end=end_dt.strftime('%Y-%m-%dT%H:%M:%S-04:00'),
                           adjustment='raw').df # Get as pandas DataFrame
        # Ensure data is sorted by time and has the correct columns
        bars_df = bars_df[['open', 'high', 'low', 'close', 'volume']]
        bars_df.index = pd.to_datetime(bars_df.index) # Ensure index is datetime
        bars_df = bars_df[~bars_df.index.duplicated(keep='last')] # Remove duplicates if any
        bars_df = bars_df.sort_index()

        if len(bars_df) >= limit:
            return bars_df.iloc[-limit:] # Return the most recent 'limit' bars
        else:
            print(f"Warning: Fetched {len(bars_df)} bars, less than required {limit}.")
            return bars_df # Or handle error if not enough data

    except Exception as e:
        print(f"Error fetching market data for {symbol}: {e}")
        return pd.DataFrame()


def calculate_hmm_features(df_full_history):
    """Calculates ONLY the features needed by the HMM model for the latest bar."""
    # df_full_history should have enough data for all lookbacks of HMM features
    df = df_full_history.copy() # Work on a copy

    # Calculate all potential components for HMM features
    df['Returns_t_calc'] = df['Close'].pct_change()
    df['Returns_t_1_calc'] = df['Returns_t_calc'].shift(1)
    df['Returns_t_2_calc'] = df['Returns_t_calc'].shift(2)
    
    df['RSI_calc'] = ta.rsi(df['Close'], length=14)

    macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd_data is not None and not macd_data.empty:
        df['MACD_Signal_calc'] = macd_data['MACDs_12_26_9']
    else:
        df['MACD_Signal_calc'] = np.nan

    # Volatility calculation
    # Assuming 'Returns_t_calc' is the basis for the 'Volatility' HMM feature
    df['Volatility_calc'] = df['Returns_t_calc'].rolling(window=20).std()

    # Prepare a DataFrame with columns named exactly as in the FEATURES list
    hmm_features_df = pd.DataFrame(index=df.index)
    hmm_features_df['Returns_t'] = df['Returns_t_calc']
    hmm_features_df['Returns_t_1'] = df['Returns_t_1_calc']
    hmm_features_df['Returns_t_2'] = df['Returns_t_2_calc']
    hmm_features_df['RSI'] = df['RSI_calc']
    hmm_features_df['MACD_Signal'] = df['MACD_Signal_calc']
    hmm_features_df['Volatility'] = df['Volatility_calc']
    
    # Select only the HMM features for the latest bar
    latest_hmm_features = hmm_features_df[FEATURES].iloc[-1:].copy()

    if latest_hmm_features.isnull().values.any():
        print(f"Warning: NaN values found in latest HMM features for {df.index[-1]}:")
        print(latest_hmm_features[latest_hmm_features.isnull().any(axis=1)])
        return pd.DataFrame()

    return latest_hmm_features


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

def get_current_price_and_account_equity(symbol):
    """Gets the latest price and account equity."""
    latest_price = None
    account_equity = None
    try:
        # Get latest trade for price (can also use latest quote or last bar close)
        latest_trade = api.get_latest_trade(symbol)
        latest_price = latest_trade.p
    except Exception as e:
        print(f"Error getting latest price for {symbol}: {e}")

    try:
        account_info = api.get_account()
        account_equity = float(account_info.equity)
    except Exception as e:
        print(f"Error getting account equity: {e}")
    
    return latest_price, account_equity

def execute_trade_strategy(current_hmm_state, symbol, current_features_df):
    """
    Executes trades based on HMM state, RSI, ATR, and volatility,
    with risk management.

    Args:
        current_hmm_state (float/int): The predicted HMM state.
        symbol (str): The trading symbol ('SPY').
        current_features_df (pd.DataFrame): DataFrame containing the latest calculated features
                                            including 'RSI', 'ATR', and 'Volatility'.
                                            Expected to be a single row.
    """
    print(f"--- Executing Trade Strategy for HMM State: {current_hmm_state} ---")

    action = TRADING_RULES_HMM.get(float(current_hmm_state), 'Hold') # Default to 'Hold' if state not in rules
    print(f"HMM State {current_hmm_state} translates to action: {action}")

    if action == 'Hold':
        print("Strategy: Hold. No trading action taken based on HMM state.")
        return

    # Fetch necessary values from the features DataFrame
    # Ensure these column names match what calculate_features() produces
    try:
        rsi = current_features_df['RSI'].iloc[0]
        atr = current_features_df['ATR'].iloc[0] 
        volatility_metric = current_features_df['Volatility'].iloc[0] # Using the HMM feature 'Volatility'
        print(f"Extracted features: RSI={rsi:.2f}, ATR={atr:.2f}, VolatilityMetric={volatility_metric:.4f}")
    except KeyError as e:
        print(f"Error: Feature {e} not found in current_features_df. Columns: {current_features_df.columns}")
        return
    except IndexError:
        print("Error: current_features_df is empty. Cannot extract features.")
        return
        
    price, cash = get_current_price_and_account_equity(symbol)

    if price is None or cash is None:
        print("Could not retrieve current price or account equity. Skipping trade execution.")
        return
    print(f"Current Price: {price}, Account Equity: {cash:.2f}")

    # --- Your execute_trade logic adapted for Alpaca ---
    OVERBOUGHT = 80
    OVERSOLD = 20
    MAX_RISK_PER_TRADE = 0.02  # 2% of total equity

    # Risk per trade in dollars
    dollar_risk = MAX_RISK_PER_TRADE * cash

    # ATR-based SL/TP
    sl_distance = 1.5 * atr
    tp_distance = 3.0 * atr

    # Calculate position size based on ATR risk
    if atr == 0 or sl_distance == 0: # Added check for atr == 0 as well
        print("ATR or SL distance is zero; cannot compute position size or SL/TP. Skipping trade.")
        return
    
    # Ensure position_size is at least 1 if any trade is to be made,
    # and that dollar_risk / sl_distance is not negative (sl_distance should be positive)
    if sl_distance <= 0:
        print(f"SL distance ({sl_distance:.2f}) is not positive. Cannot calculate position size. Skipping trade.")
        return
        
    position_size_calculated = dollar_risk / sl_distance
    if position_size_calculated < 1: # Cannot trade fractional shares of SPY easily this way
        print(f"Calculated position size ({position_size_calculated:.2f}) is less than 1. Risk per trade or equity might be too low for current ATR. Skipping trade.")
        return
    position_size = int(position_size_calculated) # Final position size to trade
    
    print(f"Dollar Risk: ${dollar_risk:.2f}, SL Distance: ${sl_distance:.2f}, TP Distance: ${tp_distance:.2f}, Calculated Position Size: {position_size}")

    # Get current position with Alpaca
    current_qty = 0
    is_long = False
    is_short = False
    try:
        position = api.get_position(symbol)
        current_qty = float(position.qty) # Alpaca qty can be float for fractional
        if current_qty > 0:
            is_long = True
        elif current_qty < 0:
            is_short = True
        print(f"Alpaca: Current position in {symbol}: {current_qty} shares.")
    except tradeapi.rest.APIError as e:
        if e.status_code == 404: # HTTP 404 Error: "position does not exist"
            print(f"Alpaca: No current position in {symbol}.")
            current_qty = 0
        else:
            print(f"Alpaca: Error getting position: {e}")
            return # Don't proceed if we can't get position info

    # --- Trading Logic ---
    if action == 'Buy':
        # Conditions: Not already long, and RSI is not overbought
        if not is_long and rsi < OVERBOUGHT:
            if is_short: # If currently short, close the short position first
                print(f"Closing existing short position of {abs(current_qty)} shares before buying.")
                try:
                    api.close_position(symbol) # Closes entire position
                    print(f"Short position closed for {symbol}.")
                    # Wait a moment for position to update if necessary, or re-fetch
                    time.sleep(2) # Small delay
                except Exception as e_close:
                    print(f"Error closing short position: {e_close}")
                    return # Don't proceed with buy if close failed
            
            # Place new buy order with SL/TP
            print(f"Attempting to BUY {position_size} shares of {symbol} with SL/TP.")
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=position_size,
                    side='buy',
                    type='market',
                    time_in_force='day', # Good Till Day
                    order_class='oto',  # One-Triggers-Other for SL/TP (bracket order)
                    stop_loss={'stop_price': round(price - sl_distance, 2)},
                    take_profit={'limit_price': round(price + tp_distance, 2)}
                )
                print(f"BUY order for {position_size} {symbol} submitted with SL: {price - sl_distance:.2f}, TP: {price + tp_distance:.2f} (RSI: {rsi:.1f})")
            except Exception as e_buy:
                print(f"Alpaca: BUY order failed: {e_buy} (RSI: {rsi:.1f})")
        elif is_long:
            print(f"Strategy: Buy signal, but already long {current_qty} shares. No action.")
        elif rsi >= OVERBOUGHT:
            print(f"Strategy: Buy signal, but RSI ({rsi:.1f}) >= {OVERBOUGHT}. No action.")

    elif action == 'Sell':
        # Conditions: Not already short, and RSI is not oversold
        if not is_short and rsi > OVERSOLD:
            if is_long: # If currently long, close the long position first
                print(f"Closing existing long position of {abs(current_qty)} shares before selling/shorting.")
                try:
                    api.close_position(symbol) # Closes entire position
                    print(f"Long position closed for {symbol}.")
                    time.sleep(2) # Small delay
                except Exception as e_close:
                    print(f"Error closing long position: {e_close}")
                    return # Don't proceed if close failed

            # Place new sell/short order with SL/TP
            print(f"Attempting to SELL/SHORT {position_size} shares of {symbol} with SL/TP.")
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=position_size, # For shorting, qty is positive
                    side='sell',
                    type='market',
                    time_in_force='day',
                    order_class='oto',
                    stop_loss={'stop_price': round(price + sl_distance, 2)},
                    take_profit={'limit_price': round(price - tp_distance, 2)}
                )
                print(f"SELL/SHORT order for {position_size} {symbol} submitted with SL: {price + sl_distance:.2f}, TP: {price - tp_distance:.2f} (RSI: {rsi:.1f})")
            except Exception as e_sell:
                print(f"Alpaca: SELL/SHORT order failed: {e_sell} (RSI: {rsi:.1f})")
        elif is_short:
            print(f"Strategy: Sell signal, but already short {current_qty} shares. No action.")
        elif rsi <= OVERSOLD:
            print(f"Strategy: Sell signal, but RSI ({rsi:.1f}) <= {OVERSOLD}. No action.")
    
    print("--- Trade Strategy Execution Complete ---")

def run_bot():
    """Main bot execution logic."""
    print(f"\n--- Running Bot Cycle: {datetime.now()} ---")

    # 1. Determine Data Fetch Limit
    lookback_rsi_strat = 14
    lookback_atr_strat = 14
    lookback_vol_strat = 20 

    min_bars_for_hmm_calc = N_HISTORICAL_BARS # This must be >= max lookback in HMM features
    min_bars_for_strat_calc = max(lookback_rsi_strat, lookback_atr_strat, lookback_vol_strat) + 1 # +1 for pct_change if needed

    data_fetch_limit = max(min_bars_for_hmm_calc, min_bars_for_strat_calc)
    print(f"Determined data fetch limit: {data_fetch_limit} bars (plus buffer).")
    
    market_data_df_full = get_latest_market_data(SYMBOL, BAR_TIMEFRAME, data_fetch_limit + 40)

    if market_data_df_full.empty or len(market_data_df_full) < data_fetch_limit:
        print(f"Insufficient market data ({len(market_data_df_full)} fetched, need at least {data_fetch_limit}). Skipping cycle.")
        return

    # 2. Calculate HMM Features

    if len(market_data_df_full) < N_HISTORICAL_BARS:
         print(f"Not enough data in market_data_df_full ({len(market_data_df_full)}) for HMM features ({N_HISTORICAL_BARS}).")
         return
    hmm_input_data_slice = market_data_df_full.iloc[-N_HISTORICAL_BARS:].copy()
    features_for_hmm_prediction = calculate_hmm_features(hmm_input_data_slice)

    if features_for_hmm_prediction.empty:
        print("Could not calculate HMM features. Skipping cycle.")
        return

    # 3. Predict HMM State
    predicted_state = predict_hmm_state(features_for_hmm_prediction)
    if predicted_state is None:
        print("Could not predict HMM state. Skipping cycle.")
        return

    # 4. Calculate Strategy-Specific Technical Indicators from the full fetched dataset
    # These are for the 'execute_trade_strategy' function and calculated on market_data_df_full
    # to use the most available data for their calculation.
    
    # Ensure columns are named 'High', 'Low', 'Close' for pandas_ta
    df_for_strat_ta = market_data_df_full.copy() # Use the full fetched df

    current_atr_value = ta.atr(df_for_strat_ta['High'], df_for_strat_ta['Low'], df_for_strat_ta['Close'], length=lookback_atr_strat).iloc[-1]
    current_rsi_strat_value = ta.rsi(df_for_strat_ta['Close'], length=lookback_rsi_strat).iloc[-1]
    
    # For strategy volatility, calculate it based on your definition
    df_for_strat_ta['Returns_strat_calc'] = df_for_strat_ta['Close'].pct_change()
    current_volatility_strat_value = df_for_strat_ta['Returns_strat_calc'].rolling(window=lookback_vol_strat).std().iloc[-1]

    print(f"Strategy Indicators for latest bar: ATR={current_atr_value:.2f}, RSI={current_rsi_strat_value:.2f}, Volatility={current_volatility_strat_value:.4f}")

    # 5. Execute Strategy
    if not api.get_clock().is_open:
        print(f"Market is currently closed. Predicted HMM state {predicted_state}, but no trading actions taken.")
    else:
        execute_trade_strategy(
            current_hmm_state=predicted_state,
            symbol=SYMBOL,
            rsi_strat=current_rsi_strat_value,
            atr_strat=current_atr_value,
            volatility_strat=current_volatility_strat_value
        )
    
    print("--- Bot Cycle Complete ---")

# --- Main Loop / Scheduler ---
if __name__ == "__main__":
    # For a simple test, run once:
    run_bot() 

    # For continuous operation:
    # Use a scheduler like 'schedule' or APScheduler
    # Example with 'schedule' library (pip install schedule)
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