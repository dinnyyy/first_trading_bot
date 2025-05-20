# lambda_function.py

import alpaca_trade_api as tradeapi # For trading client
from alpaca.data import StockHistoricalDataClient # For market data client
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import joblib
import pandas as pd
import numpy as np
import pandas_ta as ta
import time # For potential small delays after closing positions
from datetime import datetime, timedelta
import os
import pytz

# --- Load Environment Variables (best practice for Lambda) ---
# These will be set in the Lambda function's configuration
API_KEY = os.environ.get('APCA_API_KEY_ID')
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')
BASE_URL = os.environ.get('APCA_PAPER_URL') # For trading client
# Note: StockHistoricalDataClient doesn't use BASE_URL in the same way; it defaults.

# --- CONFIGURATION (can also be Lambda environment variables) ---
SYMBOL = os.environ.get('SYMBOL', 'SPY')
N_HISTORICAL_BARS_FOR_HMM = int(os.environ.get('N_HISTORICAL_BARS_FOR_HMM', 34))
# Determine max lookback for strategy TAs separately
LOOKBACK_RSI_STRAT = 14
LOOKBACK_ATR_STRAT = 14
LOOKBACK_VOL_STRAT = 20
MIN_BARS_FOR_STRAT_TA = max(LOOKBACK_RSI_STRAT, LOOKBACK_ATR_STRAT, LOOKBACK_VOL_STRAT) + 1 # +1 for pct_change if needed

BAR_TIMEFRAME_STR = os.environ.get('BAR_TIMEFRAME', '1Day') # e.g., '1Day', '1Hour', '5Min'
# Convert string to Alpaca TimeFrame object
if BAR_TIMEFRAME_STR == '1Day':
    ALPACA_TIMEFRAME = TimeFrame.Day
elif BAR_TIMEFRAME_STR == '1Hour':
    ALPACA_TIMEFRAME = TimeFrame.Hour
elif 'Min' in BAR_TIMEFRAME_STR:
    try:
        minutes = int(BAR_TIMEFRAME_STR.replace('Min',''))
        ALPACA_TIMEFRAME = TimeFrame.Minute_granularity(TimeFrame.Minute, minutes) # Or just TimeFrame.Minute if 1Min
        if minutes == 1: ALPACA_TIMEFRAME = TimeFrame.Minute
        elif minutes == 5: ALPACA_TIMEFRAME = TimeFrame.Minute_granularity(TimeFrame.Minute, 5) # Be specific
        # Add more cases for 15Min, 30Min etc. if needed
        else: raise ValueError(f"Unsupported minute timeframe: {minutes}")
    except ValueError:
        print(f"Error: Invalid BAR_TIMEFRAME '{BAR_TIMEFRAME_STR}'. Defaulting to Day.")
        ALPACA_TIMEFRAME = TimeFrame.Day
else:
    print(f"Warning: Unrecognized BAR_TIMEFRAME '{BAR_TIMEFRAME_STR}'. Defaulting to Day.")
    ALPACA_TIMEFRAME = TimeFrame.Day


MODEL_FILE_NAME = 'model.joblib'    # Assumed to be in the root of the ZIP
SCALER_FILE_NAME = 'scaler.joblib'  # Assumed to be in the root of the ZIP

FEATURES = [
    'Returns_t', 'Returns_t_1', 'Returns_t_2',
    'RSI', 'MACD_Signal', 'Volatility'
]

TRADING_RULES_HMM = { # Using float keys as predicted_state is float
    3.0: 'Sell', 0.0: 'Sell', 1.0: 'Hold',
    2.0: 'Buy',  4.0: 'Buy'
}

# --- Global variables for "warm starts" ---
trading_client = None
market_data_client = None
hmm_model_global = None
scaler_global = None

def initialize_globals():
    global trading_client, market_data_client, hmm_model_global, scaler_global
    
    if trading_client is None:
        print("Initializing Alpaca Trading API client...")
        try:
            trading_client = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
            trading_client.get_account() # Test connection
            print("Trading client initialized.")
        except Exception as e:
            print(f"FATAL: Error initializing trading client: {e}")
            raise # Re-raise to stop Lambda execution if client fails

    if market_data_client is None:
        print("Initializing Alpaca Market Data client...")
        try:
            market_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
            print("Market data client initialized.")
        except Exception as e:
            print(f"FATAL: Error initializing market data client: {e}")
            raise

    if hmm_model_global is None:
        print(f"Loading HMM model from {MODEL_FILE_NAME}...")
        try:
            hmm_model_global = joblib.load(MODEL_FILE_NAME)
            print("HMM model loaded.")
        except Exception as e:
            print(f"FATAL: Error loading HMM model: {e}")
            raise
            
    if scaler_global is None:
        print(f"Loading scaler from {SCALER_FILE_NAME}...")
        try:
            scaler_global = joblib.load(SCALER_FILE_NAME)
            print("Scaler loaded.")
        except Exception as e:
            print(f"FATAL: Error loading scaler: {e}")
            raise
    print("All globals checked/initialized.")


def get_latest_market_data_lambda(md_client, symbol, alpaca_tf, limit_bars):
    print(f"Fetching up to {limit_bars} bars of {alpaca_tf} data for {symbol}...")
    try:
        # For daily, you often want data up to *yesterday's* close to make decisions for *today*.
        # If BAR_TIMEFRAME is daily and you run this early morning, fetching up to "now" might
        # not yet include today's (incomplete) bar or might give an error if market isn't open.
        # It's often safer to explicitly define the end for daily data.
        if alpaca_tf == TimeFrame.Day:
            # Fetch data up to the end of the previous trading day relative to "now" in NY
            now_ny = datetime.now(tz=pytz.timezone('America/New_York'))
            # If it's before market close, "yesterday" is literally yesterday.
            # If it's after market close, "yesterday" is today (as today's bar has closed).
            # This logic can get complex with market hours. A simpler approach for daily:
            end_date_req = datetime.now() - timedelta(minutes=15) # Ensure we are looking for closed bars
        else: # For intraday, fetching up to "now" is usually fine.
            end_date_req = datetime.now()


        # Calculate start date robustly
        # This needs enough *calendar days* to cover `limit_bars` *trading bars*.
        # For daily, limit_bars * 1.8 is a rough estimate. Add buffer.
        if alpaca_tf == TimeFrame.Day:
            calendar_days_to_go_back = max(int(limit_bars * 1.8), limit_bars + 45) # More buffer for daily
        else: # Intraday
            # Estimate how many days back for intraday (very rough)
            trading_hours_per_day = 6.5
            bars_per_hour = 60 / alpaca_tf.value if alpaca_tf.unit == TimeFrame.Minute else 1 # Assuming alpaca_tf.value is minutes
            bars_per_day = trading_hours_per_day * bars_per_hour
            calendar_days_to_go_back = max(int(limit_bars / bars_per_day) + 5, 10) # Add buffer days

        start_date_req = datetime.now() - timedelta(days=calendar_days_to_go_back)
        
        # Ensure start_date_req is not too far in the past if limit_bars is small for intraday
        # Max lookback typically a few days for intraday via this type of query.
        # Alpaca's API might have its own limits on query range.

        print(f"Data request range: Start={start_date_req.strftime('%Y-%m-%d')}, End={end_date_req.strftime('%Y-%m-%d')}")

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=alpaca_tf,
            start=start_date_req, # Pass datetime objects
            end=end_date_req      # Pass datetime objects
        )
        bars_data = md_client.get_stock_bars(request_params)
        if not bars_data or symbol not in bars_data.data:
            print(f"No bars data returned for {symbol}.")
            return pd.DataFrame()

        bars_df = pd.DataFrame([bar.dict() for bar in bars_data.data[symbol]]) # Convert Bar objects to dicts
        if bars_df.empty:
            print(f"Fetched DataFrame is empty for {symbol}.")
            return pd.DataFrame()

        bars_df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume', 'timestamp':'Timestamp'}, inplace=True)
        bars_df['Timestamp'] = pd.to_datetime(bars_df['Timestamp'])
        bars_df.set_index('Timestamp', inplace=True)
        
        # Optional: Filter for US trading hours if intraday to remove pre/post market
        if alpaca_tf.unit == TimeFrame.Minute or alpaca_tf.unit == TimeFrame.Hour:
             bars_df = bars_df.tz_convert('America/New_York') # Ensure correct timezone for between_time
             bars_df = bars_df.between_time('09:30', '16:00')
             bars_df = bars_df.tz_convert('UTC') # Convert back to UTC if needed, or keep NY

        bars_df = bars_df[~bars_df.index.duplicated(keep='last')].sort_index()

        if len(bars_df) >= limit_bars:
            print(f"Successfully fetched {len(bars_df)} bars for {symbol}. Using the latest {limit_bars}.")
            return bars_df.iloc[-limit_bars:]
        else:
            print(f"Warning: Fetched {len(bars_df)} bars for {symbol}, less than required {limit_bars}.")
            return bars_df # Return what was fetched, subsequent checks handle insufficient data
            
    except Exception as e:
        print(f"Error in get_latest_market_data_lambda for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def calculate_hmm_features_lambda(df_full_history, hmm_features_list):
    df = df_full_history.copy()
    df['Returns_t_calc'] = df['Close'].pct_change()
    df['Returns_t_1_calc'] = df['Returns_t_calc'].shift(1)
    df['Returns_t_2_calc'] = df['Returns_t_calc'].shift(2)
    df['RSI_calc'] = ta.rsi(df['Close'], length=14)
    macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD_Signal_calc'] = macd_data['MACDs_12_26_9'] if macd_data is not None and not macd_data.empty else np.nan
    df['Volatility_calc'] = df['Returns_t_calc'].rolling(window=20).std()

    hmm_df = pd.DataFrame(index=df.index)
    # Map internal calculation names to the names expected by the HMM FEATURES list
    # This mapping ensures flexibility if your FEATURES list uses slightly different names
    # than your direct calculation outputs.
    feature_component_map = {
        'Returns_t': 'Returns_t_calc',
        'Returns_t_1': 'Returns_t_1_calc',
        'Returns_t_2': 'Returns_t_2_calc',
        'RSI': 'RSI_calc',
        'MACD_Signal': 'MACD_Signal_calc',
        'Volatility': 'Volatility_calc'
    }
    for feature_name_expected_by_hmm in hmm_features_list:
        calc_col_name = feature_component_map.get(feature_name_expected_by_hmm)
        if calc_col_name and calc_col_name in df.columns:
            hmm_df[feature_name_expected_by_hmm] = df[calc_col_name]
        else:
            print(f"Warning: Component for HMM feature '{feature_name_expected_by_hmm}' not found or mapping missing.")
            hmm_df[feature_name_expected_by_hmm] = np.nan # Ensure column exists

    latest_hmm_features = hmm_df[hmm_features_list].iloc[-1:].copy()
    if latest_hmm_features.isnull().values.any():
        print(f"NaNs in final HMM features for {df.index[-1] if not df.empty else 'N/A'}: {latest_hmm_features[latest_hmm_features.isnull().any(axis=1)]}")
        return pd.DataFrame()
    return latest_hmm_features

def predict_hmm_state_lambda(scaler_obj, model_obj, features_df):
    if features_df.empty: return None
    try:
        scaled_features = scaler_obj.transform(features_df)
        return model_obj.predict(scaled_features)[0]
    except Exception as e:
        print(f"Error predicting HMM state: {e}\nFeatures:\n{features_df}")
        return None

def get_current_price_and_account_equity_lambda(trade_client, symbol_str):
    latest_price, account_equity = None, None
    try:
        latest_trade = trade_client.get_latest_trade(symbol_str)
        latest_price = float(latest_trade.p)
    except Exception as e: print(f"Error getting latest price: {e}")
    try:
        account_info = trade_client.get_account()
        account_equity = float(account_info.equity)
    except Exception as e: print(f"Error getting account equity: {e}")
    return latest_price, account_equity

def execute_trade_strategy_lambda(trade_client_obj, current_hmm_state_val, symbol_str, 
                                  rsi_for_strat, atr_for_strat, volatility_for_strat, 
                                  rules_dict):
    print(f"--- Executing Trade Strategy for HMM State: {current_hmm_state_val} ---")
    # Convert HMM state to float for dict lookup, then to string if dict keys are strings
    action = rules_dict.get(current_hmm_state_val, 'Hold') # Assuming HMM state is float and keys are float
    # If TRADING_RULES_HMM keys are strings like '3.0', convert:
    # action = rules_dict.get(str(float(current_hmm_state_val)), 'Hold')

    print(f"HMM State {current_hmm_state_val} -> Action: {action}")
    if action == 'Hold': print("Strategy: Hold."); return

    if pd.isna(rsi_for_strat) or pd.isna(atr_for_strat) or pd.isna(volatility_for_strat):
        print(f"Strategy indicators NaN: RSI={rsi_for_strat}, ATR={atr_for_strat}, Vol={volatility_for_strat}. Skipping.")
        return
        
    price, cash = get_current_price_and_account_equity_lambda(trade_client_obj, symbol_str)
    if price is None or cash is None: print("No price/cash. Skipping."); return
    print(f"Current Price: {price}, Account Equity: {cash:.2f}")

    OVERBOUGHT, OVERSOLD, MAX_RISK_PCT = 80, 20, 0.02 # From your example
    dollar_risk = MAX_RISK_PCT * cash
    sl_dist, tp_dist = 1.5 * atr_for_strat, 3.0 * atr_for_strat

    if atr_for_strat <= 0 or sl_dist <= 0 : print(f"Invalid ATR/SL dist. ATR={atr_for_strat}. Skipping."); return
    pos_size_calc = dollar_risk / sl_dist
    if pos_size_calc < 1 : print(f"Pos size < 1 ({pos_size_calc:.2f}). Skipping."); return
    position_size = int(pos_size_calc)
    print(f"Dollar Risk: ${dollar_risk:.2f}, SL: ${sl_dist:.2f}, TP: ${tp_dist:.2f}, Size: {position_size}")

    # Volatility filter (ensure volatility_for_strat and atr_for_strat are comparable scales)
    if volatility_for_strat > (2 * atr_for_strat): # Review this condition
        print(f"High vol skip: Vol={volatility_for_strat:.4f}, 2*ATR={2*atr_for_strat:.4f}"); return

    curr_qty, is_long, is_short = 0.0, False, False
    try:
        pos = trade_client_obj.get_position(symbol_str)
        curr_qty, is_long, is_short = float(pos.qty), float(pos.qty) > 0, float(pos.qty) < 0
    except tradeapi.rest.APIError as e:
        if e.status_code != 404: print(f"Get pos error: {e}"); return
    print(f"Current pos: {curr_qty} shares.")

    order_details = {'symbol': symbol_str, 'qty': position_size, 'type': 'market', 'time_in_force': 'day', 'order_class': 'oto'}
    trade_action_taken = False

    if action == 'Buy' and not is_long and rsi_for_strat < OVERBOUGHT:
        if is_short:
            try: trade_client_obj.close_position(symbol_str); print("Closed short."); time.sleep(1)
            except Exception as ec: print(f"Close short fail: {ec}"); return
        order_details.update({'side': 'buy', 'stop_loss': {'stop_price': round(price - sl_dist, 2)}, 'take_profit': {'limit_price': round(price + tp_dist, 2)}})
        trade_action_taken = True
    elif action == 'Sell' and not is_short and rsi_for_strat > OVERSOLD:
        if is_long:
            try: trade_client_obj.close_position(symbol_str); print("Closed long."); time.sleep(1)
            except Exception as ec: print(f"Close long fail: {ec}"); return
        order_details.update({'side': 'sell', 'stop_loss': {'stop_price': round(price + sl_dist, 2)}, 'take_profit': {'limit_price': round(price - tp_dist, 2)}})
        trade_action_taken = True
    else:
        print(f"Conditions not met for {action} or already in position. RSI: {rsi_for_strat:.1f}")

    if trade_action_taken:
        try:
            trade_client_obj.submit_order(**order_details)
            print(f"{order_details['side'].upper()} order for {position_size} submitted.")
        except Exception as e_ord: print(f"{order_details['side'].upper()} order failed: {e_ord}")
    print("--- Trade Strategy Execution Complete ---")


# --- Lambda Handler Function ---
def lambda_handler(event, context):
    # `event` and `context` are passed by AWS Lambda. `event` contains trigger data.
    print(f"Lambda Event: {event}") # Log the event (e.g., from EventBridge)
    
    # Initialize clients and models if this is a cold start or they haven't been set
    # This helps reuse connections/loaded models across invocations if the container is warm
    try:
        initialize_globals()
    except Exception as e_init:
        print(f"CRITICAL: Failed to initialize globals: {e_init}")
        # Depending on the error, you might want to return an error status
        return {'statusCode': 500, 'body': f"Initialization failed: {str(e_init)}"}

    print(f"\n--- Running Bot Cycle via Lambda: {datetime.now(tz=pytz.timezone('America/New_York'))} NY ---") # Log with timezone
    
    # 1. Determine Data Fetch Limit
    # Ensure N_HISTORICAL_BARS_FOR_HMM is sufficient for all HMM feature lookbacks
    # Ensure MIN_BARS_FOR_STRAT_TA is sufficient for all strategy TA lookbacks
    data_fetch_limit = max(N_HISTORICAL_BARS_FOR_HMM, MIN_BARS_FOR_STRAT_TA)
    print(f"Determined data fetch limit: {data_fetch_limit} bars (plus buffer).")
    
    # Fetch slightly more for TA stability at the edges
    market_data_df_full = get_latest_market_data_lambda(market_data_client, SYMBOL, ALPACA_TIMEFRAME, data_fetch_limit + 40)

    if market_data_df_full.empty or len(market_data_df_full) < data_fetch_limit:
        message = f"Insufficient market data ({len(market_data_df_full)} fetched, need at least {data_fetch_limit}). Skipping."
        print(message)
        return {'statusCode': 200, 'body': message} # Graceful exit

    # 2. Calculate HMM Features
    # Pass the slice of data that HMM feature calculation expects
    if len(market_data_df_full) < N_HISTORICAL_BARS_FOR_HMM:
        message = f"Not enough data in market_data_df_full ({len(market_data_df_full)}) for HMM features ({N_HISTORICAL_BARS_FOR_HMM})."
        print(message)
        return {'statusCode': 200, 'body': message}
    hmm_input_data_slice = market_data_df_full.iloc[-N_HISTORICAL_BARS_FOR_HMM:].copy()
    features_for_hmm_prediction = calculate_hmm_features_lambda(hmm_input_data_slice, FEATURES)

    if features_for_hmm_prediction.empty:
        message = "Could not calculate HMM features. Skipping."
        print(message)
        return {'statusCode': 200, 'body': message}

    # 3. Predict HMM State
    predicted_hmm_state = predict_hmm_state_lambda(scaler_global, hmm_model_global, features_for_hmm_prediction)
    if predicted_hmm_state is None:
        message = "Could not predict HMM state. Skipping."
        print(message)
        return {'statusCode': 200, 'body': message}
    
    print(f"Predicted HMM State: {predicted_hmm_state}")

    # 4. Calculate Strategy-Specific Technical Indicators from the full fetched data
    df_for_strat_ta = market_data_df_full.copy()
    # Ensure columns exist before calculating TA
    required_cols_for_ta = ['High', 'Low', 'Close']
    if not all(col in df_for_strat_ta.columns for col in required_cols_for_ta):
        message = f"Missing required columns for TA ({required_cols_for_ta}) in fetched data. Columns: {df_for_strat_ta.columns}"
        print(message)
        return {'statusCode': 200, 'body': message}


    current_atr_value = ta.atr(df_for_strat_ta['High'], df_for_strat_ta['Low'], df_for_strat_ta['Close'], length=LOOKBACK_ATR_STRAT).iloc[-1]
    current_rsi_strat_value = ta.rsi(df_for_strat_ta['Close'], length=LOOKBACK_RSI_STRAT).iloc[-1]
    
    df_for_strat_ta['Returns_strat_calc'] = df_for_strat_ta['Close'].pct_change()
    current_volatility_strat_value = df_for_strat_ta['Returns_strat_calc'].rolling(window=LOOKBACK_VOL_STRAT).std().iloc[-1]

    print(f"Strategy Indicators for latest bar: ATR={current_atr_value:.2f}, RSI={current_rsi_strat_value:.2f}, Volatility={current_volatility_strat_value:.4f}")

    # 5. Execute Strategy if Market is Open
    is_market_open = False
    try:
        is_market_open = trading_client.get_clock().is_open
    except Exception as e_clock:
        print(f"Error getting market clock: {e_clock}. Assuming market is closed for safety.")
        
    if not is_market_open:
        message = f"Market is currently closed. Predicted HMM state {predicted_hmm_state}, but no trading actions taken."
        print(message)
    else:
        print("Market is open. Proceeding with trade strategy execution.")
        execute_trade_strategy_lambda(
            trading_client, predicted_hmm_state, SYMBOL,
            current_rsi_strat_value, current_atr_value, current_volatility_strat_value,
            TRADING_RULES_HMM
        )
        message = f"Trade strategy executed for state {predicted_hmm_state}."
    
    final_message = f"--- Lambda Bot Cycle Complete. {message} ---"
    print(final_message)
    return {'statusCode': 200, 'body': final_message}

# For local testing (optional, ensure .env is set up and files are in path)
# if __name__ == "__main__":
#     # Need to mock AWS context or ensure local env vars are set
#     # Python-dotenv should handle .env file for local testing if present
#     if not all([API_KEY, API_SECRET, BASE_URL]):
#         print("API Keys or Base URL not found in environment variables. Exiting local test.")
#     else:
#         print("