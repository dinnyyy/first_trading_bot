# lambda_function.py
import sys
sys.path.append('/var/task/package')
import os
import time
import joblib
import pandas as pd
import numpy as np
# Then use nan in your code (without np prefix)
import pandas_ta as ta
from datetime import datetime, timedelta
import pytz

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# --- Load Environment Variables ---
API_KEY = os.environ.get('APCA_API_KEY_ID')
API_SECRET = os.environ.get('APCA_API_SECRET_KEY')
BASE_URL = os.environ.get('APCA_PAPER_URL')

# --- CONFIGURATION ---
SYMBOL = os.environ.get('SYMBOL', 'SPY')
N_HISTORICAL_BARS_FOR_HMM = int(os.environ.get('N_HISTORICAL_BARS_FOR_HMM', 34))
LOOKBACK_RSI_STRAT = 14
LOOKBACK_ATR_STRAT = 14
LOOKBACK_VOL_STRAT = 20
MIN_BARS_FOR_STRAT_TA = max(LOOKBACK_RSI_STRAT, LOOKBACK_ATR_STRAT, LOOKBACK_VOL_STRAT) + 1

BAR_TIMEFRAME_STR = os.environ.get('BAR_TIMEFRAME', '1Day')
if BAR_TIMEFRAME_STR == '1Day':
    ALPACA_TIMEFRAME = TimeFrame.Day
elif BAR_TIMEFRAME_STR == '1Hour':
    ALPACA_TIMEFRAME = TimeFrame.Hour
elif 'Min' in BAR_TIMEFRAME_STR:
    minutes = int(BAR_TIMEFRAME_STR.replace('Min', ''))
    ALPACA_TIMEFRAME = TimeFrame.Minute if minutes == 1 else TimeFrame.Minute(minutes)
else:
    ALPACA_TIMEFRAME = TimeFrame.Day

MODEL_FILE_NAME = 'model.joblib'
SCALER_FILE_NAME = 'scaler.joblib'

FEATURES = ['Returns_t', 'Returns_t_1', 'Returns_t_2', 'RSI', 'MACD_Signal', 'Volatility']
TRADING_RULES_HMM = {3.0: 'Sell', 0.0: 'Sell', 1.0: 'Hold', 2.0: 'Buy', 4.0: 'Buy'}

# --- Globals ---
trading_client = None
market_data_client = None
hmm_model_global = None
scaler_global = None

def initialize_globals():
    global trading_client, market_data_client, hmm_model_global, scaler_global

    if trading_client is None:
        print("Initializing Alpaca TradingClient...")
        try:
            trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
            account = trading_client.get_account()
            print(f"Trading client initialized. Account equity: ${account.equity}")
        except Exception as e:
            print(f"FATAL: Error initializing trading client: {e}")
            raise

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

    print("All globals initialized successfully.")

def get_latest_market_data_lambda(symbol, timeframe, limit):
    """Fetches the latest market data bars from Alpaca."""
    try:
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=limit)

        # Format dates as strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Create a request for stock bars (price data)
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date_str,
            end=end_date_str
        )

        # Fetch the data
        bars = market_data_client.get_stock_bars(request_params)
        cleaned_data = []
        for row in bars['SPY']:
            cleaned_row = {k: v for k, v in row}  # This unpacks the tuples
            cleaned_data.append(cleaned_row)

        bars_df = pd.DataFrame(cleaned_data)

        # Display the cleaned DataFrame
        # Convert the list of dictionaries to a pandas DataFrame
        bars_df = bars_df[['open', 'high', 'low', 'close', 'volume']]
        bars_df.index = pd.to_datetime(bars_df.index) # Ensure index is datetime
        bars_df = bars_df[~bars_df.index.duplicated(keep='last')] # Remove duplicates if any
        bars_df = bars_df.sort_index()
        bars_df = bars_df.rename(columns={
                        'open': 'Open',
                        'high': 'High', 
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })

        if len(bars_df) >= limit:
            return bars_df.iloc[-limit:] # Return the most recent 'limit' bars
        else:
            print(f"Warning: Fetched {len(bars_df)} bars, less than required {limit}.")
            return bars_df # Or handle error if not enough data

    except Exception as e:
        print(f"Error fetching market data for {symbol}: {e}")
        return pd.DataFrame()


def calculate_hmm_features_lambda(df_full_history, hmm_features_list):
    df = df_full_history.copy()
    df['Returns_t_calc'] = df['Close'].pct_change()
    df['Returns_t_1_calc'] = df['Returns_t_calc'].shift(1)
    df['Returns_t_2_calc'] = df['Returns_t_calc'].shift(2)
    df['RSI_calc'] = ta.rsi(df['Close'], length=14)
    macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD_Signal_calc'] = macd_data['MACDs_12_26_9'] if macd_data is not None else np.nan
    df['Volatility_calc'] = df['Returns_t_calc'].rolling(window=20).std()

    feature_map = {
        'Returns_t': 'Returns_t_calc',
        'Returns_t_1': 'Returns_t_1_calc',
        'Returns_t_2': 'Returns_t_2_calc',
        'RSI': 'RSI_calc',
        'MACD_Signal': 'MACD_Signal_calc',
        'Volatility': 'Volatility_calc'
    }

    hmm_df = pd.DataFrame(index=df.index)
    for feat in hmm_features_list:
        hmm_df[feat] = df[feature_map.get(feat, '')]

    latest = hmm_df.iloc[-1:]
    if latest.isnull().values.any():
        print("NaNs found in final HMM features:", latest)
        return pd.DataFrame()
    return latest


def predict_hmm_state_lambda(scaler_obj, model_obj, features_df):
    if features_df.empty:
        return None
    try:
        scaled = scaler_obj.transform(features_df)
        return model_obj.predict(scaled)[0]
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None


def get_current_price_and_account_equity_lambda(trade_client, symbol):
    try:
        request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        latest_price = StockHistoricalDataClient(API_KEY, API_SECRET).get_stock_latest_quote(request_params)[symbol].ask_price
        price = float(latest_price)
    except Exception as e:
        print(f"Failed to get latest price: {e}")
        price = None

    try:
        account = trade_client.get_account()
        equity = float(account.equity)
    except Exception as e:
        print(f"Failed to get account equity: {e}")
        equity = None

    return price, equity


def execute_trade_strategy_lambda(trade_client, hmm_state, symbol, 
                                  rsi_val, atr_val, vol_val,
                                  hmm_rules):

    print(f"\n--- Executing Trade Strategy for HMM State: {hmm_state} ---")

    action = hmm_rules.get(hmm_state, 'Hold')
    print(f"HMM State: {hmm_state} => Action: {action}")

    if action == 'Hold':
        print("Decision: Hold. No action taken.")
        return

    if any(pd.isna(v) for v in [rsi_val, atr_val, vol_val]):
        print(f"NaN values in indicators. RSI={rsi_val}, ATR={atr_val}, Vol={vol_val}")
        return

    price, equity = get_current_price_and_account_equity_lambda(trade_client, symbol)
    if price is None or equity is None:
        print("Price or account equity unavailable. Skipping trade.")
        return

    print(f"Price: {price}, Equity: {equity:.2f}")
    dollar_risk = equity * 0.02
    stop_loss_dist = 1.5 * atr_val
    take_profit_dist = 3.0 * atr_val

    if atr_val <= 0 or stop_loss_dist <= 0:
        print("Invalid ATR or SL distance.")
        return

    position_size = int(dollar_risk / stop_loss_dist)
    if position_size < 1:
        print("Calculated position size < 1. Skipping.")
        return

    print(f"Calculated position: {position_size} shares | Risk ${dollar_risk:.2f}, SL ${stop_loss_dist:.2f}, TP ${take_profit_dist:.2f}")

    if vol_val > (2 * atr_val):
        print(f"Volatility too high. Vol={vol_val:.4f}, 2*ATR={2 * atr_val:.4f}")
        return

    # Check if there is an open position
    open_position = None
    try:
        open_position = trade_client.get_open_position(symbol)
        qty = float(open_position.qty_available)
        print(f"Open position detected: {qty} shares")
    except Exception:
        print("No open position.")

    # Define order parameters
    side = OrderSide.BUY if action == 'Buy' else OrderSide.SELL
    
    # Calculate stop loss and take profit prices
    if side == OrderSide.BUY:
        stop_price = round(price - stop_loss_dist, 2)
        limit_price = round(price + take_profit_dist, 2)
    else:
        stop_price = round(price + stop_loss_dist, 2)
        limit_price = round(price - take_profit_dist, 2)
    
    # Flatten existing position (if needed)
    if open_position:
        try:
            print("Closing existing position...")
            trade_client.close_position(symbol)
            time.sleep(1)
        except Exception as e:
            print(f"Failed to close existing position: {e}")
            return

    # Submit order with bracket parameters
    try:
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=position_size,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss={"stop_price": stop_price},
            take_profit={"limit_price": limit_price}
        )
        
        submitted_order = trade_client.submit_order(order_req)
        print(f"Submitted {side.name} order: ID={submitted_order.id}")
    except Exception as e:
        print(f"Failed to submit order: {e}")
        print("Trying alternative order approach...")
        
        # Try a simple market order without OTO if the above fails
        try:
            simple_order = MarketOrderRequest(
                symbol=symbol,
                qty=position_size,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            submitted_order = trade_client.submit_order(simple_order)
            print(f"Submitted simple {side.name} order: ID={submitted_order.id}")
        except Exception as e2:
            print(f"Failed to submit simple order: {e2}")

    print("--- Trade Strategy Execution Complete ---")


def lambda_handler(event, context):
    print(f"Lambda Event: {event}")
    
    # --- Initialize Globals ---
    try:
        initialize_globals()
    except Exception as e_init:
        print(f"CRITICAL: Failed to initialize globals: {e_init}")
        return {'statusCode': 500, 'body': f"Initialization failed: {str(e_init)}"}

    now_ny = datetime.now(tz=pytz.timezone('America/New_York'))
    print(f"\n--- Running Bot Cycle: {now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

    # --- Determine How Much Data to Fetch ---
    data_fetch_limit = max(N_HISTORICAL_BARS_FOR_HMM, MIN_BARS_FOR_STRAT_TA)
    print(f"Data Fetch Limit: {data_fetch_limit} bars + buffer")

    df_full = get_latest_market_data_lambda(SYMBOL, ALPACA_TIMEFRAME, data_fetch_limit + 40)
    if df_full.empty or len(df_full) < data_fetch_limit:
        msg = f"Insufficient market data ({len(df_full)} bars). Skipping."
        print(msg)
        return {'statusCode': 200, 'body': msg}

    # --- HMM Features ---
    hmm_df_slice = df_full.iloc[-N_HISTORICAL_BARS_FOR_HMM:]
    hmm_features = calculate_hmm_features_lambda(hmm_df_slice, FEATURES)

    if hmm_features.empty:
        msg = "HMM features missing or contain NaNs. Skipping."
        print(msg)
        return {'statusCode': 200, 'body': msg}

    # --- Predict State ---
    hmm_state = predict_hmm_state_lambda(scaler_global, hmm_model_global, hmm_features)
    if hmm_state is None:
        msg = "HMM prediction failed. Skipping."
        print(msg)
        return {'statusCode': 200, 'body': msg}

    print(f"Predicted HMM State: {hmm_state}")

    # --- Technical Indicators for Strategy ---
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df_full.columns for col in required_cols):
        msg = f"Missing required TA columns. Available: {df_full.columns.tolist()}"
        print(msg)
        return {'statusCode': 200, 'body': msg}

    current_atr = ta.atr(df_full['High'], df_full['Low'], df_full['Close'], length=LOOKBACK_ATR_STRAT).iloc[-1]
    current_rsi = ta.rsi(df_full['Close'], length=LOOKBACK_RSI_STRAT).iloc[-1]
    df_full['Returns'] = df_full['Close'].pct_change()
    current_vol = df_full['Returns'].rolling(window=LOOKBACK_VOL_STRAT).std().iloc[-1]

    print(f"TA Indicators: RSI={current_rsi:.2f}, ATR={current_atr:.2f}, Vol={current_vol:.4f}")

    # --- Check Market Open ---
    try:
        clock = trading_client.get_clock()
        if not clock.is_open:
            msg = f"Market is closed. No trades executed. State: {hmm_state}"
            print(msg)
            return {'statusCode': 200, 'body': msg}
    except Exception as e:
        print(f"Failed to fetch market clock: {e}")
        return {'statusCode': 200, 'body': "Clock error. Market open status unknown. Skipping."}

    # --- Execute Strategy ---
    print("Market is open. Executing strategy...")
    execute_trade_strategy_lambda(
        trading_client,
        hmm_state,
        SYMBOL,
        current_rsi,
        current_atr,
        current_vol,
        TRADING_RULES_HMM
    )

    final_message = f"Trade cycle complete. HMM State: {hmm_state}"
    print(final_message)
    return {'statusCode': 200, 'body': final_message}
