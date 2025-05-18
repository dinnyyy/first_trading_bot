import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

# Initialize exchange
exchange = ccxt.binance()

def download_historical_data(symbol, timeframe, since, limit):
    all_data = []
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # move to next batch
            print(f"Fetched up to: {datetime.utcfromtimestamp(ohlcv[-1][0] / 1000)}")
            time.sleep(exchange.rateLimit / 1000)  # respect rate limits
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.to_csv(f'btc_{timeframe}.csv', index=True)
    print(f"Saved {len(df)} records to {symbol}_{timeframe}.csv")
#     for column in ['Returns', 'Volatility', 'Volume_Change']:
# Parameters
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 1000  # Max candles per request
now = exchange.milliseconds()
five_years_ago = exchange.milliseconds() - int(3 * 365.25 * 24 * 60 * 60 * 1000)
since = five_years_ago
# Convert and save
download_historical_data(symbol, timeframe, since, limit)
