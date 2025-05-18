from backtesting import Strategy, Backtest
from backtesting.test import SMA
import numpy as np
import pandas_ta as ta
import pandas as pd

class HMMStateStrategy(Strategy):
    def init(self):
        # Trading rules
        self.trading_rules = {
            4.0: 1.0,   # Strong Bull
            6.0: 0.6,   # Bull
            2.0: 0,     # Neutral
            1.0: -0.6,  # Bear
            0.0: -1.0   # Strong Bear
        }

        # Initialize indicators
        self.ma200 = self.I(SMA, self.data.Close, 200)
        
        # Convert backtesting._Array to numpy arrays
        close_prices = np.array(self.data.Close)
        high_prices = np.array(self.data.High)
        low_prices = np.array(self.data.Low)
        
        # Calculate indicators using pandas_ta
        self.rsi_values = self._calculate_rsi(close_prices)
        self.atr_values = self._calculate_atr(high_prices, low_prices, close_prices)
        
        # Verify lengths
        assert len(self.rsi_values) == len(close_prices), "RSI length mismatch"
        assert len(self.atr_values) == len(close_prices), "ATR length mismatch"

    def _calculate_rsi(self, close):
        """Calculate RSI using pandas_ta"""
        rsi = ta.rsi(pd.Series(close), length=14)
        return np.nan_to_num(rsi.values, nan=50)  # Fill NaN with neutral 50

    def _calculate_atr(self, high, low, close):
        """Calculate ATR using pandas_ta"""
        atr = ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=14)
        return np.nan_to_num(atr.values, nan=np.nanmean(atr.values))

    def next(self):
        # Current index
        i = len(self.data) - 1
        
        # Skip if insufficient data
        if i < 200 or np.isnan(self.data.State[-1]):
            return

        current_state = self.data.State[-1]
        action = self.trading_rules.get(current_state, 0)
        
        # Get current values
        rsi = self.rsi_values[i]
        atr = self.atr_values[i]
        ma200 = self.ma200[-1]
        
        # Market regime filter
        bull_market = self.data.Close[-1] > ma200
        
        # Entry conditions
        if action > 0 and bull_market and rsi < 65:
            if not self.position.is_long:
                position_size = min(
                    self.base_position_size,
                    self.max_risk/(2.5*atr/self.data.Close[-1])
                )
                self.buy(size=position_size)
                
        elif action < 0 and not bull_market and rsi > 35:
            if not self.position.is_short:
                position_size = min(
                    self.base_position_size,
                    self.max_risk/(2.5*atr/self.data.Close[-1])
                )
                self.sell(size=position_size)

def run_backtest(data):
    # Verify required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'State']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    bt = Backtest(data, HMMStateStrategy, cash=1000000, commission=0.0008)
    stats = bt.run()
    return bt, stats

def preprocess_data(data_path, state_path):
    # Load market data
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=24).std()
    df.dropna(inplace=True)

    # Load state data
    state_df = pd.read_csv(state_path,
        parse_dates=['Start Time', 'End Time'],  # Now matches the stripped names
        skipinitialspace=True  # This removes leading spaces from column names
    )
    # Add state information to market data
    df['State'] = np.nan
    for _, row in state_df.iterrows():
        mask = (df.index >= row['Start Time']) & (df.index <= row['End Time'])
        df.loc[mask, 'State'] = row['State']

    return df


def analyze_results(stats):
    print("\n=== Detailed Performance Metrics ===")
    print(f"Start: {stats._equity_curve.index[0]}")
    print(f"End: {stats._equity_curve.index[-1]}")
    print(f"Duration: {stats._equity_curve.index[-1] - stats._equity_curve.index[0]}")
    print(f"Exposure Time [%]: {stats['Exposure Time [%]']:.2f}")
    print(f"Equity Final [$]: {stats['Equity Final [$]']:.2f}")
    print(f"Equity Peak [$]: {stats['Equity Peak [$]']:.2f}")
    print(f"Return [%]: {stats['Return [%]']:.2f}")
    print(f"Buy & Hold Return [%]: {stats['Buy & Hold Return [%]']:.2f}")
    print(f"Return (Ann.) [%]: {stats['Return (Ann.) [%]']:.2f}")
    print(f"Volatility (Ann.) [%]: {stats['Volatility (Ann.) [%]']:.2f}")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio: {stats['Sortino Ratio']:.2f}")
    print(f"Calmar Ratio: {stats['Calmar Ratio']:.2f}")
    print(f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
    print(f"Avg. Drawdown [%]: {stats['Avg. Drawdown [%]']:.2f}")
    print(f"Max. Drawdown Duration: {stats['Max. Drawdown Duration']}")
    print(f"Total Trades: {stats['# Trades']}")
    print(f"Win Rate [%]: {stats['Win Rate [%]']:.2f}")
    print(f"Best Trade [%]: {stats['Best Trade [%]']:.2f}")
    print(f"Worst Trade [%]: {stats['Worst Trade [%]']:.2f}")
    print(f"Avg. Trade [%]: {stats['Avg. Trade [%]']:.2f}")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Expectancy [%]: {stats['Expectancy [%]']:.2f}")

if __name__ == "__main__":
    # Replace the paths with your actual files
    DATA_PATH = 'data/btc_1h_3years.csv'
    STATE_PATH = 'hmm/7_regimes/volar_frsi_ts_volz/bitcoin_state_changes.csv'  # The CSV file with your state data

    # Preprocess data and add predicted states
    data = preprocess_data(DATA_PATH, STATE_PATH)

    # Run the backtest
    bt, stats = run_backtest(data)

    # Analyze results
    analyze_results(stats)
    bt.plot()