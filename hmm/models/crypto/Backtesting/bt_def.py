import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta

class HMMStateStrategy(Strategy):
    def init(self):
        # Define trading rules based on state analysis
        self.trading_rules = {
            4.0: 'Buy',   # Strong Bull
            6.0: 'Buy',   # Bull
            2.0: 'Hold',  # Neutral
            1.0: 'Sell',  # Bear
            0.0: 'Sell'   # Strong Bear
        }

    def next(self):
        # Retrieve the current state from the data
        current_state = self.data.State[-1]
        
        # Get trading action for the current state
        action = self.trading_rules.get(current_state, 'Hold')

        # Execute trading logic based on the action determined by state
        if action == 'Buy' and not self.position.is_long:
            if self.position.is_short:
                self.position.close()  # Close any short positions first
            try:
               # Adjust size to a smaller proportion, and validate position logic
                self.buy()
            except Exception as e:
                print(f"Buy order failed: {e}")
        elif action == 'Sell' and not self.position.is_short:
            if self.position.is_long:
                self.position.close()  # Close any long positions first
            try:
                self.sell()
            except Exception as e:
                print(f"Sell order failed: {e}")
        #if action == 'Buy' and not self.position.is_long:
        #    try:
        #        self.buy()  # Adjust the size as needed
        #    except Exception as e:
        #        print(f"Buy order failed: {e}")
        #elif action == 'Sell' and self.position.is_long:
        #    try:
        #        self.position.close()  # Close long positions
        #    except Exception as e:
        #        print(f"Sell order failed: {e}")

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

def run_backtest(data,  commission=0.0008, margin=1.0):
    bt = Backtest(data, HMMStateStrategy, cash=1000000, commission=commission, margin=margin)
    stats = bt.run()
    return bt, stats

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