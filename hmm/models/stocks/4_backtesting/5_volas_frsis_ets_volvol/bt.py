import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import pandas_ta as ta
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, save

class HMMStateStrategy(Strategy):
    def init(self):
        # Define trading rules based on state analysis
        self.trading_rules = {
            0.0: 'Buy',    # Strong Bull (Sharpe 0.40)
            1.0: 'B',   # Bull (Sharpe 0.04 - too weak)
            3.0: 'Hold',   # Neutral (Sharpe 0.21)
            2.0: 'S',   # Bear (Sharpe 0.13)
            4.0: 'Sell'    # Strong Bear (Sharpe -0.52)
        }
        self.rsi = self.data.rsi


    def next(self):
        # Retrieve the current state from the data
        current_state = self.data.State[-1]
        current_rsi = self.rsi[-1]  # Get the most recent RSI value
        
        # Get trading action for the current state
        action = self.trading_rules.get(current_state, 'Hold')

        # Execute trading logic based on the action determined by state
        self.execute_trade(action, current_rsi)

    def execute_trade(self, action, rsi):
        """
        Executes trades with:
        - Volatility-adjusted position sizing
        - RSI threshold confirmation
        - Bracket orders for risk management
        - Position conflict checks
        """
        # Dynamic position sizing (capped at 5% of portfolio)
        
        # Define RSI thresholds (adjustable)
        OVERBOUGHT = 70
        OVERSOLD = 30
        price = self.data.Close[-1]
        cash = self.equity
        position_size = int(cash/price)
        if action == 'Buy' and not self.position.is_long:
            # Additional RSI filter - only buy when not overbought
            if rsi < OVERBOUGHT:
                if self.position.is_short:
                    self.position.close()  
                
                try:
                    self.buy(
                        size=position_size,
                        sl=self.data.Close[-1] * 0.95,  # 5% stop-loss
                        tp=self.data.Close[-1] * 1.10,  # 10% take-profit
                    )
                except Exception as e:
                    print(f"Buy order failed: {e} (RSI: {rsi:.1f})")

        elif action == 'Sell' and not self.position.is_short:
            # Additional RSI filter - only sell when not oversold
            if rsi > OVERSOLD:
                if self.position.is_long:
                    self.position.close()  
                
                try:
                    self.sell(
                        size=position_size,
                        sl=self.data.Close[-1] * 1.05,  # 5% stop-loss
                        tp=self.data.Close[-1] * 0.90,  # 10% take-profit
                    )
                except Exception as e:
                    print(f"Sell order failed: {e} (RSI: {rsi:.1f})")


def preprocess_data(data_path, state_path):
    """Load and preprocess S&P 500 data (last 10 years only)"""
    # Load market data
    df = pd.read_csv(data_path, skiprows=2, header=None)

    # Set headers manually
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df.iloc[1:] # Remove the potentially problematic first row after header assignment

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)    
    # Filter for last 10 years
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=10)
    df = df[df.index >= cutoff_date]
    
    # Calculate returns and volatility
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=21).std()  # 21 trading days = 1 month
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df.dropna(inplace=True)

    # Load state data and filter for same date range
    state_df = pd.read_csv(
        state_path,
        parse_dates=['Start Time', 'End Time'],
        skipinitialspace=True
    )
    state_df = state_df[state_df['End Time'] >= cutoff_date]
    
    # Map states to market data
    df['State'] = np.nan
    for _, row in state_df.iterrows():
        mask = (df.index >= row['Start Time']) & (df.index <= row['End Time'])
        df.loc[mask, 'State'] = row['State']

    # Forward-fill missing states (optional)
    df['State'] = df['State'].ffill()
    
    return df.dropna(subset=['State', 'returns'])

def run_backtest(data, margin=1.0):
    bt = Backtest(data, HMMStateStrategy, cash=10000, commission=0, margin=margin)
    stats = bt.run()
    return bt, stats

def save_results(bt, stats, output_dir='backtest_results'):
    """
    Save backtest results with:
    1. Formatted performance metrics CSV
    2. Equity curve plot PNG
    """
    import os
    from datetime import datetime
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save performance metrics to CSV
    csv_path = os.path.join(output_dir, f'performance_metrics.csv')
    
    metrics = {
        'Metric': [
            'Start', 'End', 'Duration', 'Exposure Time [%]',
            'Equity Final [$]', 'Equity Peak [$]', 'Return [%]',
            'Buy & Hold Return [%]', 'Return (Ann.) [%]',
            'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Sortino Ratio',
            'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]',
            'Max. Drawdown Duration', 'Total Trades', 'Win Rate [%]',
            'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
            'Profit Factor', 'Expectancy [%]'
        ],
        'Value': [
            stats._equity_curve.index[0],
            stats._equity_curve.index[-1],
            stats._equity_curve.index[-1] - stats._equity_curve.index[0],
            f"{stats['Exposure Time [%]']:.2f}",
            f"{stats['Equity Final [$]']:.2f}",
            f"{stats['Equity Peak [$]']:.2f}",
            f"{stats['Return [%]']:.2f}",
            f"{stats['Buy & Hold Return [%]']:.2f}",
            f"{stats['Return (Ann.) [%]']:.2f}",
            f"{stats['Volatility (Ann.) [%]']:.2f}",
            f"{stats['Sharpe Ratio']:.2f}",
            f"{stats['Sortino Ratio']:.2f}",
            f"{stats['Calmar Ratio']:.2f}",
            f"{stats['Max. Drawdown [%]']:.2f}",
            f"{stats['Avg. Drawdown [%]']:.2f}",
            stats['Max. Drawdown Duration'],
            stats['# Trades'],
            f"{stats['Win Rate [%]']:.2f}",
            f"{stats['Best Trade [%]']:.2f}",
            f"{stats['Worst Trade [%]']:.2f}",
            f"{stats['Avg. Trade [%]']:.2f}",
            f"{stats['Profit Factor']:.2f}",
            f"{stats['Expectancy [%]']:.2f}"
        ]
    }
    
    pd.DataFrame(metrics).to_csv(csv_path, index=False)
    print(f"Saved performance metrics to: {csv_path}")

    # 2. Save equity curve plot
    plot_path = os.path.join(output_dir, f'equity_curve.html')

    try:
        bokeh_fig = bt.plot()  # This returns a Bokeh GridPlot
        output_file(plot_path)
        save(bokeh_fig)
        print(f"Saved equity curve plot to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    # Replace the paths with your actual files
    DATA_PATH = 'data/sp500_daily_all.csv'
    STATE_PATH = 'hmm/models/stocks/2_selected_models/5_volas_frsis_ets_volvol/sp500_state_changes.csv'  # The CSV file with your state data

    # Preprocess data and add predicted states
    data = preprocess_data(DATA_PATH, STATE_PATH)

    # Run the backtest
    bt, stats = run_backtest(data)
    save_results(bt, stats)