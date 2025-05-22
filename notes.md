
# Brief points
- stocks vs crypto? Would using sp500 be better or bitcoin. What is more common for bots.
- Bybit or binance for crypto
- Alpaca for stocks
- is hmm good for a trading bot? if not what alternatives. How can i expand on this model
- Trade forex currency pairs
    allows to use symmetric rules for long a short trades
- dont start model speculation with indicators
- use droplet on digital ocean for Linux vps (https://cloud.digitalocean.com/droplets?i=588327)

Basic Model to begin with:
1. Data Source
- Daily or hourly S&P 500 data (e.g., SPY ETF).
- Indicators: RSI, MACD, Moving Averages, ATR, etc.

2. Indicators as Features
- Lagged returns: r_t, r_{t-1}, r_{t-2}
- RSI
- MACD signal
- Volatility (rolling std or GARCH estimate)
- Volume (optional)
- Stochastic oscilator

3. Hidden Markov Model
- Use a GaussianHMM (e.g., 2 or 3 hidden states).
- Fit to returns and/or indicators.
- Use the inferred state as a regime label.

4. Trading Rule
- For example:
    State 0 (calm): Go long if RSI < 30 and MACD is positive
    State 1 (high vol): Avoid trading or tighten stops
    State 2 (trend): Ride trend if MA(5) > MA(20)

More advanced model:

1. Use GARCH for volatility prediction (forecast risk).
2. Use HMM to classify market regimes (e.g., quiet/trending/explosive).
3. Combine these as features in a machine learning model (e.g., logistic regression or XGBoost) that:
    - Takes in lagged returns, volatility, indicators, regime labels, etc.
    - Outputs: Probability of price going up, or a trading signal.
4. Build a simple rule-based bot from that:
    - Long if prob > 0.55 and vol is low
    - Short if prob < 0.45 and trend is weakening
    - No trade otherwise
5. Implement strong risk management

# Best models (so far)

Crypto
- volar_frsi_ts_volz 7 regime

Stocks
- volas_frsis_ets_volvol 5 regime
- retrunsl_rsi_macd_vola 5 regime

# Learning

Podcasts
- top traders unplugged
- flirting with models
- better systems trader
- jane street podcasts - signals

Youtube
- Partime Larry
 - Peak Algo Trading Full Course 2023 | Commodity Market Futures (video) -> https://www.youtube.com/watch?v=rrbahhoePN4
 - Algo Trading with Kevin Davey (channel) -> https://www.youtube.com/watch?v=pK1ukXMeDcw
 - Darwinex Algo Trading for a Living (playlist) -> https://www.youtube.com/watch?v=RqVqbqYyHpI&list=PLv-cA-4O3y96b0gk8x3yPbqVM2gqafZAw
 
People
- Jim Simons
