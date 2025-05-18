import yfinance as yf

# Download maximum daily historical data for S&P 500
df = yf.download('^GSPC', interval='1d', period='max')

# Save to CSV
df.to_csv('sp500_daily_all.csv')
print(f"Saved {len(df)} rows to sp500_daily_all.csv")
