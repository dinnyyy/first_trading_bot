import pandas as pd
import numpy as np
import time 
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import joblib
import pandas_ta as ta
import time

def load_and_preprocess_data(file_path, years_of_data=10):
    print(f"Loading S&P 500 data from {file_path}...")

    df = pd.read_csv(file_path, skiprows=2, header=None)

    # Set headers manually
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df.iloc[1:] # Remove the potentially problematic first row after header assignment

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    print("Original DataFrame head after loading all data:")
    print(df.head())
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Original data range: {df.index.min()} to {df.index.max()}")

    # --- Filter for the last 'years_of_data' years ---
    if not df.empty and years_of_data is not None:
        most_recent_date = df.index.max()
        start_date_filter = most_recent_date - pd.DateOffset(years=years_of_data)
        
        df = df[df.index >= start_date_filter]
        print(f"\nFiltered DataFrame for the last {years_of_data} years.")
        print(f"Data now ranges from: {df.index.min()} to {df.index.max()}")
        print(f"Filtered DataFrame shape: {df.shape}")
    elif df.empty:
        print("Warning: DataFrame is empty after initial loading. Cannot filter by date.")
        return df # Return empty df if it's already empty
    # ----------------------------------------------------

    # Ensure 'Close', 'High', 'Low', 'Open', 'Volume' are numeric, coercing errors
    cols_to_convert = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in cols_to_convert:
        # Replace non-numeric strings like '-' or empty strings with NaN before conversion
        df[col] = df[col].replace({'-': np.nan, '': np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where key financial data might be NaN after conversion
    # Especially 'Close' is critical for many calculations
    df.dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)
    
    if df.empty:
        print("Warning: DataFrame is empty after NaN removal post numeric conversion. Check data quality.")
        return df

    print("Adjusted DataFrame head after filtering and numeric conversion:")
    print(df.head())
    
    
    print("Calculating daily returns and volatility...")
    # Daily returns with safeguards (using Close price)
    
    print("Calculating daily returns and volatility...")
    # Daily returns with safeguards (using Close price)
    df['Returns'] = df['Close'].pct_change()
    df['Returns'] = df['Returns'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility using 20-day rolling window
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    print("Calculating volume metrics...")
    # Volume change with safeguards
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], np.nan)
    df['Volatility_Scaled'] = np.log(df['Volatility'])  # More continuous representation
    # Volume z-score using 20-day window
    df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

    print("Calculating technical indicators...")
    # Bollinger Bands (20-day)
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
    
    # RSI (14-day)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # MACD (12/26/9)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']


    # ATR (14-day)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['MACD_Strength'] = df['MACD_Hist'] / df['ATR']  # Normalized by volatility

    # EMA Deviation (20-day)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['Price_EMA_Diff'] = (df['Close'] - df['EMA_20']) / df['EMA_20']

    # ADX (14-day)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['Trend_Strength'] = df['ADX']

    print("Creating derived features...")
    # Volatility regime (top quartile = high volatility)
    df['Vol_Regime'] = np.where(df['Volatility'] > df['Volatility'].quantile(0.75), 1, 0)
        
    # RSI Z-score
    df['RSI_Zscore'] = (df['RSI'] - 50) / 20
    
    # Fisher Transform of RSI
    df['Fisher_RSI'] = 0.5 * np.log((1 + df['RSI']/100)/(1 - df['RSI']/100))
    df['Fisher_RSI_Smoothed'] = df['Fisher_RSI'].rolling(5).mean()
    df['Fisher_RSI_Scaled'] = np.tanh(df['Fisher_RSI_Smoothed'])
    df['Trend_Direction'] = np.sign(df['Close'].diff(5))
    df['Enhanced_Trend'] = df['ADX'] * df['Trend_Direction']
    df['Vol_Volume'] = df['Volume_Zscore'] * df['Volatility_Scaled']
    df['Vol_Regime_Score'] = df['Volatility_Scaled'] * np.sign(df['Volatility'].pct_change(5))
    df['Enhanced_Trend_v2'] = df['ADX'] * (df['RSI_Zscore'].clip(-1, 1))

    
    # Handle extreme values
    df['Volume_Change'] = df['Volume_Change'].clip(-3, 3)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Dropping NaN values...")
    df.dropna(inplace=True)

    print(f"S&P 500 data preprocessed. Shape: {df.shape}")
    print("\nPreview of processed data:")
    print("Available columns in training data:")
    print(df.head())
    return df

def train_hmm(data, features, n_components=3):
    print(f'Training HMM with {n_components} components...')
    X = data[features].values

    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Fitting HMM model...")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X_scaled)

    print("HMM training completed.")
    return model, scaler, X_scaled


# Predict states
def predict_states(model, data, scaler, features):
    print("Predicting states...")
    X = data[features].values
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    print(f"States predicted. Unique states: {np.unique(states)}")
    return states

# Analyze states
def analyze_states(data, states, model, feature_names):
    print("Analyzing states...")
    df_analysis = data.copy()
    df_analysis['State'] = states

    for state in range(model.n_components):
        print(f"\nAnalyzing State {state}:")
        state_data = df_analysis[df_analysis['State'] == state]
        print(state_data[feature_names].describe())
        print(f"Number of periods in State {state}: {len(state_data)}")

def predict_next_state(model, current_state):
    return np.argmax(model.transmat_[current_state])

def save_state_changes(states, data, state_names, output_file):
    state_changes = []
    current_state = states[0]
    start_time = data.index[0]

    for i, state in enumerate(states[1:], 1):
        if state != current_state:
            state_changes.append((start_time, data.index[i-1], current_state))
            current_state = state
            start_time = data.index[i]

    state_changes.append((start_time, data.index[i-1], current_state))

    os.makedirs('data', exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Start Time, End Time, State, State Name\n")
        for start, end, state in state_changes:
            f.write(f"{start},{end},{state},{state_names[state]})\n")

    print(f"State changes have been saved to {output_file}")

def plot_results(data, states, model):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot price with state regions
    ax1.plot(data.index, data['Close'])
    ax1.set_title('SP500 Price and HMM States')
    ax1.set_ylabel('Price')

    for state in range(model.n_components):
        mask = (states == state)
        ax1.fill_between(data.index, 
                        data['Close'].min(), 
                        data['Close'].max(),
                        where=mask, 
                        alpha=0.3, 
                        label=f'State {state}')

    ax1.legend()

    # Plot returns
    ax2.plot(data.index, data['Returns'])
    ax2.set_title('SP500 Returns')
    ax2.set_ylabel('Returns')
    ax2.set_xlabel('Date')

    plt.tight_layout()
    plt.savefig('hmm_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as hmm_results.png")
    

def calculate_prediction_accuracy(true_states, predicted_states):
    return np.mean(np.array(true_states[1:]) == np.array(predicted_states[:-1]))

def calculate_bic(model, X):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    n_params = (model.n_components - 1) + model.n_components * (model.n_components - 1) + 2 * model.n_components * n_features
    bic = -2 * model.score(X) + n_params * np.log(n_samples)
    return bic

def time_series_cv(X, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        model = hmm.GaussianHMM(n_components=7, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X_train)
        scores.append(model.score(X_test))

    return np.mean(scores), np.std(scores)

def analyze_feature_importance(model, feature_names):
    importance = np.abs(model.means_).sum(axis=0)
    importance /= importance.sum()
    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.4f}")

def train_test_split_timeseries(data, train_ratio=0.5):
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data

def validate_model(train_states, oos_states, model, state_names):
    # 1. State distribution comparison
    train_counts = np.bincount(train_states, minlength=model.n_components)
    oos_counts = np.bincount(oos_states, minlength=model.n_components)
    
    print("\n=== State Distribution Comparison ===")
    dist_df = pd.DataFrame({
        'State': [state_names[i] for i in range(model.n_components)],
        'Train %': train_counts/train_counts.sum(),
        'OOS %': oos_counts/oos_counts.sum()
    })
    print(dist_df)
    
    # 2. Transition matrix stability
    print("\nTransition Matrix:")
    print(pd.DataFrame(model.transmat_, 
                      columns=[f"To {state_names[i]}" for i in range(model.n_components)],
                      index=[f"From {state_names[i]}" for i in range(model.n_components)]))
    
    # 3. Next-state prediction accuracy
    next_states = [predict_next_state(model, s) for s in oos_states[:-1]]
    accuracy = np.mean(np.array(next_states) == np.array(oos_states[1:]))
    print(f"\nNext-State Prediction Accuracy (OOS): {accuracy:.2%}")

def save_model_stats(model, train_data, oos_data, train_states, oos_states, features, state_names, filename='model_stats.txt'):
    with open(filename, 'w') as f:
        # Redirect print output to both console and file
        def print_both(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=f)
        
        print_both("=== HMM Model Statistics ===")
        print_both(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Basic Info
        print_both("=== Model Configuration ===")
        print_both(f"Number of states: {model.n_components}")
        print_both(f"Features used: {', '.join(features)}")
        print_both(f"Training periods: {len(train_data)}")
        print_both(f"OOS periods: {len(oos_data)}\n")
        
        # 2. State Distributions
        print_both("=== State Distributions ===")
        train_counts = np.bincount(train_states, minlength=model.n_components)
        oos_counts = np.bincount(oos_states, minlength=model.n_components)
        
        dist_df = pd.DataFrame({
            'State': state_names,
            'Train Count': train_counts,
            'Train %': train_counts/train_counts.sum(),
            'OOS Count': oos_counts,
            'OOS %': oos_counts/oos_counts.sum()
        })
        print_both(dist_df.to_string())
        print_both()
        
        # 3. Transition Matrix
        print_both("=== Transition Matrix ===")
        trans_df = pd.DataFrame(model.transmat_, 
                              columns=[f"To {name}" for name in state_names],
                              index=[f"From {name}" for name in state_names])
        print_both(trans_df.to_string(float_format="%.3f"))
        print_both()
        
        # 4. State Characteristics
        print_both("=== State Characteristics ===")
        for i in range(model.n_components):
            print_both(f"\nState {i} ({state_names[i]}):")
            print_both("Mean:", model.means_[i])
            print_both("Covariance:\n", model.covars_[i])
        print_both()
        
        # 5. Performance Metrics
        print_both("=== Performance Metrics ===")
        
        # Training metrics
        train_next_state_preds = [predict_next_state(model, state) for state in train_states]
        train_accuracy = calculate_prediction_accuracy(train_states, train_next_state_preds)
        train_ll = model.score(scaler.transform(train_data[features]))
        
        # OOS metrics
        oos_next_state_preds = [predict_next_state(model, state) for state in oos_states]
        oos_accuracy = calculate_prediction_accuracy(oos_states, oos_next_state_preds)
        oos_ll = model.score(scaler.transform(oos_data[features]))
        
        print_both(f"{'Metric':<25} {'Training':<15} {'OOS':<15}")
        print_both(f"{'Accuracy':<25} {train_accuracy:.2%}{'':<5} {oos_accuracy:.2%}")
        print_both(f"{'Log-Likelihood':<25} {train_ll:.2f}{'':<5} {oos_ll:.2f}")
        print_both(f"{'BIC':<25} {calculate_bic(model, scaler.transform(train_data[features])):.2f}{'':<5} "
                f"{calculate_bic(model, scaler.transform(oos_data[features])):.2f}")
        
        # 6. Feature Importance
        print_both("\n=== Feature Importance ===")
        importance = np.abs(model.means_).sum(axis=0)
        importance /= importance.sum()
        for name, imp in zip(features, importance):
            print_both(f"{name}: {imp:.4f}")

# Main execution
features = [
    'Vol_Regime_Score',       # Improved volatility measure
    'Fisher_RSI_Scaled',      # Keep (works well)
    'Enhanced_Trend_v2',       # ADX + RSI-filtered
    'MACD_Strength'           # Momentum confirmation
]
state_names = [str(i) for i in range(7)]

print("Starting main execution...")
file_path = r'C:\Users\joshd\OneDrive - Monash University\Projects\Python\Trading\Intro_using_AI\data\sp500_daily_all.csv'
data = load_and_preprocess_data(file_path)

# ========== CRITICAL OOS SPLIT ==========
train_data, oos_data = train_test_split_timeseries(data, train_ratio=0.7)
print(f"\nData split: Train={len(train_data)} periods, OOS={len(oos_data)} periods")

# ========== TRAIN ONLY ON TRAINING DATA ==========
print("\nTraining HMM model...")
model, scaler, X_scaled = train_hmm(train_data, features, n_components=7)

# ========== PREDICT ON BOTH SETS ==========
print("\n=== Training Set ===")
train_states = predict_states(model, train_data, scaler, features)
analyze_states(train_data, train_states, model, features)
    
print("\n=== Out-of-Sample Set ===") 
oos_states = predict_states(model, oos_data, scaler, features)
analyze_states(oos_data, oos_states, model, features)

# ========== VALIDATION ==========
validate_model(train_states, oos_states, model, state_names)

# ========== VISUALIZATION ==========
print("\nPlotting results...")
plot_results(pd.concat([train_data, oos_data]),  # Combine for full timeline
            np.concatenate([train_states, oos_states]), 
            model)
    
# ========== MODEL METRICS ==========
print("\n=== Model Evaluation ===")
print("Transition Matrix:")
print(model.transmat_)
    
print("\nState Characteristics:")
for i in range(model.n_components):
    print(f"\nState {i} ({state_names[i]}):")
    print("Mean:", model.means_[i])
    print("Covariance:", model.covars_[i])
    
# ========== SAVE RESULTS ==========
output_file = 'sp500_state_changes.csv'
save_state_changes(np.concatenate([train_states, oos_states]), 
                    pd.concat([train_data, oos_data]), 
                    state_names, 
                    output_file)
    
# ===== MODIFIED VERSION FOR OOS TESTING =====
print("\n=== Training Set Metrics ===")
# Training set evaluation
train_next_state_preds = [predict_next_state(model, state) for state in train_states]
train_accuracy = calculate_prediction_accuracy(train_states, train_next_state_preds)
train_log_likelihood = model.score(scaler.transform(train_data[features]))
train_bic = calculate_bic(model, scaler.transform(train_data[features]))

print(f"Train State Prediction Accuracy: {train_accuracy:.2%}")
print(f"Train Log-likelihood: {train_log_likelihood:.2f}")
print(f"Train BIC: {train_bic:.2f}")

print("\n=== Out-of-Sample Metrics ===")
# OOS set evaluation
oos_next_state_preds = [predict_next_state(model, state) for state in oos_states]
oos_accuracy = calculate_prediction_accuracy(oos_states, oos_next_state_preds)
oos_log_likelihood = model.score(scaler.transform(oos_data[features])) 
oos_bic = calculate_bic(model, scaler.transform(oos_data[features]))

print(f"OOS State Prediction Accuracy: {oos_accuracy:.2%}") 
print(f"OOS Log-likelihood: {oos_log_likelihood:.2f}")
print(f"OOS BIC: {oos_bic:.2f}")

# Cross-validation (use only training data to avoid leakage)
print("\n=== Cross-Validation ===")
cv_mean, cv_std = time_series_cv(scaler.transform(train_data[features]))    
print(f"CV Score: {cv_mean:.2f} (+/- {cv_std:.2f})")

# Feature importance (on full model)
print("\nFeature Importance Analysis:")
analyze_feature_importance(model, features)

save_model_stats(
    model=model,
    train_data=train_data,
    oos_data=oos_data,
    train_states=train_states,
    oos_states=oos_states,
    features=features,
    state_names=state_names,
    filename='model_stats.txt'
)

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\nModel and scaler saved successfully.")