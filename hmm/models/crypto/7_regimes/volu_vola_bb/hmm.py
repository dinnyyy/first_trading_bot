'''
Transition Matrix.
[[9.56926232e-001 6.17607791e-180 7.79221541e-004 8.93532614e-019
  2.06524488e-046 4.22945460e-002 1.04693724e-102]
 [8.19659411e-004 7.63720096e-002 1.82582098e-002 1.59464289e-002
  1.45936998e-001 6.55054300e-053 7.42666695e-001]
 [6.90044809e-103 2.07411175e-003 8.87003088e-001 7.09589064e-002
  1.26427565e-002 1.82299399e-010 2.73211373e-002]
 [4.76241946e-002 9.55030063e-036 3.31161132e-001 1.55767824e-002
  1.71968463e-001 4.33121367e-001 5.48060897e-004]
 [3.62368789e-072 3.20233910e-002 2.20974206e-002 6.59431773e-002
  8.75611513e-001 2.82025752e-003 1.50424056e-003]
 [1.57149391e-002 2.01401169e-077 2.53773533e-002 7.47267626e-002
  3.66067052e-003 8.79546143e-001 9.74131965e-004]
 [4.45678290e-065 2.25511474e-001 3.12094728e-031 3.25771808e-003
  4.52290941e-041 1.33804611e-109 7.71230808e-001]]

Printing means and covariances of each state...
State 0 (Bullish Trending):
Mean: [ 1.87889553  1.89930041 -0.08073279]
Covariance: [[ 1.60885126  0.30183721 -0.05955157]
 [ 0.30183721  1.1970062   0.03752578]
 [-0.05955157  0.03752578  0.83132374]]

State 1 (Bearish Trending):
Mean: [-0.75801896 -0.87328458  1.54522282]
Covariance: [[ 0.03780176  0.04505076 -0.02596134]
 [ 0.04505076  0.09411318 -0.01417941]
 [-0.02596134 -0.01417941  1.67983636]]

State 2 (Sideways Consolidation):
Mean: [-0.40495965 -0.05094492 -0.17382837]
Covariance: [[ 0.06542287  0.0258121  -0.0147981 ]
 [ 0.0258121   0.05435368 -0.00292134]
 [-0.0147981  -0.00292134  0.44539961]]

State 3 (Upward Consolidation):
Mean: [0.06842489 0.18504826 2.59054238]
Covariance: [[ 0.2726283   0.07616533 -0.0542435 ]
 [ 0.07616533  0.25415505  0.04880232]
 [-0.0542435   0.04880232  1.12823832]]

State 4 (Downward Consolidation):
Mean: [-0.0598774  -0.4896916  -0.23933228]
Covariance: [[ 0.11579805  0.06216634 -0.00186656]
 [ 0.06216634  0.08019945 -0.00394146]
 [-0.00186656 -0.00394146  0.38959327]]

State 5 (Downward Capitulation):
Mean: [ 0.48929403  0.55249973 -0.24761814]
Covariance: [[ 0.32453849 -0.07288809 -0.01824488]
 [-0.07288809  0.15549276  0.01153592]
 [-0.01824488  0.01153592  0.37115705]]

State 6 (Upward Capitulation):
Mean: [-0.84018812 -0.93016571 -0.31465345]
Covariance: [[ 0.02814506  0.03991536 -0.00535109]
 [ 0.03991536  0.09579165 -0.00056577]
 [-0.00535109 -0.00056577  0.2458214 ]]

Bitcoin HMM analysis completed.
State Prediction Accuracy: 0.85
Log-likelihood: -34453.28
BIC: 69785.90
Cross-Validation Score: -6328.77 (+/- 910.39)

Feature Importance Analysis:
BB_width: 0.3067
Volatility: 0.3395
Volume_Change: 0.3539
'''

import pandas as pd
import numpy as np
import time 
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta
import ccxt
exchange = ccxt.binance()
from sklearn.model_selection import TimeSeriesSplit
import joblib
import pandas_ta as ta

def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    
    # Read CSV with custom column names, no header
    df = pd.read_csv(file_path)

    print("Creating datetime index...")
    # Current time in milliseconds
    now_ms = exchange.milliseconds()

    # Subtract 5 years in milliseconds
    five_years_ago_ms = now_ms - int(5 * 365.25 * 24 * 60 * 60 * 1000)

    # Convert to datetime
    five_years_ago_dt = datetime.utcfromtimestamp(five_years_ago_ms / 1000)

    # Format as yyyy-mm-dd
    formatted_date = five_years_ago_dt.strftime('%Y-%m-%d')

    df.index = pd.date_range(start=formatted_date, periods=len(df), freq='h')
    
    print("Calculating returns and volatility...")
    # Calculate returns with safeguards
    df['Returns'] = df['Close'].pct_change()
    df['Returns'] = df['Returns'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate volatility
    df['Volatility'] = df['Returns'].rolling(window=24).std()

    print("Calculating volume change...")
    # Calculate volume change with safeguards
    df['Volume_Change'] = df['Volume'].pct_change()

    print("Calculating Bollinger Bands...")
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']  # Middle band
    
    # === RSI ===
    df['rsi'] = ta.rsi(df['Close'], length=14)

    # === MACD ===
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    # === ATR ===
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # === EMA Deviation ===
    df['ema_20'] = ta.ema(df['Close'], length=20)
    df['price_ema_diff'] = (df['Close'] - df['ema_20']) / df['ema_20']
    
    # Handle extreme volume changes
    volume_change = df['Volume_Change']
    valid_changes = volume_change[~np.isinf(volume_change)]
    
    # Calculate reasonable bounds (e.g., 3 standard deviations)
    if len(valid_changes) > 0:
        mean_change = valid_changes.mean()
        std_change = valid_changes.std()
        upper_bound = mean_change + 3 * std_change
        lower_bound = mean_change - 3 * std_change
        
        # Clip volume changes to these bounds
        df['Volume_Change'] = df['Volume_Change'].clip(lower_bound, upper_bound)
    else:
        # If no valid changes, replace inf values with 0
        df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], 0)

    print("Dropping NaN values...")
    df.dropna(inplace=True)

    # Add data quality checks
    print("Performing data quality checks...")
    for column in ['BB_width', 'Volatility', 'Volume_Change']:
        inf_count = np.isinf(df[column]).sum()
        nan_count = np.isnan(df[column]).sum()
        print(f"{column}:")
        print(f"  Infinite values: {inf_count}")
        print(f"  NaN values: {nan_count}")
        print(f"  Max value: {df[column].max()}")
        print(f"  Min value: {df[column].min()}")
        print()

    print(f"Data preprocessed. Shape: {df.shape}")
    return df

def train_hmm(data, features, n_components=3):
    print(f'Training HMM with {n_components} components...')
    X = data[features].values

    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Fitting HMM model...")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
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
    ax1.set_title('Bitcoin Price and HMM States')
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
    ax2.set_title('Bitcoin Returns')
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
        model = hmm.GaussianHMM(n_components=7, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X_train)
        scores.append(model.score(X_test))

    return np.mean(scores), np.std(scores)

def analyze_feature_importance(model, feature_names):
    importance = np.abs(model.means_).sum(axis=0)
    importance /= importance.sum()
    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.4f}")

# Main execution
features = ['BB_width', 'Volatility', 'Volume_Change']
print("Starting main execution...")
file_path = r'C:\Users\joshd\OneDrive - Monash University\Projects\Python\Trading\Intro_using_AI\data\btc_1h_2years.csv'
data = load_and_preprocess_data(file_path)

print("Training HMM model...")
model, scaler, X_scaled = train_hmm(data, features, n_components=7)

print("Predicting states...")
states = predict_states(model, data, scaler, features)

state_names = [
    "Bullish Trending",
    "Bearish Trending",
    "Sideways Consolidation",
    "Upward Consolidation",
    "Downward Consolidation",
    "Downward Capitulation",
    "Upward Capitulation"
]

output_file = 'bitcoin_state_changes.csv'
save_state_changes(states, data, state_names, output_file)

print("Analyzing states...")
analyze_states(data, states, model, features)

print("Plotting results...")
plot_results(data, states, model)

print("Printing transition matrix...")
print("Transition Matrix.")
print(model.transmat_)

print("\nPrinting means and covariances of each state...")
for i in range(model.n_components):
    print(f"State {i} ({state_names[i]}):")
    print("Mean:", model.means_[i])
    print("Covariance:", model.covars_[i])
    print()

print("Bitcoin HMM analysis completed.")

next_state_predictions = [predict_next_state(model, state) for state in states]

accuracy = calculate_prediction_accuracy(states, next_state_predictions)
print(f"State Prediction Accuracy: {accuracy:.2f}")

log_likelihood = model.score(X_scaled)
print(f"Log-likelihood: {log_likelihood:.2f}")

bic = calculate_bic(model, X_scaled)
print(f"BIC: {bic:.2f}")

cv_mean, cv_std = time_series_cv(X_scaled)    
print(f"Cross-Validation Score: {cv_mean:.2f} (+/- {cv_std:.2f})")

print("\nFeature Importance Analysis:")
analyze_feature_importance(model, features)

# Save model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully.")