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

    df['Returns_t'] = df['Close'].pct_change()  # r_t (same as existing 'Returns')
    df['Returns_t_1'] = df['Returns_t'].shift(1)  # r_{t-1}
    df['Returns_t_2'] = df['Returns_t'].shift(2)  # r_{t-2} 


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

def find_optimal_states_metrics_train_oos(
    train_data_df,
    oos_data_df,
    feature_list,
    calculate_bic_func, # Pass your BIC calculation function
    min_states=2,
    max_states=10,
    n_iter=1000,
    random_state=42,
    cov_type="full"
):
    """
    Trains HMM models with varying states on train_data_df, then evaluates
    log-likelihood and BIC on both train_data_df and oos_data_df.

    Args:
        train_data_df (pd.DataFrame): DataFrame for training.
        oos_data_df (pd.DataFrame): DataFrame for out-of-sample evaluation.
        feature_list (list): List of column names for features.
        calculate_bic_func (function): Function to calculate BIC, taking (model, X_scaled_data).
        min_states (int): Minimum HMM states to test.
        max_states (int): Maximum HMM states to test.
        n_iter (int): Iterations for HMM fitting.
        random_state (int): Random state for HMM.
        cov_type (str): Covariance type for GaussianHMM.

    Returns:
        pd.DataFrame: With columns ['Num_States', 'Train_LogLikelihood', 'Train_BIC',
                                  'OOS_LogLikelihood', 'OOS_BIC', 'Converged'].
    """
    print(f"Evaluating HMM models (states {min_states}-{max_states}) on Train and OOS data...")

    # --- Input Validations ---
    if train_data_df.empty or not feature_list:
        print("Warning: train_data_df is empty or feature_list is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=['Num_States', 'Train_LogLikelihood', 'Train_BIC', 'OOS_LogLikelihood', 'OOS_BIC', 'Converged'])
    
    if not all(feature in train_data_df.columns for feature in feature_list):
        print(f"Error: Features not found in train_data_df: {[f for f in feature_list if f not in train_data_df.columns]}")
        return pd.DataFrame(columns=['Num_States', 'Train_LogLikelihood', 'Train_BIC', 'OOS_LogLikelihood', 'OOS_BIC', 'Converged'])

    X_train = train_data_df[feature_list].values
    if X_train.shape[0] == 0:
        print("Warning: Train feature matrix X_train is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=['Num_States', 'Train_LogLikelihood', 'Train_BIC', 'OOS_LogLikelihood', 'OOS_BIC', 'Converged'])
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("Error: Train feature matrix X_train contains NaN/Inf values. Please preprocess.")
        return pd.DataFrame(columns=['Num_States', 'Train_LogLikelihood', 'Train_BIC', 'OOS_LogLikelihood', 'OOS_BIC', 'Converged'])

    # --- Prepare OOS data (if available) ---
    oos_available = False
    X_oos = np.array([]) # Initialize to handle cases where oos_data_df might be empty or problematic
    if not oos_data_df.empty:
        if not all(feature in oos_data_df.columns for feature in feature_list):
            print(f"Warning: Features not found in oos_data_df: {[f for f in feature_list if f not in oos_data_df.columns]}. OOS metrics will be NaN.")
        else:
            X_oos_temp = oos_data_df[feature_list].values
            if X_oos_temp.shape[0] > 0:
                if np.any(np.isnan(X_oos_temp)) or np.any(np.isinf(X_oos_temp)):
                    print("Warning: OOS feature matrix X_oos contains NaN/Inf values. OOS metrics will be NaN.")
                else:
                    X_oos = X_oos_temp # Assign only if valid
                    oos_available = True
            else:
                print("Warning: OOS feature matrix X_oos is empty. OOS metrics will be NaN.")
    else:
        print("Warning: oos_data_df is empty. OOS metrics will be NaN.")

    # --- Scale Data (Fit on Train, Transform Both) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if oos_available:
        X_oos_scaled = scaler.transform(X_oos)
    else:
        X_oos_scaled = np.array([]) # Ensure it's defined

    results = []

    for n_components in range(min_states, max_states + 1):
        if X_train_scaled.shape[0] < n_components:
            print(f"  Skipping {n_components} states for training: num train samples ({X_train_scaled.shape[0]}) < n_components.")
            continue
        
        print(f"  Training HMM with {n_components} states...")
        model = hmm.GaussianHMM(n_components=n_components,
                                covariance_type=cov_type,
                                n_iter=n_iter,
                                random_state=random_state,
                                tol=1e-3,
                                verbose=False,
                                params="stmc",
                                init_params="stmc")
        
        converged = False
        train_log_likelihood = np.nan
        train_bic = np.nan
        oos_log_likelihood = np.nan
        oos_bic = np.nan

        try:
            model.fit(X_train_scaled)
            converged = model.monitor_.converged
            if not converged:
                print(f"    Warning: Model with {n_components} states (training) did not converge after {n_iter} iterations.")

            # --- Train Metrics ---
            train_log_likelihood = model.score(X_train_scaled)
            train_bic = calculate_bic_func(model, X_train_scaled)

            # --- OOS Metrics ---
            if oos_available and X_oos_scaled.shape[0] > 0:
                # Ensure oos_data is not too small for certain covariance types if model.score has internal checks
                # For GaussianHMM, model.score should generally work if the model is fitted.
                try:
                    oos_log_likelihood = model.score(X_oos_scaled)
                    oos_bic = calculate_bic_func(model, X_oos_scaled)
                except Exception as e_oos_score:
                    print(f"    Error scoring OOS data with {n_components} states model: {e_oos_score}")
                    # oos_log_likelihood and oos_bic remain NaN
            
        except ValueError as e_fit:
            print(f"    Error training model with {n_components} states: {e_fit}")
        except Exception as e_generic:
            print(f"    An unexpected error occurred with {n_components} states: {e_generic}")

        results.append({
            'Num_States': n_components,
            'Train_LogLikelihood': train_log_likelihood,
            'Train_BIC': train_bic,
            'OOS_LogLikelihood': oos_log_likelihood,
            'OOS_BIC': oos_bic,
            'Converged': converged
        })

    results_df = pd.DataFrame(results)
    print("\nFinished evaluating models on Train and OOS data.")

    output_filename = "hmm_state_metrics_summary.txt"
    try:
        with open(output_filename, 'w') as f:
            f.write(f"HMM State Metrics (Train & OOS) - Evaluated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Features used: " + ", ".join(feature_list) + "\n")
            f.write(f"Covariance type: {cov_type}\n\n")
            results_df_string = results_df.to_string(index=False) # Get DataFrame as a string
            f.write(results_df_string)
        print(f"\nMetrics table saved to {output_filename}")
    except Exception as e:
        print(f"Error saving metrics table to {output_filename}: {e}")


    return results_df

# Main execution
features = [
    'Returns_t',       # Current return
    'Returns_t_1',     # 1-period lagged return
    'Returns_t_2',     # 2-period lagged return
    'RSI',             # RSI(14)
    'MACD_Signal',     # MACD signal line
    'Volatility'       # 20-day rolling std
    # Alternatively use 'GARCH_Vol' if you implemented it
]
state_names = [str(i) for i in range(10)]

print("Starting main execution...")
file_path = r'C:\Users\joshd\OneDrive - Monash University\Projects\Python\Trading\Intro_using_AI\data\sp500_daily_all.csv'
data = load_and_preprocess_data(file_path)

# ========== CRITICAL OOS SPLIT ==========
train_data, oos_data = train_test_split_timeseries(data, train_ratio=0.7)
print(f"\nData split: Train={len(train_data)} periods, OOS={len(oos_data)} periods")

train_data_cleaned = train_data.dropna(subset=features).copy()
oos_data_cleaned = oos_data.dropna(subset=features).copy()

print("\nStarting HMM state optimization process (Train & OOS)...")

# Make sure calculate_bic is defined before this call
# def calculate_bic(model, X_scaled_data): ... (as defined previously)

state_metrics_train_oos_df = find_optimal_states_metrics_train_oos(
    train_data_df=train_data_cleaned,
    oos_data_df=oos_data_cleaned, # Pass the cleaned OOS data
    feature_list=features,
    calculate_bic_func=calculate_bic,
    min_states=2,
    max_states=15, # Adjust as needed
    n_iter=1000,
    random_state=42
)

print("\nModel Metrics (Train & OOS) for Different Numbers of States:")
print(state_metrics_train_oos_df)
