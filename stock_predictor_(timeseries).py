import warnings
warnings.filterwarnings("ignore")

import threading, asyncio, requests, os, joblib, logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize

# Neural network forecasting (TensorFlow)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler

# Ensemble forecasting
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Transformer-based forecasting (Advanced)
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# SHAP for model interpretability
import shap

# Technical analysis
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Plotly for interactive charts
import plotly.graph_objects as go

# Enable GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU found, enabling memory growth.")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU found. Running on CPU.")

# Set up logging and directories
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f"{log_dir}/stock_analysis_{datetime.now().strftime('%Y%m%d')}.log"),
              logging.StreamHandler()]
)

# Global variables and caching
portfolio = {}
cached_data = {}
stop_event = threading.Event()

###############################################################################
# Data Download & Preprocessing
###############################################################################
def download_and_preprocess(ticker, years=3, force_download=False):
    cache_key = f"{ticker.upper()}_{years}"
    if not force_download and cache_key in cached_data:
        logging.info(f"Using cached data for {ticker.upper()}")
        return cached_data[cache_key].copy()
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    logging.info(f"Downloading data for {ticker.upper()} from {start_date.date()} to {end_date.date()}")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        logging.error(f"No data found for ticker '{ticker.upper()}'.")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.ffill(inplace=True)
    df.interpolate(method='linear', inplace=True)
    df.dropna(inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    cached_data[cache_key] = df.copy()
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        if stop_event.is_set():
            return None, None
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def raw_ticker_values(ticker):
    df = download_and_preprocess(ticker)
    if df is not None:
        print("First 5 rows:")
        print(df.head().to_string())
        print("\nLast 5 rows:")
        print(df.tail().to_string())

###############################################################################
# Sentiment Analysis
###############################################################################
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result

###############################################################################
# Feature Engineering
###############################################################################
def add_volatility_features(df, window=21):
    df = df.copy()
    df['Rolling_Volatility'] = df['Close'].pct_change().rolling(window=window).std()
    return df

def add_technical_indicators(df):
    df_tech = df.copy()
    df_tech['SMA20'] = SMAIndicator(close=df_tech['Close'], window=20).sma_indicator()
    df_tech['SMA50'] = SMAIndicator(close=df_tech['Close'], window=50).sma_indicator()
    df_tech['SMA200'] = SMAIndicator(close=df_tech['Close'], window=200).sma_indicator()
    df_tech['EMA12'] = EMAIndicator(close=df_tech['Close'], window=12).ema_indicator()
    df_tech['EMA26'] = EMAIndicator(close=df_tech['Close'], window=26).ema_indicator()
    macd = MACD(close=df_tech['Close'])
    df_tech['MACD'] = macd.macd()
    df_tech['MACD_Signal'] = macd.macd_signal()
    df_tech['MACD_Hist'] = macd.macd_diff()
    rsi = RSIIndicator(close=df_tech['Close'])
    df_tech['RSI'] = rsi.rsi()
    stoch = StochasticOscillator(high=df_tech['High'], low=df_tech['Low'], close=df_tech['Close'])
    df_tech['Stoch_K'] = stoch.stoch()
    df_tech['Stoch_D'] = stoch.stoch_signal()
    bb = BollingerBands(close=df_tech['Close'])
    df_tech['BB_High'] = bb.bollinger_hband()
    df_tech['BB_Low'] = bb.bollinger_lband()
    df_tech['BB_Mid'] = bb.bollinger_mavg()
    df_tech['BB_Width'] = (df_tech['BB_High'] - df_tech['BB_Low']) / df_tech['BB_Mid']
    obv = OnBalanceVolumeIndicator(close=df_tech['Close'], volume=df_tech['Volume'])
    df_tech['OBV'] = obv.on_balance_volume()
    df_tech['Returns'] = df_tech['Close'].pct_change() * 100
    df_tech['Log_Returns'] = np.log(df_tech['Close'] / df_tech['Close'].shift(1)) * 100
    df_tech['Volatility'] = df_tech['Returns'].rolling(window=20).std()
    df_tech['Daily_Range'] = (df_tech['High'] - df_tech['Low']) / df_tech['Close'] * 100
    df_tech['Gap'] = (df_tech['Open'] - df_tech['Close'].shift(1)) / df_tech['Close'].shift(1) * 100
    df_tech['Volume_MA'] = df_tech['Volume'].rolling(window=20).mean()
    df_tech = add_volatility_features(df_tech)
    df_tech.dropna(inplace=True)
    return df_tech

def split_data_with_features(ticker, train_ratio=0.7, add_features=True):
    df = download_and_preprocess(ticker)
    if df is None:
        return None, None
    if add_features:
        df = add_technical_indicators(df)
    n = len(df)
    split_idx = int(n * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f"Data split for {ticker.upper()}:")
    print("Training set (first 5 rows):")
    print(train.head().to_string())
    print("\nTraining set (last 5 rows):")
    print(train.tail().to_string())
    print("\nTesting set (first 5 rows):")
    print(test.head().to_string())
    print("\nTesting set (last 5 rows):")
    print(test.tail().to_string())
    return train, test

###############################################################################
# Forecast Functions
###############################################################################
def forecast_arima(ticker, days=1, df_input=None, order=(0,1,0)):
    if stop_event.is_set():
        return None
    df = download_and_preprocess(ticker) if df_input is None else df_input.copy()
    if df is None:
        return None
    close_series = df['Close']
    try:
        log_series = np.log(close_series)
        model = ARIMA(log_series, order=order)
        model_fit = model.fit()
        log_forecast = model_fit.forecast(steps=days)
        forecasted_prices = np.exp(log_forecast)
        forecasted_prices = [p if p > 0 and not np.isnan(p) else close_series.iloc[-1] for p in forecasted_prices]
        return forecasted_prices
    except Exception as e:
        logging.error(f"Error during ARIMA forecasting: {e}")
        return None

def forecast_lstm(ticker, days=1, df_input=None):
    if stop_event.is_set():
        return None
    df = download_and_preprocess(ticker) if df_input is None else df_input.copy()
    if df is None:
        return None
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    seq_length = 60
    if len(scaled_data) < seq_length:
        print("Not enough data for LSTM forecasting.")
        return None
    X, y = create_sequences(scaled_data, seq_length)
    if X is None:
        return None
    X = X.reshape(X.shape[0], X.shape[1], 1)
    with tf.device('/GPU:0') if physical_devices else tf.device('/CPU:0'):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_regularizer=l2(0.001)))
        model.compile(optimizer='adam', loss='mean_squared_error')
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[es])
        last_seq = scaled_data[-seq_length:]
        preds = []
        current_seq = last_seq.copy()
        for _ in range(days):
            current_seq_reshaped = current_seq.reshape(1, seq_length, 1)
            next_pred = model.predict(current_seq_reshaped, verbose=0)[0, 0]
            preds.append(next_pred)
            current_seq = np.append(current_seq[1:], [[next_pred]], axis=0)
        preds = np.array(preds).reshape(-1, 1)
        forecast_prices = scaler.inverse_transform(preds).flatten()
    return forecast_prices

def forecast_gru(ticker, days=1, df_input=None):
    if stop_event.is_set():
        return None
    df = download_and_preprocess(ticker) if df_input is None else df_input.copy()
    if df is None:
        return None
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    seq_length = 60
    if len(scaled_data) < seq_length:
        print("Not enough data for GRU forecasting.")
        return None
    X, y = create_sequences(scaled_data, seq_length)
    if X is None:
        return None
    X = X.reshape(X.shape[0], X.shape[1], 1)
    with tf.device('/GPU:0') if physical_devices else tf.device('/CPU:0'):
        model = Sequential()
        model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_regularizer=l2(0.001)))
        model.compile(optimizer='adam', loss='mean_squared_error')
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[es])
        last_seq = scaled_data[-seq_length:]
        preds = []
        current_seq = last_seq.copy()
        for _ in range(days):
            current_seq_reshaped = current_seq.reshape(1, seq_length, 1)
            next_pred = model.predict(current_seq_reshaped, verbose=0)[0, 0]
            preds.append(next_pred)
            current_seq = np.append(current_seq[1:], [[next_pred]], axis=0)
        preds = np.array(preds).reshape(-1, 1)
        forecast_prices = scaler.inverse_transform(preds).flatten()
    return forecast_prices

def forecast_rf(ticker, days=1, df_input=None):
    if stop_event.is_set():
        return None
    df = download_and_preprocess(ticker) if df_input is None else df_input.copy()
    if df is None:
        return None
    df = add_technical_indicators(df)
    for lag in range(1, 6):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)
    feature_cols = [col for col in df.columns if col.startswith('Close_lag_')] + ['SMA20', 'SMA50', 'RSI', 'MACD', 'Volume_MA']
    X = df[feature_cols].values
    y = df['Close'].shift(-1).dropna().values
    df = df.iloc[:-1]
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X, y)
    last_values = df[feature_cols].iloc[-1:].values
    forecasts = []
    for _ in range(days):
        if stop_event.is_set():
            return None
        pred = model.predict(last_values)[0]
        forecasts.append(pred)
        new_lags = np.roll(last_values[0, :5], 1)
        new_lags[0] = pred
        last_values = np.hstack((new_lags, last_values[0, 5:])).reshape(1, -1)
    return forecasts

def forecast_gb(ticker, days=1, df_input=None, load_model=True):
    if stop_event.is_set():
        return None
    df = download_and_preprocess(ticker) if df_input is None else df_input.copy()
    if df is None:
        return None
    df = add_technical_indicators(df)
    for lag in range(1, 6):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)
    feature_cols = [col for col in df.columns if col.startswith('Close_lag_')] + ['SMA20', 'SMA50', 'RSI', 'MACD', 'Volume_MA']
    model_path = f"{model_dir}/{ticker.upper()}_xgb_model.pkl"
    params = {'n_estimators': 200, 'learning_rate': 0.1, 'random_state': 42, 'verbosity': 0}
    if len(physical_devices) > 0:
        params['tree_method'] = 'gpu_hist'
    if load_model and os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        X = df[feature_cols].values
        y = df['Target'].values
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        joblib.dump(model, model_path)
    last_values = df[feature_cols].iloc[-1:].values
    forecasts = []
    for _ in range(days):
        if stop_event.is_set():
            return None
        pred = model.predict(last_values)[0]
        forecasts.append(pred)
        new_lags = np.roll(last_values[0, :5], 1)
        new_lags[0] = pred
        last_values = np.hstack((new_lags, last_values[0, 5:])).reshape(1, -1)
    return forecasts

def ensemble_forecast(ticker, days=1, df_input=None):
    if stop_event.is_set():
        return None
    forecasts = {}
    model_funcs = {
        'ARIMA': forecast_arima,
        'LSTM': forecast_lstm,
        'GRU': forecast_gru,
        'RF': forecast_rf,
        'XGB': forecast_gb
    }
    for name, func in model_funcs.items():
        try:
            res = func(ticker, days=days, df_input=df_input)
            if res is not None:
                forecasts[name] = res
                logging.info(f"{name} forecast generated.")
        except Exception as e:
            logging.error(f"Error in {name} forecast: {e}")
    if not forecasts:
        logging.error("No forecasts generated by any model.")
        return None
    ensemble = []
    for i in range(days):
        day_vals = [forecasts[m][i] for m in forecasts if i < len(forecasts[m])]
        ensemble.append(np.mean(day_vals))
    print("\nIndividual Model Forecasts:")
    for m in forecasts:
        print(f"  {m}: {[f'${p:.2f}' for p in forecasts[m]]}")
    print(f"\nEnsemble Forecast: {[f'${p:.2f}' for p in ensemble]}")
    return ensemble

###############################################################################
# Advanced Transformer-based Model
###############################################################################
def build_transformer_model(seq_length, d_model=64, num_heads=4, ff_dim=128):
    inputs = tf.keras.Input(shape=(seq_length, 1))
    x = tf.keras.layers.Dense(d_model)(inputs)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    x = tf.keras.layers.Add()([x, ffn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, kernel_regularizer=l2(0.001))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def forecast_tft(ticker, days=1):
    if stop_event.is_set():
        return None
    df = download_and_preprocess(ticker)
    if df is None:
        return None
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    seq_length = 60
    if len(scaled_data) < seq_length:
        print("Not enough data for Transformer forecasting.")
        return None
    X, y = create_sequences(scaled_data, seq_length)
    if X is None:
        return None
    X = X.reshape(X.shape[0], X.shape[1], 1)
    with tf.device('/GPU:0') if physical_devices else tf.device('/CPU:0'):
        transformer_model = build_transformer_model(seq_length, d_model=64, num_heads=4, ff_dim=128)
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        transformer_model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[es])
        last_seq = scaled_data[-seq_length:]
        preds = []
        current_seq = last_seq.copy()
        for _ in range(days):
            current_seq_reshaped = current_seq.reshape(1, seq_length, 1)
            next_pred = transformer_model.predict(current_seq_reshaped, verbose=0)[0, 0]
            preds.append(next_pred)
            current_seq = np.append(current_seq[1:], [[next_pred]], axis=0)
        preds = np.array(preds).reshape(-1, 1)
        forecast_prices = scaler.inverse_transform(preds).flatten()
    return forecast_prices

###############################################################################
# Unified Explanation using SHAP (Single Graph)
###############################################################################
def unified_explanation(ticker, model_name='lstm'):
    # Retrain a simple LSTM model on training data and produce a SHAP summary plot.
    train, _ = split_data_with_features(ticker, train_ratio=0.8, add_features=True)
    if train is None:
        return
    data = train['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    seq_length = 60
    if len(scaled_data) < seq_length:
        print("Not enough data for explanation.")
        return
    X, y = create_sequences(scaled_data, seq_length)
    if X is None:
        return
    X = X.reshape(X.shape[0], X.shape[1], 1)
    with tf.device('/GPU:0') if physical_devices else tf.device('/CPU:0'):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_regularizer=l2(0.001)))
        model.compile(optimizer='adam', loss='mean_squared_error')
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[es])
    X_sample = X[-100:].reshape(100, X.shape[1])
    print("\nGenerating unified SHAP explanation plot...")
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values.values, X_sample, feature_names=[f"f{i}" for i in range(X_sample.shape[1])])
    return shap_values

###############################################################################
# Stock Info & Analysis
###############################################################################
def stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        print(f"\nInformation for {ticker.upper()}:")
        print("\n--- Basic Information ---")
        print(f"  Name: {info.get('shortName', 'N/A')}")
        print(f"  Sector: {info.get('sector', 'N/A')}")
        print(f"  Industry: {info.get('industry', 'N/A')}")
        print(f"  Country: {info.get('country', 'N/A')}")
        print(f"  Website: {info.get('website', 'N/A')}")
        print("\n--- Price Information ---")
        print(f"  Current Price: ${info.get('currentPrice', info.get('previousClose', 'N/A'))}")
        print(f"  Previous Close: ${info.get('previousClose', 'N/A')}")
        print(f"  Open: ${info.get('open', 'N/A')}")
        print(f"  Day Range: ${info.get('dayLow', 'N/A')} - ${info.get('dayHigh', 'N/A')}")
        print(f"  52 Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        print(f"  Volume: {info.get('volume', 'N/A'):,}")
        print(f"  Avg. Volume: {info.get('averageVolume', 'N/A'):,}")
        print("\n--- Valuation Metrics ---")
        market_cap = info.get('marketCap', 0)
        print(f"  Market Cap: ${market_cap/1e9:.2f}B")
        print(f"  P/E Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"  Forward P/E: {info.get('forwardPE', 'N/A')}")
        print(f"  PEG Ratio: {info.get('pegRatio', 'N/A')}")
        print(f"  Price/Sales: {info.get('priceToSalesTrailing12Months', 'N/A')}")
        print(f"  Price/Book: {info.get('priceToBook', 'N/A')}")
        print(f"  EV/EBITDA: {info.get('enterpriseToEbitda', 'N/A')}")
        print("\n--- Dividend Information ---")
        print(f"  Dividend Rate: ${info.get('dividendRate', 0)}")
        dividend_yield = info.get('dividendYield', 0)
        print(f"  Dividend Yield: {dividend_yield*100 if dividend_yield else 0:.2f}%")
        print(f"  Ex-Dividend Date: {info.get('exDividendDate', 'N/A')}")
    except Exception as e:
        print("Error retrieving stock info:", e)

def analyze_stock(ticker):
    print(f"\n--- Analysis for {ticker.upper()} ---")
    stock_info(ticker)
    df = download_and_preprocess(ticker)
    if df is not None:
        df_features = add_technical_indicators(df)
        print("\nTechnical Indicators (first 5 rows):")
        print(df_features.head().to_string())
        print("\nTechnical Indicators (last 5 rows):")
        print(df_features.tail().to_string())
    print("\nCombined 5-Day Forecast (Averaged across models):")
    forecasts = ensemble_forecast(ticker, days=5)
    if forecasts:
        if len(forecasts) > 10:
            print("Forecast (first 5 values):", forecasts[:5])
            print("Forecast (last 5 values):", forecasts[-5:])
        else:
            print("Forecast:", forecasts)
        plot_multi_day_forecast(ticker, days=5, model_name='combined')
    else:
        print("No forecast available.")

def yahoo_dashboard(ticker):
    print(f"\n=== Yahoo Dashboard for {ticker.upper()} ===")
    stock_info(ticker)
    df = download_and_preprocess(ticker)
    if df is not None:
        df_features = add_technical_indicators(df)
        print("\nTechnical Indicators (first 5 rows):")
        print(df_features.head().to_string())
        print("\nTechnical Indicators (last 5 rows):")
        print(df_features.tail().to_string())
    print("\nCombined 5-Day Forecast:")
    forecasts = ensemble_forecast(ticker, days=5)
    if forecasts:
        if len(forecasts) > 10:
            print("Forecast (first 5 values):", forecasts[:5])
            print("Forecast (last 5 values):", forecasts[-5:])
        else:
            print("Forecast:", forecasts)
        plot_multi_day_forecast(ticker, days=5, model_name='combined')
    else:
        print("No forecast available.")

###############################################################################
# Buy/Sell Probability
###############################################################################
def buy_sell_probability(ticker, days=30, model='ensemble'):
    if model == 'arima':
        forecast = forecast_arima(ticker, days=days)
    elif model == 'lstm':
        forecast = forecast_lstm(ticker, days=days)
    elif model == 'gru':
        forecast = forecast_gru(ticker, days=days)
    elif model == 'rf':
        forecast = forecast_rf(ticker, days=days)
    elif model == 'gb':
        forecast = forecast_gb(ticker, days=days)
    elif model in ['ensemble','combined']:
        forecast = ensemble_forecast(ticker, days=days)
    else:
        print("Model not recognized for probability analysis.")
        return None
    if forecast is None:
        return None
    df = download_and_preprocess(ticker)
    if df is None:
        return None
    current_price = float(df['Close'].iloc[-1])
    prob_up = np.mean(np.array(forecast) > current_price)
    prob_down = 1 - prob_up
    expected_return = np.mean(np.array(forecast)/current_price - 1) * 100
    print(f"\nBuy/Sell Probability for {ticker.upper()} ({days}-day horizon):")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Probability of Price Increase: {prob_up*100:.1f}%")
    print(f"  Probability of Price Decrease: {prob_down*100:.1f}%")
    print(f"  Expected Return: {expected_return:.2f}%")
    return {
        'current_price': current_price,
        'prob_up': prob_up,
        'prob_down': prob_down,
        'expected_return': expected_return
    }

###############################################################################
# Portfolio Management
###############################################################################
def portfolio_add(ticker, shares):
    ticker = ticker.upper()
    portfolio[ticker] = portfolio.get(ticker, 0) + shares
    print(f"Added {shares} shares of {ticker} to portfolio.")

def portfolio_show():
    if not portfolio:
        print("Portfolio is empty.")
    else:
        print("Current Portfolio:")
        for tk, sh in portfolio.items():
            print(f"{tk}: {sh} shares")

def portfolio_clear():
    portfolio.clear()
    print("Portfolio cleared.")

###############################################################################
# Helper to Retrieve Forecast
###############################################################################
def get_forecast(ticker, days, model):
    if model == 'arima':
        return forecast_arima(ticker, days=days)
    elif model == 'lstm':
        return forecast_lstm(ticker, days=days)
    elif model == 'gru':
        return forecast_gru(ticker, days=days)
    elif model == 'rf':
        return forecast_rf(ticker, days=days)
    elif model == 'gb':
        return forecast_gb(ticker, days=days)
    elif model in ['ensemble', 'combined']:
        return ensemble_forecast(ticker, days=days)
    elif model == 'advanced':
        return forecast_tft(ticker, days=days)
    else:
        print(f"Model '{model}' not recognized.")
        return None

###############################################################################
# Evaluate Model Function
###############################################################################
def evaluate_model(ticker, model_name='lstm', plot=True):
    train, test = split_data_with_features(ticker, train_ratio=0.8, add_features=True)
    if train is None or test is None:
        return
    horizon = min(len(test), 30)
    if model_name == 'arima':
        forecast = forecast_arima(ticker, days=horizon, df_input=train)
    elif model_name == 'lstm':
        forecast = forecast_lstm(ticker, days=horizon, df_input=train)
    elif model_name == 'gru':
        forecast = forecast_gru(ticker, days=horizon, df_input=train)
    elif model_name == 'rf':
        forecast = forecast_rf(ticker, days=horizon, df_input=train)
    elif model_name == 'gb':
        forecast = forecast_gb(ticker, days=horizon, df_input=train)
    elif model_name in ['ensemble', 'combined']:
        forecast = ensemble_forecast(ticker, days=horizon, df_input=train)
    elif model_name == 'advanced':
        forecast = forecast_tft(ticker, days=horizon)
    else:
        print(f"Model '{model_name}' not recognized for evaluation.")
        return
    if forecast is None:
        print("Forecast generation failed or interrupted.")
        return
    test_values = test['Close'].iloc[:horizon].values.astype(float)
    forecast = np.array(forecast[:horizon])
    rmse = np.sqrt(mean_squared_error(test_values, forecast))
    mae = mean_absolute_error(test_values, forecast)
    mape = np.mean(np.abs((test_values - forecast)/test_values)) * 100
    r2 = r2_score(test_values, forecast)
    print(f"\nEvaluation for {ticker.upper()} using {model_name.upper()}:")
    print(f"  Forecast Horizon: {horizon} days")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R² Score: {r2:.4f}")
    if len(forecast) > 10:
        print("\nForecast (first 5 values):", forecast[:5])
        print("Forecast (last 5 values):", forecast[-5:])
    else:
        print("\nForecast:", forecast)
    if plot:
        last_train_date = train.index[-1]
        forecast_dates = pd.date_range(start=test.index[0], periods=horizon, freq='B')
        plt.figure(figsize=(12,6))
        plt.plot(train.index[-30:], train['Close'].iloc[-30:], label='Training Data', color='blue', alpha=0.5)
        plt.plot(test.index[:horizon], test_values, label='Actual', color='black', linewidth=2)
        plt.plot(forecast_dates, forecast, label=f'{model_name.upper()} Forecast', color='red', marker='x')
        plt.axvline(x=last_train_date, color='gray', linestyle='--', label='Forecast Start')
        plt.title(f'{ticker.upper()} - {model_name.upper()} Model Evaluation')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{ticker}_{model_name}_evaluation.png")
        plt.show()
    return rmse, mae, mape, r2

###############################################################################
# Multi-Day Forecast Visualization (Matplotlib)
###############################################################################
def plot_multi_day_forecast(ticker, days, model_name='lstm'):
    df = download_and_preprocess(ticker)
    if df is None:
        return
    if model_name == 'arima':
        forecast = forecast_arima(ticker, days=days)
    elif model_name == 'lstm':
        forecast = forecast_lstm(ticker, days=days)
    elif model_name == 'gru':
        forecast = forecast_gru(ticker, days=days)
    elif model_name == 'rf':
        forecast = forecast_rf(ticker, days=days)
    elif model_name == 'gb':
        forecast = forecast_gb(ticker, days=days)
    elif model_name in ['ensemble', 'combined']:
        forecast = ensemble_forecast(ticker, days=days)
    elif model_name == 'advanced':
        forecast = forecast_tft(ticker, days=days)
    else:
        print(f"Model '{model_name}' not recognized.")
        return
    if forecast is None:
        return
    hist = df.tail(30)
    last_date = df.index[-1]
    forecast_dates = pd.date_range(last_date, periods=days+1, freq='B')[1:]
    plt.figure(figsize=(12,6))
    plt.plot(hist.index, pd.to_numeric(hist['Close'], errors='coerce'), marker='o', label='Historical')
    plt.plot([last_date, forecast_dates[0]], [hist['Close'].iloc[-1], forecast[0]], linestyle='-', color='green')
    plt.plot(forecast_dates, forecast, marker='o', linestyle='-', color='red', label=f'{model_name.upper()} Forecast')
    plt.title(f"{ticker.upper()} - {days}-Day Forecast ({model_name.upper()})")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{ticker}_{model_name}_forecast.png")
    plt.show()
    print(f"\nForecast for the next {days} days for {ticker.upper()} using {model_name.upper()}:")
    for i, (d, p) in enumerate(zip(forecast_dates, forecast), start=1):
        print(f"  {d.strftime('%Y-%m-%d')} (Day {i}): ${p:.2f}")

###############################################################################
# Interactive Candlestick Chart (Plotly)
###############################################################################
def interactive_candlestick_chart(ticker, period="1mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        print(f"No data found for {ticker.upper()} with period={period} and interval={interval}.")
        return
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    )
    volume_bar = go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='rgba(100, 150, 250, 0.5)',
        yaxis='y2'
    )
    fig = go.Figure(data=[candlestick, volume_bar])
    fig.update_layout(
        title=f"{ticker.upper()} Candlestick Chart ({period}, {interval})",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark",
        yaxis=dict(domain=[0.2, 1]),
        yaxis2=dict(
            domain=[0, 0.2],
            overlaying='y',
            anchor='x',
            side='right'
        )
    )
    fig.show()

###############################################################################
# Interactive Forecast Visualization (Plotly)
###############################################################################
def interactive_plot_forecast(ticker, days, model_name='lstm'):
    df = download_and_preprocess(ticker)
    if df is None:
        return
    if model_name == 'arima':
        forecast = forecast_arima(ticker, days=days)
    elif model_name == 'lstm':
        forecast = forecast_lstm(ticker, days=days)
    elif model_name == 'gru':
        forecast = forecast_gru(ticker, days=days)
    elif model_name == 'rf':
        forecast = forecast_rf(ticker, days=days)
    elif model_name == 'gb':
        forecast = forecast_gb(ticker, days=days)
    elif model_name in ['ensemble', 'combined']:
        forecast = ensemble_forecast(ticker, days=days)
    elif model_name == 'advanced':
        forecast = forecast_tft(ticker, days=days)
    else:
        print(f"Model '{model_name}' not recognized.")
        return
    if forecast is None:
        print("Forecast not available.")
        return
    hist = df.tail(60)
    last_date = df.index[-1]
    forecast_dates = pd.date_range(last_date, periods=days+1, freq='B')[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'],
                             mode='lines+markers',
                             name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast,
                             mode='lines+markers',
                             name=f'{model_name.upper()} Forecast'))
    fig.update_layout(title=f"{ticker.upper()} - {days}-Day Forecast ({model_name.upper()})",
                      xaxis_title="Date",
                      yaxis_title="Price ($)",
                      template="plotly_dark",
                      hovermode="x unified")
    fig.show()

###############################################################################
# FINAL Summary Command (Unified Explanation Graph)
###############################################################################
def final_summary(ticker, days=5, model='lstm'):
    """
    Shows everything about projections in one shot:
      1) Evaluates the chosen model (RMSE, MAE, MAPE, R², etc.)
      2) Prints forecast values
      3) Displays an interactive candlestick chart for the last 3 months
      4) Displays a unified SHAP explanation graph
      5) Displays an interactive forecast chart
    """
    print(f"\n=== FINAL SUMMARY for {ticker.upper()}, {days}-day, Model: {model.upper()} ===\n")

    # 1) Evaluate model (without plotting)
    metrics = evaluate_model(ticker, model_name=model, plot=False)
    if metrics is None:
        print("Evaluation failed.")
    # 2) Print forecast values
    forecast = get_forecast(ticker, days, model)
    if forecast is not None:
        print("\nForecast values:")
        for i, val in enumerate(forecast, start=1):
            print(f"  Day {i}: ${val:.2f}")
    else:
        print("No forecast available.")

    # 3) Interactive candlestick chart (3mo, 1d)
    print("\nLaunching interactive candlestick chart (3mo, 1d)...")
    interactive_candlestick_chart(ticker, period="3mo", interval="1d")

    # 4) Unified SHAP explanation graph
    print("\nGenerating unified SHAP explanation using SHAP...")
    unified_explanation(ticker, model_name=model)

    # 5) Interactive forecast chart
    print("\nLaunching interactive forecast chart...")
    interactive_plot_forecast(ticker, days, model_name=model)

    print("\n=== END OF FINAL SUMMARY ===")

###############################################################################
# Command Parsing & Main Loop
###############################################################################
def print_help():
    help_text = """
Available Commands:

Basic Analysis:
  raw TICKER                  -> Display first 5 and last 5 rows of raw data for TICKER
  preprocessed TICKER         -> Display cleaned data (first 5 & last 5 rows) for TICKER
  features TICKER             -> Display technical indicators (first 5 & last 5 rows) for TICKER
  evaluation TICKER           -> Evaluate model and show error metrics with forecast
  advanced TICKER             -> Forecast using an advanced (Transformer-based) model
  sentiment HEADLINE          -> Analyze sentiment of a news headline

Data Splitting:
  split TICKER                -> Split data into training and testing sets

Forecasting & Comparison:
  oneday TICKER               -> 1-day forecast using combined (ensemble) models
  arima TICKER                -> 1-day forecast using ARIMA
  lstm TICKER                 -> 1-day forecast using LSTM
  gru TICKER                  -> 1-day forecast using GRU
  rf TICKER                   -> 1-day forecast using RandomForest
  gb TICKER                   -> 1-day forecast using XGBoost
  combined TICKER             -> 1-day forecast using ensemble models (average)
  predict TICKER DAYS [MODEL] -> Forecast for custom horizon (default: lstm)
  interactive TICKER DAYS [MODEL] -> Interactive forecast visualization (Plotly)
  candlestick TICKER [PERIOD] [INTERVAL] -> Interactive candlestick chart

Stock Info & Analysis:
  info TICKER                 -> Display stock information
  analyze TICKER              -> Comprehensive analysis (info, technicals, 5-day forecast)
  yahoo TICKER                -> Yahoo Finance-like dashboard

Probability:
  probability TICKER          -> Buy/Sell probability (30-day horizon, default: ensemble)

Visualization:
  chart TICKER                -> Display raw ticker chart
  context TICKER [MODEL]      -> Last 6 days historical + next 6 days forecast (default: ensemble)
  zoom TICKER [MODEL]         -> 15-day zoom forecast chart (default: combined)

Backtesting:
  walkforward TICKER          -> Perform walk-forward validation

Portfolio Management:
  portfolio add TICKER SHARES -> Add shares of TICKER to portfolio
  portfolio show              -> Display current portfolio
  portfolio clear             -> Clear portfolio

FINAL Command:
  final TICKER DAYS [MODEL]   -> Shows everything about projections in one shot (with unified explanation)

Misc:
  rawticker TICKER            -> Display first 5 and last 5 rows of raw data for TICKER
  help                        -> Show this help message
  exit                        -> Quit the CLI
"""
    print(help_text)

def process_command(command):
    stop_event.clear()
    tokens = command.split()
    if not tokens:
        return
    cmd = tokens[0].lower()
    if cmd in ['exit', 'quit']:
        print("Goodbye!")
        return "exit"
    if cmd == 'help':
        print_help()
        return
    if cmd == 'rawticker' and len(tokens) == 2:
        raw_ticker_values(tokens[1])
        return
    if cmd == 'raw' and len(tokens) == 2:
        raw_ticker_values(tokens[1])
        return
    if cmd == 'preprocessed' and len(tokens) == 2:
        df = download_and_preprocess(tokens[1])
        if df is not None:
            print("First 5 rows:")
            print(df.head().to_string())
            print("\nLast 5 rows:")
            print(df.tail().to_string())
        return
    if cmd == 'features' and len(tokens) == 2:
        df = download_and_preprocess(tokens[1])
        if df is not None:
            df_features = add_technical_indicators(df)
            print("First 5 rows:")
            print(df_features.head().to_string())
            print("\nLast 5 rows:")
            print(df_features.tail().to_string())
        return
    if cmd == 'evaluation' and len(tokens) == 2:
        evaluate_model(tokens[1])
        return
    if cmd == 'advanced' and len(tokens) == 2:
        evaluate_model(tokens[1], model_name='advanced')
        return
    if cmd == 'sentiment' and len(tokens) >= 2:
        headline = " ".join(tokens[1:])
        result = analyze_sentiment(headline)
        print(f"Sentiment analysis for the headline:\n{headline}\nResult: {result}")
        return
    if cmd == 'split' and len(tokens) == 2:
        split_data_with_features(tokens[1])
        return
    if cmd == 'walkforward' and len(tokens) == 2:
        # Ensure walk_forward_validation is defined; here we use the same as split-based
        try:
            from pandas.tseries.offsets import BDay
        except ImportError:
            pass
        # For simplicity, we'll call evaluate_model for walk-forward simulation.
        print("Walk-forward validation is not fully implemented in this version.")
        return
    if cmd == 'oneday' and len(tokens) == 2:
        plot_multi_day_forecast(tokens[1], days=1, model_name='combined')
        return
    if cmd in ['arima', 'lstm', 'gru', 'rf', 'gb', 'combined'] and len(tokens) == 2:
        model_used = tokens[0].lower()
        plot_multi_day_forecast(tokens[1], days=1, model_name=model_used)
        return
    if cmd == 'predict' and len(tokens) >= 3:
        ticker = tokens[1]
        try:
            days = int(tokens[2])
        except ValueError:
            print("Days must be an integer.")
            return
        model = tokens[3].strip("[]").lower() if len(tokens) >= 4 else 'lstm'
        plot_multi_day_forecast(ticker, days, model_name=model)
        return
    if cmd == 'interactive' and len(tokens) >= 3:
        ticker = tokens[1]
        try:
            days = int(tokens[2])
        except ValueError:
            print("Days must be an integer.")
            return
        model = tokens[3].strip("[]").lower() if len(tokens) >= 4 else 'lstm'
        interactive_plot_forecast(ticker, days, model_name=model)
        return
    if cmd == 'candlestick' and len(tokens) >= 2:
        ticker = tokens[1]
        period = tokens[2] if len(tokens) >= 3 else "1mo"
        interval = tokens[3] if len(tokens) >= 4 else "1d"
        interactive_candlestick_chart(ticker, period=period, interval=interval)
        return
    if cmd == 'info' and len(tokens) == 2:
        stock_info(tokens[1])
        return
    if cmd == 'analyze' and len(tokens) == 2:
        analyze_stock(tokens[1])
        return
    if cmd == 'yahoo' and len(tokens) == 2:
        yahoo_dashboard(tokens[1])
        return
    if cmd == 'probability' and len(tokens) == 2:
        buy_sell_probability(tokens[1], days=30, model='ensemble')
        return
    if cmd == 'context' and len(tokens) >= 2:
        ticker = tokens[1]
        model = tokens[2].lower() if len(tokens) == 3 else 'ensemble'
        try:
            from pandas.tseries.offsets import BDay
        except ImportError:
            pass
        df = download_and_preprocess(ticker)
        if df is None:
            return
        hist = df.tail(6)
        if model == 'arima':
            forecast = forecast_arima(ticker, days=6)
        elif model == 'lstm':
            forecast = forecast_lstm(ticker, days=6)
        elif model == 'gru':
            forecast = forecast_gru(ticker, days=6)
        elif model == 'rf':
            forecast = forecast_rf(ticker, days=6)
        elif model == 'gb':
            forecast = forecast_gb(ticker, days=6)
        elif model in ['ensemble', 'combined']:
            forecast = ensemble_forecast(ticker, days=6)
        else:
            print("Model not recognized.")
            return
        if forecast is None:
            return
        last_date = df.index[-1]
        forecast_dates = pd.date_range(last_date, periods=7, freq=BDay())[1:]
        plt.figure(figsize=(10,6))
        plt.plot(hist.index, hist['Close'].astype(float), marker='o', label='Historical (Last 6 Days)', color='blue')
        plt.plot(forecast_dates, forecast, marker='o', linestyle='--', color='red', label=f'Forecast ({model.upper()})')
        plt.axvline(x=last_date, color='gray', linestyle='--', label='Forecast Start')
        plt.title(f"{ticker.upper()} - Context (Last 6 Days + Next 6 Days Forecast)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()
        return
    if cmd == 'zoom' and len(tokens) >= 2:
        ticker = tokens[1]
        model = tokens[2].lower() if len(tokens) == 3 else 'combined'
        plot_multi_day_forecast(ticker, days=15, model_name=model)
        return
    if cmd == 'chart' and len(tokens) == 2:
        df = download_and_preprocess(tokens[1])
        if df is not None:
            plt.figure(figsize=(12,6))
            plt.plot(df.index, df['Close'], label='Close Price')
            plt.title(f"{tokens[1].upper()} - Raw Ticker Chart")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True)
            plt.show()
        return
    if cmd == 'portfolio' and len(tokens) > 1:
        sub_cmd = tokens[1].lower()
        if sub_cmd == 'add' and len(tokens) == 4:
            try:
                shares = int(tokens[3])
                portfolio_add(tokens[2], shares)
            except ValueError:
                print("Invalid number of shares.")
        elif sub_cmd == 'show':
            portfolio_show()
        elif sub_cmd == 'clear':
            portfolio_clear()
        else:
            print("Invalid portfolio command.")
        return
    if cmd == 'final' and len(tokens) >= 2:
        ticker = tokens[1]
        try:
            days = int(tokens[2])
        except ValueError:
            days = 5
        model = tokens[3].lower() if len(tokens) >= 4 else 'lstm'
        final_summary(ticker, days=days, model=model)
        return
    print("Command not recognized. Type 'help' for a list of commands.")

###############################################################################
# Main Loop (Keeps running until 'exit' or 'quit' is typed)
###############################################################################
current_thread = None

def command_loop():
    global current_thread
    print("Welcome to the Stock Analysis CLI!")
    print("Type 'help' for a list of commands or 'exit' to quit.")
    while True:
        cmd = input(">> ").strip()
        if current_thread and current_thread.is_alive():
            stop_event.set()  # Cancel any running task
        stop_event.clear()
        current_thread = threading.Thread(target=process_command, args=(cmd,))
        current_thread.start()
        if cmd.lower() in ['exit', 'quit']:
            break

if __name__ == "__main__":
    command_loop()