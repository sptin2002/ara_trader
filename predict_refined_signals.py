from datetime import datetime, timedelta
import os
import json
import yaml
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
import qlib
from qlib.data import D
from tensorflow.keras.models import load_model

# Import custom regularized layer to ensure safe model deserialization
from train_refined_model import BilinearGateLayer

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def load_yaml_factors():
    """PHASE 4: Blueprint Integration with Dynamic Structural Tracking"""
    with open('qlib_factor_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    f_cfg = cfg['data_handler_config']['feature_config']
    fields = f_cfg['fields']
    names = f_cfg['names']
    
    # Define execution anchors needed by downstream risk layers
    atomic_atr = "Mean(Greater($high-$low, Greater(Abs($high-Ref($close,1)), Abs($low-Ref($close,1)))), 14)"
    
    # Keep explicit track of what each expression represents
    expressions = [f[0] for f in fields] + ["$close", atomic_atr]
    
    return expressions, list(names)

def get_structural_features(ticker, conn):
    df = pd.read_sql(
        f"SELECT high, low FROM ticker_data WHERE ticker='{ticker.upper()}' ORDER BY date DESC LIMIT 50", conn
    )
    if df.empty:
        return 0, 0, 0, 0

    swing_highs = df['high'] == df['high'].rolling(5, center=True).max()
    swing_lows = df['low'] == df['low'].rolling(5, center=True).min()
    
    nearest_res = df['high'][swing_highs].iloc[0] if any(swing_highs) else df['high'].max()
    nearest_sup = df['low'][swing_lows].iloc[0] if any(swing_lows) else df['low'].min()
    
    res_strength = 2.0 
    sup_strength = 3.5 
    
    return nearest_res, res_strength, nearest_sup, sup_strength

def fetch_and_normalize_sentiment(ticker, conn, lookback_window=30):
    query = f"""
        SELECT date, avg_sentiment, news_volume 
        FROM ticker_data 
        WHERE LOWER(ticker) = LOWER('{ticker}') 
        ORDER BY date DESC LIMIT {lookback_window + 15}
    """
    sent_df = pd.read_sql(query, conn)
    if sent_df.empty or len(sent_df) < lookback_window:
        return None

    sent_df = sent_df.iloc[::-1].reset_index(drop=True)
    sent_df['scaled_sentiment'] = sent_df['avg_sentiment'] * np.log1p(sent_df['news_volume'])
    
    mean_vol = sent_df['news_volume'].rolling(lookback_window).mean()
    std_vol = sent_df['news_volume'].rolling(lookback_window).std().replace(0, 1.0) 
    sent_df['scaled_volume'] = (sent_df['news_volume'] - mean_vol) / std_vol
    
    production_row = sent_df.dropna().iloc[-1]
    return np.array([production_row['scaled_sentiment'], production_row['scaled_volume']], dtype=np.float32)

def run_inference():
    config = load_config()
    qlib.init(provider_uri='qlib_data/bin')
    conn = sqlite3.connect(config['settings']['db_name'])
    
    expressions, core_names = load_yaml_factors()
    win_size = config['model_settings']['lookback_window']
    candidates = []

    end_dt = datetime.now().strftime('%Y-%m-%d')
    start_dt = (datetime.now() - timedelta(days=win_size * 2)).strftime('%Y-%m-%d')
    
    lowercase_watchlist = [s.lower() for s in config['watchlist']]
    
    print(f"[*] Extracting live alpha features from Binary Layer...")
    # Request features from Qlib. Qlib returns a DataFrame where columns are the raw expressions
    all_data = D.features(lowercase_watchlist, expressions, start_time=start_dt, end_time=end_dt)
    all_data.sort_index(inplace=True)

    # Identify the dynamic column indices assigned by Qlib to prevent positional alignment drift
    close_expr = "$close"
    atr_expr = "Mean(Greater($high-$low, Greater(Abs($high-Ref($close,1)), Abs($low-Ref($close,1)))), 14)"

    for ticker in config['watchlist']:
        search_ticker = ticker.lower()
        model_path = f"models/refined/{ticker}_refined.keras"
        if not os.path.exists(model_path): 
            continue

        try:
            if search_ticker not in all_data.index.get_level_values(0):
                print(f"[!] {ticker} missing from binary acceleration maps.")
                continue

            ticker_df = all_data.loc[search_ticker].dropna()
            if len(ticker_df) < win_size: 
                continue

            # Safely load the regularized Keras network artifact
            model = load_model(model_path, custom_objects={"BilinearGateLayer": BilinearGateLayer})
            expected_features = model.input_shape[0][-1]
            
            # Match the input shape to the raw Qlib expressions order
            trained_feature_exprs = expressions[:expected_features]

            # Step 1: Slice Technical Input Tensor using explicit expression names
            tech_input = ticker_df.iloc[-win_size:][trained_feature_exprs].values.reshape(1, win_size, -1)
            
            # Step 2: Extract Context Sentiment Matrices
            news_input = fetch_and_normalize_sentiment(ticker, conn, lookback_window=win_size)
            if news_input is None:
                print(f"[-] {ticker}: Blocked. Insufficient text arrays.")
                continue
            news_input = news_input.reshape(1, 2)

            # Step 3: Compute Model Prediction Probability
            prob = float(model([tech_input, news_input], training=False).numpy()[0][0])

            if prob >= config['thresholds']['min_probability']:
                n_res, r_str, n_sup, s_sup = get_structural_features(ticker, conn)
                
                # Fetch target data records by explicit expression key lookups
                raw_close_price = float(ticker_df.iloc[-1][close_expr])
                raw_atr_value = float(ticker_df.iloc[-1][atr_expr])

                candidates.append({
                    "Ticker": ticker,
                    "Price": round(raw_close_price, 2), 
                    "Confidence": round(prob, 3),
                    "Nearest_Res": round(n_res, 2),
                    "Res_Strength": r_str,
                    "Nearest_Sup": round(n_sup, 2),
                    "Sup_Strength": s_sup,
                    "ATR": round(raw_atr_value, 2)
                })
                print(f"[SIGNAL] {ticker} passed inference classification gate! Prob: {prob:.1%}")
                
        except Exception as e:
            print(f"[Error] Failed inference sequences for {ticker}: {e}")

    if candidates:
        output_df = pd.DataFrame(candidates)
        output_df.to_csv("ml_candidates.csv", index=False)
        print(f"[*] Target selection complete. {len(candidates)} vectors exported to ml_candidates.csv.")
    else:
        if os.path.exists("ml_candidates.csv"):
            os.remove("ml_candidates.csv")
        print("[-] Inference loop finished. 0 candidates matched your alpha threshold profiles.")
        
    conn.close()

if __name__ == "__main__":
    run_inference()