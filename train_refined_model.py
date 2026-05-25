import os
import json
import yaml
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import qlib
from qlib.data import D
from pathlib import Path
from qlib.constant import REG_US

# ---------------------------------------------------------
# OPTION A COMPLIANT PRODUCTION LAYER: REGULARIZED BILINEAR GATE
# ---------------------------------------------------------
class BilinearGateLayer(layers.Layer):
    """
    Replaces standard linear Concatenation with an L2-regularized multiplicative 
    bilinear interaction. Suppresses extreme non-linear memorization and prevents 
    the network from allocating massive weights to unstable technical-sentiment pairs.
    """
    def __init__(self, output_dim=32, l2_reg=1e-4, **kwargs):
        super(BilinearGateLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.l2_reg = l2_reg

    def build(self, input_shape):
        tech_dim = input_shape[0][-1]
        news_dim = input_shape[1][-1]
        
        # Enforced Option A: Inject L2 Weight Regularization to penalize noise fitting
        self.W = self.add_weight(
            shape=(tech_dim, news_dim, self.output_dim),
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            trainable=True,
            name="bilinear_interaction_kernel"
        )
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name="bilinear_interaction_bias"
        )
        super(BilinearGateLayer, self).build(input_shape)

    def call(self, inputs):
        tech_feat, news_feat = inputs
        # Batch matrix multiplication via Einstein summation contraction: out = tech @ W @ news
        interaction = tf.einsum('bi,ijk,bj->bk', tech_feat, self.W, news_feat)
        return tf.nn.bias_add(interaction, self.b)

    def get_config(self):
        config = super(BilinearGateLayer, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "l2_reg": self.l2_reg
        })
        return config


def verify_infrastructure():
    base_path = Path("qlib_data/bin")
    if not base_path.exists():
        print(f"[ERROR] Infrastructure path missing: {base_path.absolute()}")
        return False
    
    cal_file = base_path / "calendars/day.txt"
    if not cal_file.exists():
        print(f"[ERROR] Calendar missing! Qlib cannot function without: {cal_file.absolute()}")
        return False
        
    folders = [f.name for f in base_path.iterdir() if f.is_dir() and f.name != 'calendars']
    print(f"[*] Infrastructure Verified. Found {len(folders)} tickers in Acceleration Layer.")
    return True

# --- CONFIGURATION & QLIB INIT ---
def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def load_yaml_factors():
    with open('qlib_factor_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    f_cfg = cfg['data_handler_config']['feature_config']
    l_cfg = cfg['data_handler_config']['label_config']
    return [f[0] for f in f_cfg['fields']], f_cfg['names'], l_cfg['fields'][0][0], l_cfg['names'][0]

def init_qlib():
    uri = str(Path("qlib_data/bin").resolve())
    qlib.init(provider_uri=uri, region=REG_US)

def fetch_training_data(ticker, config):
    exprs, names, label_expr, label_name = load_yaml_factors()
    
    try:
        # Isolated data retrieval to fully protect against DataFrame axis length mismatches
        df_features = D.features([ticker.lower()], exprs, freq="day")
        df_label = D.features([ticker.lower()], [label_expr], freq="day")
        
        if df_features is None or df_features.empty or df_label is None or df_label.empty:
            print(f"[!] No data returned from Qlib for {ticker}.")
            return pd.DataFrame(), []

        df_features.reset_index(level=0, drop=True, inplace=True)
        df_features.index = pd.to_datetime(df_features.index).normalize()
        
        df_label.reset_index(level=0, drop=True, inplace=True)
        df_label.index = pd.to_datetime(df_label.index).normalize()

        # Handle any trailing index trackers dynamically
        if len(df_features.columns) == len(names):
            df_features.columns = names
        else:
            df_features.columns = names + [f"extra_col_{i}" for i in range(len(df_features.columns) - len(names))]
            
        df_label.columns = [label_name]
        df = pd.merge(df_features, df_label, left_index=True, right_index=True)
        
        # Sync with Phase 2 normalized text parameters from SQLite
        conn = sqlite3.connect(config['settings']['db_name'])
        sent_df = pd.read_sql(
            f"SELECT date, avg_sentiment, news_volume FROM ticker_data WHERE LOWER(ticker)=LOWER('{ticker}')", conn
        )
        conn.close()
        
        if sent_df.empty:
            return pd.DataFrame(), []
        
        sent_df['date'] = pd.to_datetime(sent_df['date']).dt.normalize()
        sent_df.set_index('date', inplace=True)

        full_df = df.merge(sent_df, left_index=True, right_index=True).sort_index()

        target_gain = config['model_settings'].get('target_gain_pct', 0.015)
        lookahead = config['model_settings'].get('lookahead_days', 5)
        full_df['target_bin'] = (full_df[label_name].shift(-lookahead) > target_gain).astype(int)

        actual_tech_cols = [c for c in df_features.columns if not c.startswith('extra_col')]
        return full_df.dropna(), actual_tech_cols
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), []

def prepare_sequences(df, tech_cols, win_size):
    news_cols = ['avg_sentiment', 'news_volume']
    X_tech, X_news, y = [], [], []
    for i in range(len(df) - win_size):
        X_tech.append(df[tech_cols].iloc[i:i+win_size].values)
        X_news.append(df[news_cols].iloc[i+win_size-1].values)
        y.append(df['target_bin'].iloc[i+win_size-1])
    return np.array(X_tech), np.array(X_news), np.array(y)

# --- TFT MODEL DESIGN WITH REGULARIZED INTERACTION GATE ---
def build_tft_production_model(input_shape_tech, input_shape_news):
    tech_in = layers.Input(shape=input_shape_tech, name="tech_input")
    x = layers.TimeDistributed(layers.Dense(64))(tech_in) 
    x = layers.LayerNormalization()(x)
    
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    
    x = layers.GRU(32, return_sequences=False)(x) 
    
    news_in = layers.Input(shape=input_shape_news, name="news_input")
    y = layers.Dense(16, activation='relu')(news_in)
    
    # Bilinear Gate with explicit L2 Penalties active to limit memory capacity
    gated_interaction = BilinearGateLayer(output_dim=32, l2_reg=1e-4, name="bilinear_gate")([x, y])
    gated_interaction = layers.LayerNormalization()(gated_interaction)
    
    z = layers.Dense(32, activation='relu')(gated_interaction)
    out = layers.Dense(1, activation='sigmoid')(z)
    
    model = models.Model(inputs=[tech_in, news_in], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_system():
    config = load_config()
    if not verify_infrastructure():
        return
        
    init_qlib()
    win_size = config['model_settings']['lookback_window']
    validation_split = config['model_validation'].get('validation_split', 0.2)
    
    os.makedirs("models/refined", exist_ok=True)

    try:
        available = D.list_instruments(D.instruments('all'))
        print(f"[*] Brain linked to symbols: {list(available.keys())[:5]}")

        for ticker in config['watchlist']:
            print(f"[*] Training Refinement Gate for {ticker}...")
            data, tech_names = fetch_training_data(ticker, config)
            
            if len(data) < (win_size + 50): 
                print(f"[-] {ticker}: Insufficient data ({len(data)} rows). Skipping.")
                continue
            
            X_t, X_n, y = prepare_sequences(data, tech_names, win_size)
            split = int(len(X_t) * (1 - validation_split))
            
            model = build_tft_production_model((win_size, len(tech_names)), (2,))
            stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = model.fit(
                [X_t[:split], X_n[:split]], y[:split],
                validation_data=([X_t[split:], X_n[split:]], y[split:]),
                epochs=config['model_settings']['training']['epochs'], 
                batch_size=config['model_settings']['training']['batch_size'],
                callbacks=[stop_early],
                verbose=0
            )

            # PRODUCTION REFINEMENT GATE MONITORING
            # OPTION C UPGRADE: Extract values relative to the absolute lowest validation loss point
            best_epoch_idx = np.argmin(history.history['val_loss'])
            val_acc = history.history['val_accuracy'][best_epoch_idx]
            train_acc = history.history['accuracy'][best_epoch_idx]
            delta = abs(train_acc - val_acc)

            min_acc = config['model_validation']['min_train_accuracy']   
            max_delta = config['model_validation']['max_delta_threshold'] 

            if val_acc >= min_acc and delta <= max_delta:
                model.save(f"models/refined/{ticker}_refined.keras")
                print(f"[OK] {ticker} Saved. Val Acc: {val_acc:.2%}, Delta: {delta:.2%}")
            else:
                print(f"[!] {ticker} REJECTED BY GUARDRAIL. Val Acc: {val_acc:.2%}, Delta: {delta:.2%}")
            
    except Exception as e:
        print(f"[!!!] Error training system loop: {e}")

if __name__ == "__main__":
    train_system()