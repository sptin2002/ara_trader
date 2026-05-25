import sqlite3
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

def date_to_qlib_int(dt_series):
    """Convert datetime series to Qlib integer format YYYYMMDD"""
    return pd.to_datetime(dt_series).dt.strftime("%Y%m%d").astype(int)

def calculate_rolling_zscore(df, column, window=30):
    """
    Calculates the rolling Z-Score of a column per ticker to ensure 
    scale invariance across different assets on the watchlist.
    """
    # Group by ticker and apply a rolling mean and std
    rolling_mean = df.groupby('ticker')[column].transform(lambda x: x.rolling(window, min_periods=5).mean())
    rolling_std = df.groupby('ticker')[column].transform(lambda x: x.rolling(window, min_periods=5).std())
    
    # Avoid division by zero anomalies
    z_score = (df[column] - rolling_mean) / (rolling_std + 1e-8)
    return z_score.fillna(0.0)

def bridge_sqlite_to_qlib():
    db_path = 'trading_vault.db'
    provider_uri = Path("qlib_data/bin")
    freq = "day"
    
    # 1. Reset Acceleration Layer Directory
    if provider_uri.exists():
        shutil.rmtree(provider_uri)
    provider_uri.mkdir(parents=True)

    # 2. Load Data from SQLite Source of Truth
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM ticker_data ORDER BY date ASC", conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek # 5 = Saturday, 6 = Sunday

    # ---------------------------------------------------------
    # PHASE 1 MECHANICS: WEEKEND COMPRESSION
    # ---------------------------------------------------------
    df['target_date'] = df['date']
    df.loc[df['day_of_week'] == 5, 'target_date'] = df['date'] + pd.Timedelta(days=2) # Sat -> Mon
    df.loc[df['day_of_week'] == 6, 'target_date'] = df['date'] + pd.Timedelta(days=1) # Sun -> Mon

    # Compute a volume-weighted product array to properly merge averages
    df['weighted_sentiment'] = df['avg_sentiment'] * df['news_volume']

    # Compress news data grouped by ticker and target trading date
    news_aggregation = df.groupby(['ticker', 'target_date']).agg(
        total_volume=('news_volume', 'sum'),
        total_weighted_sent=('weighted_sentiment', 'sum')
    ).reset_index()

    # Re-calculate correct mathematically blended average sentiment
    news_aggregation['blended_sentiment'] = np.where(
        news_aggregation['total_volume'] > 0,
        news_aggregation['total_weighted_sent'] / news_aggregation['total_volume'],
        0.0
    )

    # Filter original data down to true market trading sessions (Monday - Friday)
    trading_sessions_df = df[df['day_of_week'] < 5].copy()
    trading_sessions_df.drop(columns=['avg_sentiment', 'news_volume', 'weighted_sentiment', 'day_of_week', 'target_date'], inplace=True)

    # Merge aggregated weekend information back into Monday trading rows
    final_processed_df = trading_sessions_df.merge(
        news_aggregation,
        left_on=['ticker', 'date'],
        right_on=['ticker', 'target_date'],
        how='left'
    )

    # Fill empty states
    final_processed_df['avg_sentiment'] = final_processed_df['blended_sentiment'].fillna(0.0)
    final_processed_df['news_volume'] = final_processed_df['total_volume'].fillna(0).astype(int)
    
    # Drop intermediate grouping columns
    df = final_processed_df.drop(columns=['target_date', 'total_volume', 'total_weighted_sent', 'blended_sentiment']).copy()
    df.sort_values(by=['ticker', 'date'], inplace=True)

    # ---------------------------------------------------------
    # PHASE 2 MECHANICS: FEATURE ENGINEERING & NORMALIZATION
    # ---------------------------------------------------------
    # 1. Logarithmic Volume Weighting to scale down high-volume outliers gently
    df['log_news_volume'] = np.log1p(df['news_volume'])

    # 2. Mathematical Fusion Factor (Volume Weighted Sentiment Velocity)
    # Scales the raw average sentiment score by the log of the attention it received
    df['vol_weighted_sentiment'] = df['avg_sentiment'] * df['log_news_volume']

    # 3. Scale Invariance via Rolling 30-Day Cross-Sectional Z-Scores per Ticker
    # This centers the features around a localized moving average, eliminating scale variance
    df['scaled_sentiment'] = calculate_rolling_zscore(df, 'vol_weighted_sentiment', window=30)
    df['scaled_volume'] = calculate_rolling_zscore(df, 'log_news_volume', window=30)

    # ---------------------------------------------------------
    # QLIB EXPORT MECHANICS
    # ---------------------------------------------------------
    # 3. Calendar Export: provider_uri/calendars/day.txt (Trading Days Only)
    cal_dir = provider_uri / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    unique_dates = df['date'].drop_duplicates().sort_values()
    calendar_ints = date_to_qlib_int(unique_dates)
    calendar_ints.to_csv(cal_dir / f"{freq}.txt", index=False, header=False)

    # 4. Instruments Export: provider_uri/instruments/all.txt
    inst_dir = provider_uri / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    tickers = df['ticker'].str.lower().unique()
    with open(inst_dir / "all.txt", "w") as f:
        for ticker in tickers:
            ticker_df = df[df['ticker'].str.lower() == ticker]
            start_int = date_to_qlib_int(pd.Series([ticker_df['date'].min()])).iloc[0]
            end_int = date_to_qlib_int(pd.Series([ticker_df['date'].max()])).iloc[0]
            f.write(f"{ticker}\t{start_int}\t{end_int}\n")

    # 5. Map Dates to Clean Calendar Indices
    cal_map = {date_int: idx for idx, date_int in enumerate(calendar_ints)}

    # 6. Feature Generation: Export transformed normalized attributes as Qlib Binary Fields
    features_dir = provider_uri / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # We now pass the highly scaled, scale-invariant attributes to the neural networks
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'scaled_sentiment', 'scaled_volume']

    for ticker in tickers:
        symbol_path = features_dir / ticker
        symbol_path.mkdir(parents=True, exist_ok=True)
        symbol_df = df[df['ticker'].str.lower() == ticker].sort_values('date')
        
        current_dates_int = date_to_qlib_int(symbol_df['date'])
        indices = np.array([cal_map[d] for d in current_dates_int], dtype=np.int64)
        
        full_calendar_size = len(calendar_ints)
        for col in feature_cols:
            full_data = np.full(full_calendar_size, np.nan, dtype=np.float32)
            col_values = symbol_df[col].values.astype(np.float32)
            full_data[indices] = col_values

            # Qlib requires a leading 64-bit float configuration offset start index (0.0)
            storage_array = np.hstack(([0.0], full_data)).astype('<f4')
            with open(symbol_path / f"{col}.{freq}.bin", "wb") as f:
                f.write(storage_array.tobytes())

    print(f"[✓] Phase 2 Complete: Qlib data ready at {provider_uri.absolute()}")
    print(f"    Features generated: {feature_cols}")

if __name__ == "__main__":
    bridge_sqlite_to_qlib()