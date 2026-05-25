import pandas as pd
import sqlite3
import asyncio
import json
import requests
import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from ib_insync import IB, util, Contract
import os
from dotenv import load_dotenv


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def get_finbert_sentiment(titles, tokenizer, model):
    if not titles: return 0.0, 0
    inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # FinBERT labels: 0: pos, 1: neg, 2: neut
    sentiments = (probs[:, 0] - probs[:, 1]).numpy()
    return float(sentiments.mean()), len(titles)

def fetch_raw_news_titles(ticker, start_date, end_date, api_key):
    daily_titles = {}
    if not isinstance(start_date, str):
        start_dt = datetime.now() - timedelta(days=7)
    else:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')

    adj_start = (start_dt - timedelta(days=1)).strftime('%Y-%m-%dT16:00:00Z')
    end_date_str = end_date if isinstance(end_date, str) else datetime.now().strftime('%Y-%m-%d')
    
    url = (f"https://api.polygon.io/v2/reference/news?ticker={ticker}"
           f"&published_utc.gte={adj_start}&published_utc.lte={end_date_str}T23:59:59Z"
           f"&limit=1000&apiKey={api_key}")
    
    try:
        while url:
            response = requests.get(url)
            if response.status_code == 429:
                time.sleep(60); continue
            if response.status_code != 200: return None 
            
            data = response.json()
            for article in data.get('results', []):
                dt = datetime.fromisoformat(article['published_utc'].replace('Z', '+00:00'))
                
                # Convert UTC timestamp to Eastern Time for market boundary evaluation
                dt_est = dt.astimezone(ZoneInfo("America/New_York"))
                cutoff_time = dt_est.replace(hour=15, minute=45, second=0, microsecond=0)
                
                # ENFORCE STRICT INTRADAY CUTOFF: Push post-15:45 headlines to T+1
                if dt_est >= cutoff_time:
                    date_str = (dt_est + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    date_str = dt_est.strftime('%Y-%m-%d')
                    
                if article.get('title'):
                    daily_titles.setdefault(date_str, []).append(article['title'])
            
            next_url = data.get('next_url')
            url = f"{next_url}&apiKey={api_key}" if next_url else None
            time.sleep(10)
        return daily_titles
    except:
        return None

async def sync_data():
    load_dotenv()
    poly_key = os.getenv("POLYGON_API_KEY")

    config = load_config()
    db_name = config['settings'].get('db_name', 'trading_vault.db')
    ib_port = config['settings'].get('ibkr_port', 7497)
    
    # ---------------------------------------------------------
    # PHASE 1: IBKR PRICE SYNC & CALENDAR INITIALIZATION
    # ---------------------------------------------------------
    ib = IB()
    try:
        print("[*] PHASE 1: Connecting to IBKR and Initializing Continuous Calendar...")
        await asyncio.wait_for(ib.connectAsync('127.0.0.1', ib_port, clientId=1), timeout=10)
        
        watchlist = [(s, 'STK') for s in config.get('watchlist', [])] + [(s, 'IND') for s in config.get('macro_symbols', [])]
        today_str = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(db_name)

        for symbol, sec_type in watchlist:
            res = pd.read_sql(f"SELECT MAX(date) as last_date FROM ticker_data WHERE ticker='{symbol}'", conn)
            last_date = res.iloc[0,0]
            
            if not last_date:
                last_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            all_dates = pd.date_range(start=last_date, end=today_str).strftime('%Y-%m-%d').tolist()
            for d in all_dates:
                conn.execute('''INSERT OR IGNORE INTO ticker_data (date, ticker, news_volume, avg_sentiment) 
                                VALUES (?, ?, -1, 0.0)''', (d, symbol))
            conn.commit()

            duration_days = (datetime.now() - datetime.strptime(last_date, '%Y-%m-%d')).days + 1
            duration = f"{max(duration_days, 1)} D"
            
            print(f"    [>] Syncing prices for {symbol} ({duration})...")
            contract = Contract(symbol=symbol, secType=sec_type, exchange=('CBOE' if sec_type=='IND' else 'SMART'), currency='USD')
            bars = await ib.reqHistoricalDataAsync(contract, '', duration, '1 day', 'TRADES', True)
            
            if bars:
                df_prices = util.df(bars)
                df_prices['date'] = pd.to_datetime(df_prices['date']).dt.strftime('%Y-%m-%d')
                for _, row in df_prices.iterrows():
                    conn.execute('''UPDATE ticker_data SET open=?, high=?, low=?, close=?, volume=? 
                                    WHERE date=? AND ticker=?''', 
                                 (row['open'], row['high'], row['low'], row['close'], row['volume'], row['date'], symbol))
                conn.commit()

            conn.execute('''
                UPDATE ticker_data SET 
                open = (SELECT close FROM ticker_data b2 WHERE b2.ticker=ticker_data.ticker AND b2.date < ticker_data.date AND b2.close IS NOT NULL ORDER BY b2.date DESC LIMIT 1),
                high = (SELECT close FROM ticker_data b2 WHERE b2.ticker=ticker_data.ticker AND b2.date < ticker_data.date AND b2.close IS NOT NULL ORDER BY b2.date DESC LIMIT 1),
                low = (SELECT close FROM ticker_data b2 WHERE b2.ticker=ticker_data.ticker AND b2.date < ticker_data.date AND b2.close IS NOT NULL ORDER BY b2.date DESC LIMIT 1),
                close = (SELECT close FROM ticker_data b2 WHERE b2.ticker=ticker_data.ticker AND b2.date < ticker_data.date AND b2.close IS NOT NULL ORDER BY b2.date DESC LIMIT 1),
                volume = 0 WHERE ticker=? AND open IS NULL''', (symbol,))
            conn.commit()

        ib.disconnect()
        conn.close()
        print("[*] PHASE 1 Complete: Continuous Calendar & Prices Synchronized.")
        
    except Exception as e:
        print(f"[!!!] IBKR Error: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # PHASE 2: SENTIMENT ENRICHMENT
    # ---------------------------------------------------------
    print("[*] PHASE 2: Loading FinBERT for News Enrichment...")
    tokenizer = AutoTokenizer.from_pretrained("finbert-local", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("finbert-local", local_files_only=True)   
    conn = sqlite3.connect(db_name)

    for symbol, _ in watchlist:
        work_df = pd.read_sql(f"SELECT date FROM ticker_data WHERE ticker='{symbol}' AND news_volume=-1", conn)

        if work_df.empty:
            continue

        print(f"    [*] {symbol}: Processing sentiment for {len(work_df)} calendar days.")
        raw_news = fetch_raw_news_titles(symbol, work_df['date'].min(), work_df['date'].max(), poly_key)

        for d in work_df['date']:
            titles = raw_news.get(d, []) if raw_news else []
            avg, vol = get_finbert_sentiment(titles, tokenizer, model)
            
            conn.execute("UPDATE ticker_data SET avg_sentiment=?, news_volume=? WHERE ticker=? AND date=?",
                         (avg, vol, symbol, d))
        conn.commit()

    conn.close()
    print("[*] All processes complete. Continuous Sequence Ready.")

if __name__ == "__main__":
    asyncio.run(sync_data())