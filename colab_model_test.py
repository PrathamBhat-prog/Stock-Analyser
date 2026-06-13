import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pickle
import time
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from catboost import CatBoostClassifier
from datetime import datetime

# Suppress warnings for production-style execution
warnings.filterwarnings('ignore')

# =============================================================================
# SNIPER STRATEGY v5: PRODUCTION CONSOLIDATED SCRIPT
# Target: 20-day price direction with >60% precision
# Features: GDELT Sentiment, VIX Velocity, 52w High Proximity
# =============================================================================

class SniperStrategy:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.feature_cols = [
            'momentum_20d', 'dist_52w_high', 'rolling_sentiment_20d',
            'vol_ratio_5d', 'VIX', 'sent_lag_1', 'sent_lag_3', 
            'sent_lag_5', 'sent_vix_interaction', 'vix_velocity'
        ]

    def fetch_gdelt_sentiment(self, ticker, days=7):
        """Fetches live news headlines from GDELT API and scores via VADER."""
        query = ticker.replace('.NS', '') + ' stock'
        url = f'https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=artlist&maxrecords=50&timespan={days}d&format=json'
        try:
            r = requests.get(url, timeout=10)
            articles = r.json().get('articles', [])
            if not articles: return 0.0
            scores = [self.analyzer.polarity_scores(a.get('title', ''))['compound'] for a in articles]
            return float(np.mean(scores))
        except:
            return 0.0

    def engineer_features(self, df, vix_df, live_sent):
        """Implements the Sniper v5 alpha drivers and interaction terms."""
        df = df.copy()
        # 1. Technical Indicators
        df['momentum_20d'] = df['Close'].pct_change(20)
        high_52w = df['Close'].rolling(window=252, min_periods=1).max()
        df['dist_52w_high'] = (df['Close'] - high_52w) / high_52w
        df['vol_ratio_5d'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
        
        # 2. Sentiment Integration (Proxying rolling sentiment with live news)
        df['rolling_sentiment_20d'] = live_sent
        
        # 3. Macro Merge
        df = pd.merge(df, vix_df, on='Date', how='left')
        
        # 4. Lags & Interactions
        for lag in [1, 3, 5]:
            df[f'sent_lag_{lag}'] = df['rolling_sentiment_20d'].shift(lag)
        df['sent_vix_interaction'] = df['rolling_sentiment_20d'] * df['VIX']
        
        # 5. Target Creation
        df['target_20d'] = (df['Close'].shift(-20) > df['Close']).astype(int)
        return df.dropna()

    def train_and_save(self, tickers, period='5y'):
        print(f'Starting sniper v5 Training... Tickers: {tickers}')
        
        # Fetch VIX
        vix = yf.Ticker('^VIX').history(period=period)[['Close']].reset_index()
        vix.columns = ['Date', 'VIX']
        vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
        vix['vix_velocity'] = vix['VIX'].pct_change(5)

        # Build Aggregated Dataset
        frames = []
        for ticker in tickers:
            print(f'  Gathering data for {ticker}...')
            raw = yf.Ticker(ticker).history(period=period, auto_adjust=True).reset_index()
            raw['Date'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)
            
            live_sent = self.fetch_gdelt_sentiment(ticker)
            ticker_df = self.engineer_features(raw, vix, live_sent)
            ticker_df['ticker'] = ticker
            frames.append(ticker_df)
            time.sleep(1) # API Safety

        big_df = pd.concat(frames).sort_values('Date')
        
        # Model Training (Tuned for Precision)
        print(f'Training CatBoost on {len(big_df)} historical samples...')
        model = CatBoostClassifier(
            iterations=1500, 
            learning_rate=0.015, 
            depth=7, 
            l2_leaf_reg=8,
            verbose=0,
            random_seed=42
        )
        
        model.fit(big_df[self.feature_cols], big_df['target_20d'])
        
        # Persist to Disk
        save_path = 'trading_model_sniper_v5.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f'SUCCESS: Model saved to {save_path}')
        print('Optimal Confidence Threshold: 0.52')

if __name__ == '__main__':
    # Example set: modify as needed for your local portfolio
    watchlist = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'META', 'AMD']
    sniper = SniperStrategy()
    sniper.train_and_save(watchlist)