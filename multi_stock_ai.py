import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from datetime import datetime
import json

# 설정
tickers = ["AAPL", "TSLA", "GOOG"]  # 분석할 종목
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
data_dir = "stock_data"
os.makedirs(data_dir, exist_ok=True)

# 1. 데이터 로드 또는 다운로드
def load_or_download_data(ticker, start_date, end_date, csv_file):
    if os.path.exists(csv_file):
        try:
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            last_date = data.index.max()
            if pd.notna(last_date) and last_date >= pd.to_datetime(end_date) - pd.Timedelta(days=1):
                print(f"{csv_file} is up-to-date.")
                return data
        except Exception as e:
            print(f"Error loading {csv_file}: {e}. Downloading new data...")
    
    print(f"Downloading {ticker} data...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data.columns = [col.title() for col in data.columns]
    data.to_csv(csv_file)
    return data

# 2. 기술 지표 계산 및 백테스트
def analyze_stock(ticker):
    csv_file = os.path.join(data_dir, f"{ticker}_data.csv")
    data = load_or_download_data(ticker, start_date, end_date, csv_file)
    
    # 기술 지표 계산
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['SMA_Signal'] = np.where(data['SMA50'] > data['SMA200'], 1, -1)

    def calculate_rsi(data, periods=14):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    data['RSI'] = calculate_rsi(data)
    data['RSI_Signal'] = np.where(data['RSI'] < 40, 2.0, np.where(data['RSI'] > 60, -2.0, 0))

    def calculate_macd(data, fast=12, slow=26, signal=9):
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist
    data['MACD'], data['Signal_Line'], data['MACD_Hist'] = calculate_macd(data)
    data['MACD_Signal'] = np.where((data['MACD'] > data['Signal_Line']) & (data['MACD_Hist'] > 0), 1, -1)

    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']
    data['BB_Signal'] = np.where(data['Close'] < data['BB_Lower'], 1.5, np.where(data['Close'] > data['BB_Upper'], -1.5, 0))

    # 거래량 기반 신호
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Signal'] = np.where(data['Volume'] > data['Volume_MA'] * 1.5, 1, 0)

    # Stochastic Oscillator
    data['L14'] = data['Low'].rolling(window=14).min()
    data['H14'] = data['High'].rolling(window=14).max()
    data['%K'] = 100 * (data['Close'] - data['L14']) / (data['H14'] - data['L14'])
    data['%D'] = data['%K'].rolling(window=3).mean()
    data['Stochastic_Signal'] = np.where(data['%K'] < 30, 1.5, np.where(data['%K'] > 70, -1.5, 0))

    # ATR 계산
    data['TR'] = np.maximum(data['High'] - data['Low'], 
                            np.maximum(abs(data['High'] - data['Close'].shift()), 
                                       abs(data['Low'] - data['Close'].shift())))
    data['ATR'] = data['TR'].rolling(window=14).mean()

    # OBV 계산
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    data['OBV_Signal'] = np.where(data['OBV'] > data['OBV'].rolling(window=20).mean(), 1, -1)

    # 단기 가격 변화율
    data['Return_3'] = data['Close'].pct_change(periods=3)

    # 불리시/베어리시 카운트 (5일 롤링)
    data['Bullish_Count'] = (
        (data['SMA_Signal'] == 1).astype(int) +
        data['RSI_Signal'] +
        (data['MACD_Signal'] == 1).astype(int) +
        data['BB_Signal'] +
        data['Volume_Signal'] +
        data['Stochastic_Signal'] +
        (data['OBV_Signal'] == 1).astype(int)
    )
    data['Bearish_Count'] = (
        (data['SMA_Signal'] == -1).astype(int) +
        data['RSI_Signal'] +
        (data['MACD_Signal'] == -1).astype(int) +
        data['BB_Signal'] +
        data['Stochastic_Signal'] +
        (data['OBV_Signal'] == -1).astype(int)
    )
    data['Bullish_Sum_5'] = data['Bullish_Count'].rolling(window=5).sum()
    data['Bearish_Sum_5'] = data['Bearish_Count'].rolling(window=5).sum()

    # 5일 후 가격 변화
    data['Price_Change_5'] = data['Close'].shift(-5) - data['Close']
    data['Price_Label'] = np.where(data['Price_Change_5'] > 0, 1, 0)

    # XGBoost
    features = ['Bullish_Sum_5', 'Bearish_Sum_5', 'RSI', '%K', 'Volume', 'ATR', 'OBV', 'Return_3']
    data = data.dropna()
    X = data[features]
    y = data['Price_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    probabilities = model.predict_proba(X)
    data['SELL_Prob'] = probabilities[:, 0]
    data['BUY_Prob'] = probabilities[:, 1]

    # 백테스트 (스탑로스 및 트레일링 스탑)
    initial_cash = 100000
    cash = initial_cash
    shares = 0
    trades = []
    buy_price = 0
    max_price = 0
    for date, row in data.iterrows():
        if row['BUY_Prob'] > 0.75 and cash > row['Close']:
            shares_to_buy = int(cash // row['Close'])
            cash -= shares_to_buy * row['Close']
            shares += shares_to_buy
            buy_price = row['Close']
            max_price = buy_price
            trades.append({'date': date, 'type': 'BUY', 'price': row['Close'], 'shares': shares_to_buy})
        elif shares > 0:
            max_price = max(max_price, row['Close'])
            # 스탑로스: 5% 하락, 트레일링 스탑: 최고점에서 3% 하락
            if row['SELL_Prob'] > 0.75 or row['Close'] < buy_price * 0.95 or row['Close'] < max_price * 0.97:
                cash += shares * row['Close']
                trades.append({'date': date, 'type': 'SELL', 'price': row['Close'], 'shares': shares})
                shares = 0
                buy_price = 0
                max_price = 0
    final_value = cash + shares * data['Close'].iloc[-1]
    return_ = (final_value - initial_cash) / initial_cash * 100

    # 결과 저장
    data.to_csv(csv_file)

    # 차트 데이터
    chart_data = {
        "type": "line",
        "data": {
            "labels": data.index[-30:].strftime('%Y-%m-%d').tolist(),
            "datasets": [
                {"label": "Close Price", "data": data['Close'][-30:].tolist(), "borderColor": "#3b82f6", "fill": False},
                {"label": "BUY Prob", "data": data['BUY_Prob'][-30:].tolist(), "borderColor": "#22c55e", "fill": False},
                {"label": "SELL Prob", "data": data['SELL_Prob'][-30:].tolist(), "borderColor": "#ef4444", "fill": False}
            ]
        },
        "options": {"scales": {"y": {"beginAtZero": False}}}
    }

    return {
        'ticker': ticker,
        'accuracy': accuracy,
        'return': return_,
        'recent_probs': data[['Close', 'SELL_Prob', 'BUY_Prob']].tail(5),
        'trades': trades[-5:],  # 최근 5개 거래 내역
        'chart_data': chart_data
    }

# 3. 여러 종목 분석
results = []
for ticker in tickers:
    result = analyze_stock(ticker)
    results.append(result)

# 4. 요약 테이블
summary_df = pd.DataFrame([
    {
        '종목': r['ticker'],
        '모델 정확도': round(r['accuracy'], 3),
        '수익률 (%)': round(r['return'], 2)
    } for r in results
])
print("\n종목별 요약 테이블:")
print(summary_df.to_string(index=False))

# 5. AAPL 최근 5일 결과
print("\nAAPL 최근 5일 BUY/SELL 확률:")
print(results[0]['recent_probs'])

# 6. AAPL 최근 5개 거래 내역
print("\nAAPL 최근 5개 거래 내역:")
print(pd.DataFrame(results[0]['trades']).to_string(index=False))

# 7. AAPL 차트 JSON
print("\nAAPL 차트 JSON (Chart.js용):")
print(json.dumps(results[0]['chart_data'], indent=2))