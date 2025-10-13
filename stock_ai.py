import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

# 설정
ticker = "AAPL"
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")  # 오늘 날짜
csv_file = f"{ticker}_data.csv"

# 1. CSV 파일 존재 여부 및 최신성 확인
def load_or_download_data(ticker, start_date, end_date, csv_file):
    if os.path.exists(csv_file):
        try:
            # CSV 로드, 첫 번째 열을 인덱스로 사용
            existing_data = pd.read_csv(csv_file, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
            # 인덱스가 datetime인지 확인
            if not pd.api.types.is_datetime64_any_dtype(existing_data.index):
                existing_data.index = pd.to_datetime(existing_data.index, errors='coerce')
            last_date = existing_data.index.max()
            current_date = pd.to_datetime(end_date)
            
            if pd.notna(last_date) and last_date >= current_date - pd.Timedelta(days=1):
                print(f"{csv_file} 데이터가 최신 상태입니다. 기존 데이터를 사용합니다.")
                return existing_data
        except Exception as e:
            print(f"CSV 파일 로드 중 에러: {e}. 새 데이터를 다운로드합니다...")
    
    print(f"{ticker} 데이터를 다운로드합니다...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    # 멀티-인덱스 처리
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0).str.title()
    else:
        data.columns = data.columns.str.title()
    data.to_csv(csv_file, date_format='%Y-%m-%d')  # CSV로 저장
    print(f"데이터를 {csv_file}에 저장했습니다.")
    return data

# 데이터 로드 또는 다운로드
data = load_or_download_data(ticker, start_date, end_date, csv_file)
# 열 이름 확인
if 'Close' not in data.columns:
    raise KeyError(f"'Close' 열이 데이터에 없습니다. 사용 가능한 열: {data.columns.tolist()}")

# 2. 기술 지표 계산
# SMA (50일, 200일)
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()
data['SMA_Signal'] = np.where(data['SMA50'] > data['SMA200'], 1, -1)  # 1: Bullish, -1: Bearish

# RSI (14일)
def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
data['RSI'] = calculate_rsi(data)
data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))  # 1: Bullish, -1: Bearish

# MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist
data['MACD'], data['Signal_Line'], data['MACD_Hist'] = calculate_macd(data)
data['MACD_Signal'] = np.where((data['MACD'] > data['Signal_Line']) & (data['MACD_Hist'] > 0), 1, -1)  # 1: Bullish, -1: Bearish

# Bollinger Bands (20일)
data['BB_Mid'] = data['Close'].rolling(window=20).mean()
data['BB_Std'] = data['Close'].rolling(window=20).std()
data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']
data['BB_Signal'] = np.where(data['Close'] < data['BB_Lower'], 1, np.where(data['Close'] > data['BB_Upper'], -1, 0))  # 1: Bullish, -1: Bearish

# 열 확인
required_columns = ['Close', 'SMA_Signal', 'RSI_Signal', 'MACD_Signal', 'BB_Signal', 'BB_Lower', 'BB_Upper']
if not all(col in data.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in data.columns]
    raise KeyError(f"필요한 열이 누락되었습니다: {missing_cols}. 사용 가능한 열: {data.columns.tolist()}")

# 3. 불리시/베어리시 지표 개수 계산
data['Bullish_Count'] = (
    (data['SMA_Signal'] == 1).astype(int) +
    (data['RSI_Signal'] == 1).astype(int) +
    (data['MACD_Signal'] == 1).astype(int) +
    (data['BB_Signal'] == 1).astype(int)
)
data['Bearish_Count'] = (
    (data['SMA_Signal'] == -1).astype(int) +
    (data['RSI_Signal'] == -1).astype(int) +
    (data['MACD_Signal'] == -1).astype(int) +
    (data['BB_Signal'] == -1).astype(int)
)

# 4. 30일 롤링 합계
data['Bullish_Sum_30'] = data['Bullish_Count'].rolling(window=30).sum()
data['Bearish_Sum_30'] = data['Bearish_Count'].rolling(window=30).sum()

# 5. 30일 후 가격 변화 (타겟 변수)
data['Price_Change_30'] = data['Close'].shift(-30) - data['Close']
data['Price_Label'] = np.where(data['Price_Change_30'] > 0, 1, 0)  # 1: BUY, 0: SELL

# 6. 상관 관계 분석
correlation_bullish = data['Bullish_Sum_30'].corr(data['Price_Change_30'])
correlation_bearish = data['Bearish_Sum_30'].corr(data['Price_Change_30'])
print(f"30일 불리시 합계와 가격 변화 상관 계수: {correlation_bullish:.3f}")
print(f"30일 베어리시 합계와 가격 변화 상관 계수: {correlation_bearish:.3f}")

# 7. 로지스틱 회귀 모델로 BUY/SELL 확률 예측
features = ['Bullish_Sum_30', 'Bearish_Sum_30']
data = data.dropna()  # 결측값 제거
X = data[features]
y = data['Price_Label']

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 및 정확도
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 테스트 정확도: {accuracy:.3f}")

# 확률 예측
probabilities = model.predict_proba(X)
data['SELL_Prob'] = probabilities[:, 0]
data['BUY_Prob'] = probabilities[:, 1]

# 결과 출력 (최근 5일)
print("\n최근 5일 BUY/SELL 확률:")
print(data[['SELL_Prob', 'BUY_Prob', 'Price_Label']].tail())

# 8. 결과 저장 (기술 지표 및 확률 포함)
data.to_csv(csv_file, date_format='%Y-%m-%d')  # 날짜 형식을 명시적으로 지정
print(f"기술 지표 및 확률을 포함한 데이터를 {csv_file}에 저장했습니다.")

# 9. 차트 생성 (불리시/베어리시 합계)
chart_data = {
  "type": "line",
  "data": {
    "labels": data.index[-30:].strftime('%Y-%m-%d').tolist(),
    "datasets": [
      {
        "label": "30일 불리시 합계",
        "data": data['Bullish_Sum_30'][-30:].tolist(),
        "borderColor": "#4CAF50",
        "fill": False
      },
      {
        "label": "30일 베어리시 합계",
        "data": data['Bearish_Sum_30'][-30:].tolist(),
        "borderColor": "#F44336",
        "fill": False
      }
    ]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": True
      }
    }
  }
}
print("\n불리시/베어리시 합계 차트가 생성되었습니다. (위 JSON은 Chart.js용으로 UI에서 시각화 가능)")