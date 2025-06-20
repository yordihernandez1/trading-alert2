import os
import io
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from telegram import Bot

# Inicializar analizador de sentimientos VADER
from nltk.data import path as nltk_path
nltk_path.append('./')
sid = SentimentIntensityAnalyzer(lexicon_file='vader_lexicon.txt')

# Parámetros generales
SYMBOLS = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'META']
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_MIN_RR = 1.33
VOLATILITY_THRESHOLD = 0.015
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")
SUMMARY_FILE = "last_alert.txt"

def analyze_daily(symbol):
    df = yf.download(symbol, period='6mo', interval='1d')
    df = df.dropna()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    last = df.iloc[-1]
    return {
        'symbol': symbol,
        'close': last['Close'],
        'rsi': last['RSI'],
        'macd': last['MACD'],
        'macd_signal': last['Signal'],
        'sma50': last['SMA50'],
        'sma200': last['SMA200'],
        'atr': last['ATR'],
        'trend': 'up' if last['SMA50'] > last['SMA200'] else 'down',
        'overbought': last['RSI'] > 70,
        'oversold': last['RSI'] < 30
    }

def analyze_intraday(symbol):
    df = yf.download(symbol, period='5d', interval='5m')
    df = df.dropna()
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['VolumeMean'] = df['Volume'].rolling(window=20).mean()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    return {
        'crossover': prev['EMA9'] < prev['EMA21'] and last['EMA9'] > last['EMA21'],
        'rsi_extreme': last['RSI'] < 30 or last['RSI'] > 70,
        'volume_spike': last['Volume'] > 1.5 * last['VolumeMean'],
        'df': df.tail(100)
    }

def analyze_sentiment(symbol):
    url = f"https://www.bing.com/news/search?q={symbol}+stock"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    headlines = [a.get_text() for a in soup.find_all('a') if a.get_text()]
    scores = [sid.polarity_scores(h)['compound'] for h in headlines[:5]]
    avg = np.mean(scores) if scores else 0
    return 'positivo' if avg > 0.2 else 'negativo' if avg < -0.2 else 'neutro'

def plot_chart(df, symbol):
    plt.figure(figsize=(10, 4))
    plt.plot(df['Close'], label='Close')
    plt.plot(df['EMA9'], label='EMA9')
    plt.plot(df['EMA21'], label='EMA21')
    plt.legend()
    plt.title(f'{symbol} Intradía')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def send_telegram_alert(text, image=None):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    if image:
        bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=image, caption=text)
    else:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)

def time_since_last_alert():
    if not os.path.exists(SUMMARY_FILE):
        return float('inf')
    with open(SUMMARY_FILE, 'r') as f:
        last = float(f.read().strip())
    return (datetime.datetime.now().timestamp() - last) / 60  # en minutos

def update_last_alert_time():
    with open(SUMMARY_FILE, 'w') as f:
        f.write(str(datetime.datetime.now().timestamp()))

def run_alert_bot():
    best = None
    for symbol in SYMBOLS:
        daily = analyze_daily(symbol)
        entry = daily['close']
        stop = entry * (1 - STOP_LOSS_PCT)
        target = entry + abs(entry - stop) * TAKE_PROFIT_MIN_RR
        atr_pct = daily['atr'] / daily['close']
        if atr_pct > VOLATILITY_THRESHOLD:
            continue
        intraday = analyze_intraday(symbol)
        if not (intraday['crossover'] or intraday['rsi_extreme'] or intraday['volume_spike']):
            continue
        rr = (target - entry) / (entry - stop)
        sentiment = analyze_sentiment(symbol)
        chart = plot_chart(intraday['df'], symbol)
        text = (
            f"📈 *Alerta de Trading: {symbol}*
"
            f"Precio: {entry:.2f}
"
            f"SL: {stop:.2f} | TP: {target:.2f}
"
            f"Riesgo/Recompensa: {rr:.2f}
"
            f"RSI: {daily['rsi']:.1f} | MACD: {daily['macd']:.2f}
"
            f"Tendencia: {daily['trend']}
"
            f"Sentimiento de noticias: {sentiment}"
        )
        send_telegram_alert(text, chart)
        update_last_alert_time()
        return  # Solo envía la mejor oportunidad

    # Si no hay alertas, enviar resumen cada 30 min
    if time_since_last_alert() >= 30:
        text = f"⏳ No se detectaron señales en los últimos 30 minutos. ({datetime.datetime.now().strftime('%H:%M')})"
        send_telegram_alert(text)
        update_last_alert_time()
