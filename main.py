import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import ta
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    logger.error("BOT_TOKEN o CHAT_ID no definidos en variables de entorno")
    raise SystemExit(1)

# Constants
SYMBOLS = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


class CustomSentimentIntensityAnalyzer(SentimentIntensityAnalyzer):
    """VADER cargando el lexicon local."""

    def __init__(self) -> None:
        lexicon_path = os.path.join(os.path.dirname(__file__), "vader_lexicon.txt")
        super().__init__(lexicon_file=lexicon_path)


def es_ventana_de_noticias() -> bool:
    now = datetime.utcnow()
    return now.hour == 13 and 30 <= now.minute <= 35


def get_news_headlines(ticker: str, num_headlines: int = 3) -> List[str]:
    query = f"{ticker} stock"
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=7)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [
            g.select_one("div.JheGif.nDgy9d").text
            for g in soup.select("div.dbsr")[:num_headlines]
            if g.select_one("div.JheGif.nDgy9d")
        ]
        logger.info("Titulares obtenidos para %s: %s", ticker, headlines)
        return headlines
    except Exception as exc:
        logger.warning("No se pudieron obtener titulares de %s: %s", ticker, exc)
        return []


def analizar_sentimiento_vader(titulares: List[str]) -> str:
    if not titulares:
        return "Sin noticias recientes."
    sia = CustomSentimentIntensityAnalyzer()
    textos = " ".join(titulares)
    scores = sia.polarity_scores(textos)
    compound = scores["compound"]
    sentimiento = (
        "Positivo" if compound >= 0.05 else "Negativo" if compound <= -0.05 else "Neutro"
    )
    resumen = "; ".join(titulares[:2])
    return f"{resumen}\nSentimiento general: {sentimiento}"


def detectar_tendencia(close: pd.Series) -> str:
    if len(close) < 10:
        return "Insuficiente"
    actual = float(close.iloc[-1])
    pasado = float(close.iloc[-10])
    if actual > pasado:
        return "Alcista"
    if actual < pasado:
        return "Bajista"
    return "Lateral"


def encontrar_soporte_resistencia(close: pd.Series, periodo: int = 14) -> Tuple[float, float]:
    soporte = min(close[-periodo:])
    resistencia = max(close[-periodo:])
    return round(soporte, 2), round(resistencia, 2)


def analizar_ticker(ticker: str) -> Optional[dict]:
    logger.info("Analizando %s", ticker)
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
    except Exception as exc:
        logger.warning("Error descargando datos de %s: %s", ticker, exc)
        return None

    if df.empty or len(df) < 20:
        logger.warning("Datos insuficientes para %s", ticker)
        return None

    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(df.columns):
        logger.warning("Columnas faltantes en datos de %s", ticker)
        return None

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    try:
        rsi = ta.momentum.RSIIndicator(close=close).rsi().squeeze()
        macd_obj = ta.trend.MACD(close=close)
        macd = macd_obj.macd().squeeze()
        macd_signal = macd_obj.macd_signal().squeeze()
        sma_50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator().squeeze()
        sma_200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator().squeeze()
        atr = (
            ta.volatility.AverageTrueRange(high=high, low=low, close=close)
            .average_true_range()
            .squeeze()
        )
    except Exception as exc:
        logger.warning("Error calculando indicadores de %s: %s", ticker, exc)
        return None

    try:
        precio = float(close.iloc[-1])
        cierre_ant = float(close.iloc[-2])
        rsi_val = float(rsi.iloc[-1])
        macd_val = float(macd.iloc[-1])
        macd_sig_val = float(macd_signal.iloc[-1])
        sma_50_val = float(sma_50.iloc[-1])
        sma_200_val = float(sma_200.iloc[-1])
        atr_val = float(atr.iloc[-1])
    except Exception as exc:
        logger.warning("Error interpretando indicadores de %s: %s", ticker, exc)
        return None

    senales_bajistas: List[str] = []
    senales_alcistas: List[str] = []
    score_bajista = 0
    score_alcista = 0

    if rsi_val > RSI_OVERBOUGHT:
        senales_bajistas.append("RSI en sobrecompra → posible venta")
        score_bajista += 1
    if macd_val < macd_sig_val:
        senales_bajistas.append("MACD cruzando a la baja")
        score_bajista += 1
    if precio < sma_50_val:
        senales_bajistas.append("Precio por debajo de la SMA50")
        score_bajista += 1
    if sma_50_val < sma_200_val:
        senales_bajistas.append("SMA50 por debajo de SMA200")
        score_bajista += 1
    if precio < cierre_ant:
        senales_bajistas.append("Última vela cerró en rojo")
        score_bajista += 1

    if rsi_val < RSI_OVERSOLD:
        senales_alcistas.append("RSI en sobreventa → posible compra")
        score_alcista += 1
    if macd_val > macd_sig_val:
        senales_alcistas.append("MACD cruzando al alza")
        score_alcista += 1
    if precio < sma_50_val and sma_50_val < sma_200_val:
        senales_alcistas.append("Precio bajo con posible recuperación")
        score_alcista += 1
    if precio > cierre_ant:
        senales_alcistas.append("Última vela cerró en verde")
        score_alcista += 1

    tendencia = detectar_tendencia(close)
    retorno_7d = ((close.iloc[-1] / close.iloc[-7]) - 1) * 100
    volatilidad = np.std(close[-14:]) / close.iloc[-1] * 100
    soporte, resistencia = encontrar_soporte_resistencia(close)

    return {
        "ticker": ticker,
        "precio": round(precio, 2),
        "rsi": round(rsi_val, 2),
        "score_bajista": score_bajista,
        "score_alcista": score_alcista,
        "senales_bajistas": senales_bajistas,
        "senales_alcistas": senales_alcistas,
        "tendencia": tendencia,
        "retorno_7d": round(retorno_7d, 2),
        "volatilidad": round(volatilidad, 2),
        "atr": round(atr_val, 2),
        "soporte": soporte,
        "resistencia": resistencia,
    }


def enviar_mensaje_telegram(token: str, chat_id: str, mensaje: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("Mensaje enviado correctamente a Telegram")
        else:
            logger.error("Error al enviar mensaje: %s", resp.text)
    except Exception as exc:
        logger.error("Excepción al enviar mensaje a Telegram: %s", exc)


def main() -> None:
    resultados = [analizar_ticker(sym) for sym in SYMBOLS]
    resultados = [r for r in resultados if r]

    if not resultados:
        logger.warning("No se pudo analizar ningún activo")
        return

    mejor = max(resultados, key=lambda r: max(r["score_bajista"], r["score_alcista"]))
    tipo = "corto" if mejor["score_bajista"] >= mejor["score_alcista"] else "largo"
    senales = mejor["senales_bajistas"] if tipo == "corto" else mejor["senales_alcistas"]

    if es_ventana_de_noticias():
        titulares = get_news_headlines(mejor["ticker"])
        resumen_noticia = analizar_sentimiento_vader(titulares)
    else:
        resumen_noticia = "Análisis de noticias disponible a las 13:30 UTC."

    sentimiento = (
        resumen_noticia.splitlines()[-1]
        if "Sentimiento general:" in resumen_noticia
        else "No disponible"
    )

    mensaje = f"""
OPORTUNIDAD DESTACADA: {mejor['ticker']}
Precio: {mejor['precio']} USD
RSI: {mejor['rsi']}

Entrada sugerida: en {tipo}

Análisis técnico cualitativo:
- Tendencia general: {mejor['tendencia']}

Datos cuantitativos:
- Volatilidad 14d: {mejor['volatilidad']}%
- Retorno 7d: {mejor['retorno_7d']}%
- ATR: {mejor['atr']}

Señales detectadas:
""" + "\n".join(f"- {s}" for s in senales) + f"""

Soporte: {mejor['soporte']} | Resistencia: {mejor['resistencia']}

Noticias recientes:
{resumen_noticia}

Pregunta para ChatGPT:
Con base en los siguientes datos técnicos para {mejor['ticker']}:
- RSI: {mejor['rsi']}
- Tendencia: {mejor['tendencia']}
- Señales: {', '.join(senales)}
- Sentimiento: {sentimiento}
- Soporte: {mejor['soporte']}, Resistencia: {mejor['resistencia']}
¿Consideras que es una buena oportunidad para entrar en {'compra' if tipo == 'largo' else 'venta'}? Justifica brevemente.
"""

    if len(mensaje) > 4000:
        mensaje = mensaje[:3900] + "\n\n(Mensaje truncado por longitud)"

    enviar_mensaje_telegram(BOT_TOKEN, CHAT_ID, mensaje)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Error inesperado en la ejecución: %s", exc)
