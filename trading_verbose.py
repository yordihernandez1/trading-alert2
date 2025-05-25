import os
import yfinance as yf
import ta
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from pathlib import Path

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

SYMBOLS = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30
UMBRAL_ALERTA = 50
LOG_ALERTA = "ultima_alerta.json"
TIEMPO_RESUMEN_MINUTOS = 30

def registrar_alerta():
    try:
        print("📝 Registrando alerta en ultima_alerta.json...")
        with open(LOG_ALERTA, "w") as f:
            json.dump({"ultima": datetime.utcnow().isoformat()}, f)
    except Exception as e:
        print(f"⚠️ No se pudo registrar la alerta: {e}")

def tiempo_desde_ultima_alerta():
    path = Path(LOG_ALERTA)
    if not path.exists():
        return 9999
    try:
        with open(path, "r") as f:
            data = json.load(f)
        ultima = datetime.fromisoformat(data.get("ultima", "2000-01-01T00:00:00"))
        delta = (datetime.utcnow() - ultima).total_seconds() / 60
        print(f"⏱ Última alerta enviada hace {int(delta)} minutos.")
        return delta
    except Exception as e:
        print(f"⚠️ Error leyendo archivo de alerta: {e}")
        return 9999

# (Resto del script se mantiene igual: análisis técnico, envío de mensaje, etc.)

# Para simplificar, en esta respuesta sólo mostramos la parte agregada que garantiza el resumen.

def encontrar_soporte_resistencia(close, periodo=14):
    soporte = min(close[-periodo:])
    resistencia = max(close[-periodo:])
    return round(soporte, 2), round(resistencia, 2)


