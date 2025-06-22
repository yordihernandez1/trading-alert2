import os
import datetime
from alerts import run_alert_bot

if __name__ == "__main__":
    print(f"[{datetime.datetime.now()}] Ejecutando bot de alertas de trading...")
    try:
        run_alert_bot()
    except Exception as e:
        print(f"Error al ejecutar el bot: {e}")
