# config.py (extendido, solo si usas la API de Grafana)

# InfluxDB
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "YOUR_INFLUX_TOKEN_HERE"
INFLUX_ORG = "AI"
INFLUX_BUCKET = "AI"
DATOS_DIR = "./datos"

# Grafana (solo si automatizas paneles)
GRAFANA_URL = "http://localhost:3000"
GRAFANA_API_TOKEN = "YOUR_GRAFANA_API_TOKEN_HERE"
DATASOURCE_UID = "bel98xu67a3nka"  # UID del datasource InfluxDB en Grafana
