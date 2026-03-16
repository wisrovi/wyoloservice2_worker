# create_dashboard.py

import requests
import json
from config import GRAFANA_URL, GRAFANA_API_KEY, DATASOURCE_UID

def crear_dashboard():
    # URL para la API de Grafana
    url = f"{GRAFANA_URL}/api/dashboards/db"

    # El cuerpo de la solicitud (JSON) para crear el dashboard
    dashboard = {
        "dashboard": {
            "id": None,
            "uid": None,
            "title": "Dashboard de Conteo de Frecuencia",
            "tags": ["conteo", "frecuencia"],
            "timezone": "browser",
            "panels": [
                {
                    "type": "graph",
                    "title": "Conteo de Workers",
                    "datasource": {
                        "type": "influxdb",
                        "uid": DATASOURCE_UID
                    },
                    "targets": [
                        {
                            "refId": "A",
                            "measurement": "conteo",
                            "field": "conteo",
                            "groupBy": [{"type": "time", "params": ["5m"]}],
                            "select": [["mean", "conteo"]],
                        }
                    ],
                    "legend": {
                        "show": True,
                        "values": ["min", "max", "current"],
                    },
                }
            ],
        },
        "overwrite": True  # Si ya existe un dashboard con el mismo nombre, se sobrescribe
    }

    # Cabeceras para la autenticación
    headers = {
        "Authorization": f"Bearer {GRAFANA_API_KEY}",
        "Content-Type": "application/json"
    }

    # Hacer la solicitud POST a la API de Grafana para crear el dashboard
    response = requests.post(url, headers=headers, data=json.dumps(dashboard))

    # Imprimir detalles de la respuesta
    if response.status_code == 200:
        print("✅ Dashboard creado exitosamente en Grafana.")
        print("Detalles de la respuesta:", response.json())  # Mostrar detalles de la respuesta
    else:
        print(f"❌ Error al crear el dashboard: {response.status_code} - {response.text}")

if __name__ == "__main__":
    crear_dashboard()
