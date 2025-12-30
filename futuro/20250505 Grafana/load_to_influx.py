# load_to_influx.py

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET

def cargar_datos_a_influx(df, campo_col, conteo_col, tag_col=None):
    # Crea el cliente de InfluxDB
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    
    try:
        # Configura la API de escritura
        write_api = client.write_api(write_options=WriteOptions(batch_size=1000, flush_interval=10_000))

        # Escribe cada fila de datos como un punto de InfluxDB
        for _, row in df.iterrows():
            point = Point("conteo").field(conteo_col, int(row[conteo_col])).tag(campo_col, str(row[campo_col]))
            if tag_col:
                point.tag(tag_col, str(row[tag_col]))
            
            # Enviar el punto a InfluxDB
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

        print("✅ Datos de conteo enviados a InfluxDB.")
    except Exception as e:
        print(f"❌ Error al escribir en InfluxDB: {e}")
    finally:
        # Cerrar el cliente para asegurarse de que se finalicen todos los procesos pendientes
        write_api.__del__()
        client.close()

