# main.py

import os
import pandas as pd
from config import DATOS_DIR
from load_to_influx import cargar_datos_a_influx

def seleccionar_archivo():
    archivos = [f for f in os.listdir(DATOS_DIR) if f.endswith(".csv") or f.endswith(".json")]
    print("\nArchivos disponibles en /datos:")
    for i, f in enumerate(archivos):
        print(f"{i + 1}. {f}")
    idx = int(input("Selecciona un archivo por número: ")) - 1
    return os.path.join(DATOS_DIR, archivos[idx])

def cargar_archivo(ruta):
    if ruta.endswith(".csv"):
        return pd.read_csv(ruta)
    else:
        return pd.read_json(ruta)

def seleccionar_columnas(df):
    print("\nColumnas encontradas:", list(df.columns))
    tiempo_col = input("Nombre de la columna de tiempo: ")
    campo_col = input("Columna a contar: ")  # Cambié "valor" por "columna a contar"
    tag_col = input("Columna de etiqueta (opcional): ") or None
    df[tiempo_col] = pd.to_datetime(df[tiempo_col])
    
    # Contar las ocurrencias de los valores en la columna seleccionada
    conteo_columna = df[campo_col].value_counts().reset_index()
    conteo_columna.columns = [campo_col, 'conteo']
    
    return conteo_columna, tiempo_col, campo_col, tag_col

if __name__ == "__main__":
    ruta = seleccionar_archivo()
    df = cargar_archivo(ruta)
    conteo_df, tiempo_col, campo_col, tag_col = seleccionar_columnas(df)
    cargar_datos_a_influx(conteo_df, campo_col, 'conteo', tag_col)
