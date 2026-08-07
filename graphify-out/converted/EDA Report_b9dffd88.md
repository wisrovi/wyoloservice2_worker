<!-- converted from EDA Report.docx -->

Análisis Exploratorio de Datos (EDA) Segmentación
# Informe de Análisis Exploratorio de Datos

Autor: "Manu G"
Proyecto: "example_segmentation"
Modelo: Segmentación
Descripción: "Este es un entrenamiento de prueba de manuel"
Fecha: 10/04/2025 13:19

# Índice
1. Distribución de Clases
2. Distribución de Tamaños de Imágenes
3. Distribución de Áreas de Bounding Boxes
4. Distribución de Relaciones de Aspecto
5. Distribución de Relaciones de Aspecto de Bounding Boxes
6. Distribución de Posiciones Centrales de Bounding Boxes
7. Gráfico de Dispersión de Ancho vs. Alto de Bounding Boxes
8. Mosaicos de Imágenes de Ejemplo
9. Resultados de Validación con Ultralytics
10. Validación de Calidad de Imágenes
11. Conclusiones del Análisis EDA


# Distribución de Clases:


# Distribución de Tamaños de Imágenes:


# Distribución de Áreas de Bounding Boxes:


# Distribución de Relaciones de Aspecto:


# Distribución de Relaciones de Aspecto de Bounding Boxes:


# Distribución de Posiciones Centrales de Bounding Boxes:


# Gráfico de Dispersión de Ancho vs. Alto de Bounding Boxes:


# Mosaicos de Imágenes de Ejemplo:







# Resultados de Validación con Ultralytics:
metrics/precision(B): 0.006
metrics/recall(B): 0.0167
metrics/mAP50(B): 0.004
metrics/mAP50-95(B): 0.0036
metrics/precision(M): 0.006
metrics/recall(M): 0.0167
metrics/mAP50(M): 0.004
metrics/mAP50-95(M): 0.0036
fitness: 0.0073

# Validación de Calidad de Imágenes:
No se encontraron imágenes corruptas.
No se encontraron imágenes con dimensiones pequeñas.
No se encontraron imágenes duplicadas.

# Conclusiones del Análisis EDA:
- No se encontraron imágenes corruptas.
- Todas las imágenes tienen dimensiones adecuadas.
- No se detectaron imágenes duplicadas.
- La distribución de clases está desbalanceada. La clase 'bedroom' tiene 52 etiquetas, mientras que la clase 'dining_room' tiene 4 etiquetas. (Gráfica: Distribución de Clases)
- Se observa variabilidad significativa en los tamaños de las imágenes. La diferencia máxima en el ancho es de 323 píxeles, y la diferencia máxima en la altura es de 480 píxeles. (Gráfica: Distribución de Tamaños de Imágenes)
- Se observa variabilidad en las áreas de los bounding boxes, lo que sugiere la presencia de objetos de diferentes tamaños. (Gráfica: Distribución de Áreas de Bounding Boxes)
- Las relaciones de aspecto son consistentes, lo que puede facilitar el entrenamiento del modelo. (Gráfica: Distribución de Relaciones de Aspecto)
- Las relaciones de aspecto de los bounding boxes varían significativamente, lo que indica una diversidad en la forma de los objetos detectados en las imágenes. (Gráfica: Distribución de Relaciones de Aspecto de Bounding Boxes)
- Las posiciones centrales de las etiquetas se agrupan en áreas específicas. Este patrón de agrupación no necesariamente refleja un sesgo, sino que podría una característica propia del dataset. (Gráfica: Distribución de Posiciones Centrales de Bounding Boxes)
- El ancho y el alto de los bounding boxes están correlacionados. (Gráfica: Dispersión de Ancho vs. Alto de Bounding Boxes)

Distribución de etiquetas por clase:
- living_room: 39 etiquetas
- toilet: 23 etiquetas
- bathroom: 51 etiquetas
- balcony: 37 etiquetas
- bedroom: 52 etiquetas
- hallway: 45 etiquetas
- kitchen: 40 etiquetas
- home_office: 9 etiquetas
- children_room: 6 etiquetas
- walk_in_closet: 26 etiquetas
- utility_room: 10 etiquetas
- dining_room: 4 etiquetas