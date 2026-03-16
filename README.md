# Worker - GPU Training Unit (Distributed Mode)

Este componente es una unidad de ejecución autónoma diseñada para **optimización ML distribuida**. Utiliza el patrón **Invoker-Executor** y se conecta a una base de datos **PostgreSQL** centralizada para sincronizar estudios de Optuna a través de múltiples nodos.

## Arquitectura Distribuida (Escenario C)

En esta configuración, la inteligencia está descentralizada a través de una base de datos central.

- **Manager (Control Plane):** Crea estudios y envía tareas vía Celery.
- **Invoker (Worker):** Recibe tareas, consulta a **PostgreSQL** los siguientes parámetros de trial, los ejecuta en un contenedor efímero, y reporta resultados directamente a la base de datos.
- **PostgreSQL (Cerebro Central):** Almacena todos los trials, estudios e hiperparámetros. Located at `192.168.10.252:23436`.
- **Redis (Broker):** Gestiona la distribución de tareas entre Manager y Workers.
- **Dashboards:** 
    - **Custom Dashboard (Puerto 8000):** Monitoreo local/global de workers Celery y resúmenes de estudios.
    - **Optuna Dashboard (Oficial):** Visualización científica (debe desplegarse con el Manager).

---

## Flujo de Secuencia (Distribuido)

```mermaid
sequenceDiagram
    participant M as Manager (Remote)
    participant I as Invoker (Local)
    participant DB as PostgreSQL (Central)
    participant D as Docker Engine
    participant E as Executor (Container)

    M->>DB: 1. Create Study
    M->>I: 2. Celery Task: train_on_gpu(config)
    I->>DB: 3. Ask for next Hyperparameters
    DB-->>I: 4. Suggested Params
    I->>I: 5. Create temporary trial folder
    I->>D: 6. docker run ml_executor -v folder:/app/data
    D->>E: 7. Train on GPU with Params
    E->>E: 8. Save results.json
    E-->>D: 9. Exit and Auto-Remove Container
    I->>DB: 10. Write Results (Accuracy) to DB
    I->>M: 11. Celery Task Completed
```

---

## Gestión de Colas y Prioridad

Cada worker está configurado para escuchar múltiples colas con un orden de prioridad estricto.

### Jerarquía de Consumo (Prioridad Estricta)
El worker consume tareas en este orden:
1.  **Cola Privada (`worker_{NAME}`)**: Máxima prioridad. Reservada para tareas críticas o asignación directa.
2.  **`gpus_high`**: Tareas de alta prioridad.
3.  **`gpus_medium`**: Tareas estándar (por defecto).
4.  **`gpus_low`**: Tareas de baja prioridad / experimentales.

### Configuración del Consumidor
El comando de inicio define el orden:
```bash
celery worker -Q ${PRIVATE_QUEUE},gpus_high,gpus_medium,gpus_low --concurrency=1
```
Usar `--concurrency=1` asegura que el worker complete una tarea antes de obtener la siguiente de la cola de mayor prioridad.

---

## Múltiples Invokers en la Misma Máquina

Puedes ejecutar múltiples Invokers en la misma máquina, cada uno con su propia cola privada:

### Crear Invokers

```bash
# Usando el script launcher
./launcher_invoker.sh --private_name worker_1
./launcher_invoker.sh --private_name worker_2
./launcher_invoker.sh --private_name gpu_node_alpha
```

Cada Invoker obtiene:
- Su propia **cola privada** (ej: `worker_1`)
- Más las **3 colas públicas** (`gpus_high`, `gpus_medium`, `gpus_low`)

### Distribución de Tareas

Cuando las tareas se envían a una cola pública (ej: `gpus_medium`), Celery las distribuye round-robin entre todos los Invokers escuchando esa cola:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Invoker 1   │     │  Invoker 2   │     │  Invoker 3   │
│ private: w1  │     │ private: w2  │     │ private: w3  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       ▼                    ▼                    ▼
    ┌─────────────────────────────────────────────┐
    │              gpus_medium queue              │
    │         (Celery round-robin)                │
    │   Task1  Task2  Task3  Task4  Task5  ...  │
    └─────────────────────────────────────────────┘
```

---

## Nombrado de Contenedores Executor

Cada contenedor Executor se nombra automáticamente siguiendo el patrón:

```
{private_name}_son_{timestamp}
```

**Ejemplos:**
- `worker_1_son_20260312_143022`
- `worker_2_son_20260312_145501`
- `gpu_node_alpha_son_20260312_151200`

Esta convención permite:
- Rastrear qué Invoker lanzó qué Executor
- Identificar el orden de ejecuciones
- Depurar y monitorear recursos

El Invoker lee la variable de entorno `PRIVATE_QUEUE` para determinar el prefijo de nombres.

---

## Despliegue Portable

### Inicio Rápido

```bash
# Navegar al directorio worker
cd worker

# Crear un Invoker con una cola privada específica
./launcher_invoker.sh --private_name worker_1
```

### Opciones de Configuración

1. **Variables de Entorno:**
   - `REDIS_URL`: URL del broker central (por defecto: `redis://redis:6379/0`)
   - `PRIVATE_QUEUE`: Nombre de la cola privada (por defecto: `worker_default`)

2. **Monitoreo y Visibilidad:**
   - Para asegurar que el worker sea visible en **Flower** o el **Dashboard Custom**, no usar `--without-heartbeat` ni `--without-gossip` en el comando de inicio.
   - La configuración `worker_send_task_events: True` está habilitada en `celery_config.py` para prevenir errores de inicialización durante monitoreo remoto.

3. **Docker Compose:**
   ```bash
   PRIVATE_QUEUE=worker_alpha docker-compose up -d
   ```

---

## Construcción de Executor Personalizado

Si necesitas personalizar el executor:

```bash
cd worker/executor
docker build -t my-custom-executor:v1.0.0 .
```

Luego actualiza `invoker/config.yaml`:

```yaml
worker:
  executor_image: "my-custom-executor:v1.0.0"
```

---

## Monitoreo

Ver contenedores en ejecución:
```bash
docker ps | grep worker_
```

Ver logs del Invoker:
```bash
docker logs worker_worker_1
```

Ver logs del Executor (por nombre):
```bash
docker logs worker_1_son_20260312_143022
```

---

## Estructura del Proyecto

```
wyoloservice2_worker/
├── executor/                    # Versión simple (simulado)
│   ├── run_training.py         # Script de entrenamiento
│   └── requirements.txt
├── executor_v2.0/              # Versión completa (YOLO real)
│   ├── wyolo_mother/          # Contenedor principal
│   │   ├── app/application/   # Lógica de aplicación
│   │   └── lib/               # Biblioteca wyolo
│   ├── wyolo_father/          # Contenedor alternativo
│   └── production/            # Configuración de producción
├── docker-compose.yml          # Orquestación Docker
└── start_environment.sh       # Script de inicio
```

---

## Variables de Entorno

| Variable | Descripción | Por defecto |
|----------|-------------|-------------|
| `REDIS_URL` | URL de Redis broker | `redis://192.168.10.252:23437/0` |
| `OPTUNA_DB_URL` | URL de PostgreSQL | `postgresql://postgres:postgres@192.168.10.252:23436/wyoloservice` |
| `WORKER_NAME` | Nombre del worker | `default` |
| `PRIVATE_QUEUE` | Cola privada | `worker_default` |

---

**William R.** - AI Leader & Solutions Architect
