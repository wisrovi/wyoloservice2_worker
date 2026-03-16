# Celery ML Cluster - Distributed, Isolated, and Priority-Driven Architecture

This project implements a comprehensive ecosystem for distributed Machine Learning model training and optimization. It leverages **Celery** for orchestration, **Optuna** for hyperparameter tuning, and a Docker-based **Invoker-Executor** pattern to ensure total isolation of training processes.

## 🏗️ System Architecture (Distributed Scenario C)

The system uses a decoupled microservices architecture communicating through **Redis** as a message broker and **PostgreSQL** as a centralized intelligence hub.

```mermaid
graph TD
    U[User / Gradio UI] -->|YAML + Priority| API[API Gateway - FastAPI]
    API -->|Task: manage_study| M[Manager - Optuna]
    M -->|Task: train_on_gpu| W[Worker Invoker - Celery]
    
    subgraph "Central Infrastructure (192.168.10.252)"
        R[(Redis: Broker)]
        PG[(PostgreSQL: Optuna DB)]
    end

    subgraph "GPU Node (Remote/Local)"
        W -->|1. Ask for Params| PG
        W -->|2. Docker Run| E[Executor Container]
        E -->|3. Actual Training| E
        E -->|4. results.json| W
        W -->|5. Write Results| PG
    end
    
    W -->|6. Return Status| M
    M -->|Best Params| API
    M -.->|Persistence| PG
```

### Key Components

1.  **API Gateway (`control-plane/api/`)**: Built with **FastAPI**. It receives YAML configurations and injects routing metadata. Manages "Private Mode" for targeted machine dispatch and public priority mapping.
2.  **Manager / Orchestrator (`control-plane/manager/`)**: Coordinates **Optuna** studies. It is a specialized Celery worker that initializes studies in the central DB and asynchronously monitors results from GPU workers.
3.  **Worker Invoker (`worker/invoker`)**: Acts as a security bridge. It listens to Celery queues and, upon receiving a task, uses the **Docker SDK** to spin up an independent **Executor** container with hardware limits (CPU 85%, RAM 60%).
4.  **Worker Executor (`worker/executor`)**: An ephemeral container containing ML libraries and GPU drivers. It is destroyed after task completion (`auto_remove=True`), keeping the host node clean.
5.  **Dashboards**:
    *   **Custom Dashboard (`worker/dashboard`)**: Local/Global monitoring of Celery workers and study summaries (Port 8000).
    *   **Optuna Dashboard (Official)**: Scientific visualization of trials and hyperparameter importance (Recommended to run on Manager node - Port 8080).

## ⚖️ Priority Management and Routing

The system implements a **strict queue hierarchy** to ensure critical tasks are processed first. The Worker Invoker consumes from queues in this exact order:

1.  **`worker_private`**: Tasks directed to a specific machine (Private Mode).
2.  **`gpus_high`**: High-priority tasks.
3.  **`gpus_medium`**: Default standard priority.
4.  **`gpus_low`**: "Filler" or low-priority tasks.

By configuring `--concurrency=1` on workers, it ensures a GPU is never oversaturated and always respects the order of importance.

## 🚀 Deployment Guide

### 1. Central Server (API + Manager + Redis + Postgres)
Spin up the base infrastructure and orchestrator:

```bash
# Set credentials in .env
REDIS_URL=redis://192.168.10.252:23437/0
OPTUNA_DB_URL=postgresql://postgres:postgres@192.168.10.252:23436/wyoloservice

docker-compose -f control-plane/docker-compose.yml up -d --build
```

- **Custom Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Optuna Dashboard**: [http://localhost:8080](http://localhost:8080)
- **Flower (Monitor)**: [http://localhost:5555](http://localhost:5555)

### 2. Remote Workers (GPU Nodes)
The `worker/` folder is autonomous. Copy it to any server with Docker and GPU:

```bash
# Set the central server connection
export REDIS_URL=redis://192.168.10.252:23437/0
export OPTUNA_DB_URL=postgresql://postgres:postgres@192.168.10.252:23436/wyoloservice
export WORKER_NAME=gpu_node_01

docker-compose up -d --build
```

## 🛠️ User Workflow

1.  **Prepare Configuration**: Create a `.yaml` file with a `train` section (fixed hyperparameters) and a `sweeper` section (Optuna search space).
2.  **Launch Study**: Upload the file via Gradio or API, selecting the desired priority.
3.  **Monitor**: Track progress in Flower or use the Optuna Dashboard to visualize the optimization curves and best-found parameters.

---
**William R.** - AI Leader & Solutions Architect
