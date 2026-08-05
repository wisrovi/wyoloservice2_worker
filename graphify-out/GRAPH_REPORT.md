# Graph Report - wyoloservice2_worker  (2026-08-05)

## Corpus Check
- 130 files · ~1,579,850 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1438 nodes · 2201 edges · 102 communities (86 shown, 16 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 204 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7aa5119c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- app/states/__init__.py
- SQLite
- tracker.py
- Pipeline
- dashboard.js
- StepRegistry
- ResourceMonitor
- trainer/trainer_wrapper.py
- ParallelExecutor
- Any
- PipelineAsync
- PipelineTracker
- APIClient
- TypeValidator
- _trainer/trainer_wrapper.py
- TaskError
- CompositionHelper
- TaskTimer
- SharedMemory
- AnalysisManager
- CheckpointManager
- PipelineExporter
- QueryManager
- ConditionAsync
- dashboard/main.py
- SystemMetricsCollector
- ProgressManager
- DatasetAnalyzer
- AlertManager
- 📜 Changelog & Version History
- What You Must Do When Invoked
- log/__init__.py
- pipe.py
- config/config.py
- .__init__
- run_training.py
- Step Decorators
- constants.py
- .set_steps
- train_service.sh
- .link_to_pipeline
- .steps
- mount-cifs.sh
- wtrain/__init__.py
- micro_train.sh
- samba_mount.sh
- start_environment.sh
- wyolo
- _wpipe/__init__.py
- wpipe - Core Package
- Parallel Execution
- Pipeline Composition
- Resource Monitoring
- train_model
- 🚀 Wyolo - Professional YOLO Training Library
- Any
- Export & Analytics
- CheckpointManager
- Sqlite.py
- Type Hinting & Validation
- TrainingReportAnalyzer
- Task Timeouts
- Wsqlite
- Configuration File Structure
- .input
- WYOLO Service 2 Executor
- graphify reference: extra exports and benchmark
- Celery ML Cluster - Distributed, Isolated, and Priority-Driven Architecture
- patched_get_shared_connection
- Metric
- 🔄 Diagram Walkthrough
- 🛠️ Getting Started
- Docker Image Naming Convention
- publish_request_redis
- PatchedWSQLite
- wyoloservice2_worker — codegraph + graphify
- check_dataset.py
- ✨ Key Features
- graphify reference: query, path, explain
- Worker Executor
- decorators/__init__.py
- 💡 Usage Examples
- MLflowYOLOModel
- 🧪 Development & Testing
- opencode.json
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- _close_connections
- ⚙️ Container Lifecycle
- graphify.js
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- extraction-spec.md

## God Nodes (most connected - your core abstractions)
1. `PipelineTracker` - 45 edges
2. `Pipeline` - 41 edges
3. `QueryManager` - 25 edges
4. `AlertManager` - 24 edges
5. `AnalysisManager` - 24 edges
6. `PipelineAsync` - 20 edges
7. `🚀 Wyolo - Professional YOLO Training Library` - 20 edges
8. `AlertFiredModel` - 19 edges
9. `📜 Changelog & Version History` - 19 edges
10. `PipelineModel` - 18 edges

## Surprising Connections (you probably didn't know these)
- `PipelineAsync` --uses--> `APIClient`  [INFERRED]
  executor_v2.0/wtrain/app/_wpipe/pipe/pipe_async.py → executor_v2.0/wtrain/app/_wpipe/api_client/api_client.py
- `Pipeline` --uses--> `APIClient`  [INFERRED]
  executor_v2.0/wtrain/app/_wpipe/pipe/pipe.py → executor_v2.0/wtrain/app/_wpipe/api_client/api_client.py
- `CheckpointManager` --uses--> `CheckpointModel`  [INFERRED]
  executor_v2.0/wtrain/app/_wpipe/checkpoint/checkpoint.py → executor_v2.0/wtrain/app/_wpipe/sqlite/tables_dto/tracker_models.py
- `PipelineExporter` --uses--> `PipelineModel`  [INFERRED]
  executor_v2.0/wtrain/app/_wpipe/export/exporter.py → executor_v2.0/wtrain/app/_wpipe/sqlite/tables_dto/tracker_models.py
- `PipelineExporter` --uses--> `SystemMetricsModel`  [INFERRED]
  executor_v2.0/wtrain/app/_wpipe/export/exporter.py → executor_v2.0/wtrain/app/_wpipe/sqlite/tables_dto/tracker_models.py

## Import Cycles
- None detected.

## Communities (102 total, 16 thin omitted)

### Community 0 - "app/states/__init__.py"
Cohesion: 0.18
Nodes (14): config_pipeline(), main(), check_gpu_available(), check_minio_buckets(), Asegura que los buckets necesarios existan en MinIO., error_capture(), not_train(), PublicResultsInput (+6 more)

### Community 1 - "SQLite"
Cohesion: 0.14
Nodes (9): SQLite database module for storing pipeline execution records., Connection, SQLite database wrapper for storing pipeline records. Attributes: db_name…, Writes or updates a record in the database. Args: input_data: Input data to…, Reads a record from the database by its ID. Args: record_id (int): The ID of…, Counts the total number of records in the database. Returns: int: Total record…, Gets or creates a thread-safe SQLite connection. Returns: sqlite3.Connection:…, Context manager entry point. (+1 more)

### Community 2 - "tracker.py"
Cohesion: 0.14
Nodes (47): AlertConfigModel, AlertFiredModel, CheckpointModel, ComparisonModel, EventModel, PerformanceStatsModel, PipelineModel, PipelineRelationModel (+39 more)

### Community 3 - "Pipeline"
Cohesion: 0.09
Nodes (25): Pipeline, Any, setter, Start tracking a pipeline step., End tracking a pipeline step., Initializes the context for a pipeline run. This includes setting the start…, Sets up and returns a progress bar generator. Configures the progress bar based…, Executes all steps of the pipeline sequentially. This method iterates through… (+17 more)

### Community 4 - "dashboard.js"
Cohesion: 0.09
Nodes (43): currentAlerts, escapeHtml(), fmtDuration(), fmtTime(), formatJSON(), getStepIcon(), graphState, hideNodeTooltip() (+35 more)

### Community 5 - "StepRegistry"
Cohesion: 0.06
Nodes (29): AutoRegister, clear_registry(), DecoratedStep, get_step_registry(), Any, Step decorators for pipeline definitions. Provides @wpipe.step() decorator for…, Execute decorated step. Args: context: The pipeline context. Returns: The…, Get step name. Returns: The step name. (+21 more)

### Community 6 - "ResourceMonitor"
Cohesion: 0.06
Nodes (24): Resource monitoring for pipeline execution. Tracks system metrics like RAM and…, Any, Resource monitoring for pipeline task execution. Tracks RAM, CPU, and other…, Start resource monitoring., Stop resource monitoring., Internal monitoring loop., Monitor system resources during task execution., Get elapsed time in seconds. Returns: Elapsed time since monitoring started. (+16 more)

### Community 7 - "trainer/trainer_wrapper.py"
Cohesion: 0.08
Nodes (20): Wyolo - Professional YOLO Training Library with MLOps integration., Elemental, gpu_compatibility_check(), obtener_info_gpu_json(), print_gpu_report(), Obtiene información detallada sobre las GPUs disponibles y la devuelve en…, create_trainer(), display_config_banner() (+12 more)

### Community 8 - "ParallelExecutor"
Cohesion: 0.07
Nodes (25): Enum, ContextMerger, DAGScheduler, ExecutionMode, ParallelExecutor, Any, Parallel execution engine for WPipe pipelines. Enables execution of multiple…, Executes pipeline steps in parallel with dependency resolution. (+17 more)

### Community 9 - "Any"
Cohesion: 0.11
Nodes (15): Condition, Any, Get the steps for the chosen branch based on the evaluation result. Args: data:…, Serialize a pipeline step for representation. Args: step: The pipeline step to…, Initialize the For loop block. Args: steps: Steps to execute in each loop.…, Convert the block to a dictionary for serialization. Returns: Dict[str, Any]:…, Check if the loop should continue its execution. Args: data: The current…, Initialize a Background block. Args: step: The step to execute in background… (+7 more)

### Community 10 - "PipelineAsync"
Cohesion: 0.11
Nodes (21): _is_async_callable(), PipelineAsync, Any, Add an event or annotation to the pipeline. Args: event_type: Category of the…, Add a checkpoint to the pipeline. Args: checkpoint_name: Unique name for the…, Add steps to be executed when an error occurs. Args: steps: List of callables…, Evaluate and fire checkpoints based on current data. Args: data: Current…, Check if a callable is async (handles both functions and callable objects).… (+13 more)

### Community 11 - "PipelineTracker"
Cohesion: 0.06
Nodes (28): PipelineTracker, Any, Unified Pipeline Tracker. Orchestrates registration, step tracking, alerts, and…, Delegate to alerts manager., Delegate to queries manager., Delegate to queries manager., Delegate to analysis manager., Delegate to analysis manager. (+20 more)

### Community 12 - "APIClient"
Cohesion: 0.09
Nodes (21): APIClient, Any, Registers a worker with the API server. Args: data (Dict[str, Any]): Worker…, Performs a worker health check. Args: data (Dict[str, Any]): Health check data.…, Registers a new process with the API server. Args: data (Dict[str, Any]):…, Signals the end of a process to the API server. Args: data (Dict[str, Any]):…, Client for communicating with the pipeline API server. Attributes: base_url…, Updates task status on the API server. Args: data (Dict[str, Any]): Task update… (+13 more)

### Community 13 - "TypeValidator"
Cohesion: 0.10
Nodes (22): Type hinting support for WPipe pipelines. Provides type validation and…, GenericPipeline, PipelineContext, Any, BaseModel, Type hinting validators for pipeline context and data. Provides utilities for…, Internal helper for Pydantic validation., Internal helper for base type validation. (+14 more)

### Community 14 - "_trainer/trainer_wrapper.py"
Cohesion: 0.09
Nodes (13): Elemental, MLflowYOLOModel, obtener_info_gpu_json(), Obtiene información detallada sobre las GPUs disponibles y la devuelve en…, create_trainer(), get_datetime(), load_config(), Elemental (+5 more)

### Community 15 - "TaskError"
Cohesion: 0.09
Nodes (20): ApiError, Codes, ProcessError, Exception, Exception module for pipeline errors. Defines custom exception classes for…, Error codes for pipeline exceptions. Attributes: UPDATE_TASK: Code for task…, Exception for API-related errors. Attributes: message: Descriptive error…, Initialize ApiError. Args: message: Error message. error_code: Error code from… (+12 more)

### Community 16 - "CompositionHelper"
Cohesion: 0.09
Nodes (18): Pipeline composition module for nested pipelines., CompositionHelper, NestedPipelineStep, PipelineAsStep, Any, Pipeline composition for nested pipelines. Allows using a Pipeline as a step…, Extract subset of context for child pipeline. Args: context: Full context keys:…, Validate that child pipeline context is compatible with parent. Args:… (+10 more)

### Community 17 - "TaskTimer"
Cohesion: 0.11
Nodes (17): Timeout management for task execution in WPipe pipelines. Provides timeout…, Any, Exception, Timeout management for task execution in WPipe pipelines. Provides timeout…, Enter context manager., Exit context manager., Get elapsed time in seconds., Check if timeout was exceeded. (+9 more)

### Community 18 - "SharedMemory"
Cohesion: 0.10
Nodes (16): Memory management module for the pipeline., get_memory(), memory_limit(), memory_limit_decorator(), Any, Memory limit utilities and shared memory storage for the pipeline., Simple key-value storage for sharing data across pipeline runs., Initialize shared memory storage. (+8 more)

### Community 19 - "AnalysisManager"
Cohesion: 0.13
Nodes (11): AnalysisManager, Any, Statistical analysis and trend calculation for the dashboard., Handles statistical aggregations and trend calculations., Identify slowest steps across all executions. Args: limit: Maximum number of…, Get comprehensive analysis of all states/steps. Returns: Dictionary with states…, Initialize the AnalysisManager with database accessors. Args: db_pipelines:…, Get comprehensive analysis of all pipelines. Returns: Dictionary with pipelines… (+3 more)

### Community 20 - "CheckpointManager"
Cohesion: 0.07
Nodes (29): CheckpointManager, Any, Checkpoint manager for pipeline resumption after interruptions. Allows…, Clear all checkpoints for a pipeline. Args: pipeline_id: Unique pipeline…, Get statistics about checkpoints for a pipeline. Args: pipeline_id: Unique…, Manages pipeline checkpoints for resume functionality., Initialize checkpoint manager. Args: db_path: Path to the tracking database, Save a checkpoint at a specific step. Args: pipeline_id: Unique pipeline… (+21 more)

### Community 21 - "PipelineExporter"
Cohesion: 0.13
Nodes (12): PipelineExporter, Any, Export and analytics module for pipeline execution data. Provides functionality…, Exports calculated pipeline statistics. Args: pipeline_id (Optional[str]): ID…, Exports data as a JSON string or file. Args: data (List[Dict[str, Any]]): Data…, Exports data as a CSV string or file. Args: data (List[Dict[str, Any]]): Data…, Export pipeline execution data to various formats. Attributes: db_path (str):…, Calculates pipeline execution statistics using WSQLite. Args: pipeline_id… (+4 more)

### Community 22 - "QueryManager"
Cohesion: 0.14
Nodes (12): Any, QueryManager, Data query module for pipeline and event retrieval., Handles data retrieval for the dashboard and API., Get all executions of a pipeline by name. Args: name: Name of the pipeline.…, Get recent fired alerts. Args: limit: Maximum number of alerts to return.…, Get alert configurations. Returns: A list of alert configuration dictionaries., Get pipeline events. Args: pipeline_id: Optional filter by pipeline ID. limit:… (+4 more)

### Community 23 - "ConditionAsync"
Cohesion: 0.16
Nodes (10): ConditionAsync, ForAsync, Any, Asynchronous logic control blocks for WPipe pipelines. This module provides…, An asynchronous conditional branch in the pipeline. Attributes: expression…, Initialize the ConditionAsync block. Args: expression: A Python expression…, Evaluate the condition expression using the provided data as context. Args:…, An asynchronous loop block in the pipeline. Attributes: steps (List[Any]): The… (+2 more)

### Community 24 - "dashboard/main.py"
Cohesion: 0.20
Nodes (11): Dashboard module for wpipe. Provides a web-based dashboard to monitor pipeline…, create_app(), get_dashboard_html(), main(), wpipe Dashboard - Enterprise-grade pipeline visualization This module provides…, Allow running the dashboard as a module: python -m wpipe.dashboard, Generate the complete dashboard HTML using Jinja2 templates. Returns: The…, Start the wpipe dashboard server. Args: db_path: Path to the SQLite database.… (+3 more)

### Community 25 - "SystemMetricsCollector"
Cohesion: 0.15
Nodes (9): get_system_metrics(), System metrics collection for WPipe pipelines. This module provides utilities…, Get current system metrics. Returns: Dict[str, float]: A dictionary containing…, Collects system metrics during pipeline execution in a background thread.…, Initialize the metrics collector. Args: tracker: Tracker instance for recording…, Start the background collection thread., Stop the background collection thread., Continuous collection loop executed in the background thread. (+1 more)

### Community 26 - "ProgressManager"
Cohesion: 0.15
Nodes (9): ProgressManager, Any, Progress management for WPipe pipelines. This module provides a singleton…, Singleton manager for Rich Progress bars. Ensures that a single progress…, Create or return the singleton instance. Returns: ProgressManager: The shared…, Initialize the instance (called after __new__)., Enter context manager for progress tracking. Returns: Progress: The Rich…, Exit context manager for progress tracking. Args: exc_type: Exception type if… (+1 more)

### Community 27 - "DatasetAnalyzer"
Cohesion: 0.13
Nodes (19): ClassificationAnalyzer, DatasetAnalyzer, DatasetEDAState, DetectionAnalyzer, Any, Path, Determine whether a YOLO dataset is intended for object detection or…, Analyze a dataset and return its statistics. Parameters ---------- dataset_path… (+11 more)

### Community 28 - "AlertManager"
Cohesion: 0.14
Nodes (10): AlertManager, Any, Alert system for pipeline and step monitoring., Check and fire step-level alerts. Args: pipeline_id: Unique identifier for the…, Handles alert threshold configuration and firing logic., Check and fire pipeline-level alerts. Args: pipeline_id: Unique identifier for…, Initialize the AlertManager. Args: db_alerts_config: Database accessor for…, Add an alert threshold configuration. Args: metric: The metric to monitor.… (+2 more)

### Community 29 - "📜 Changelog & Version History"
Cohesion: 0.07
Nodes (27): 1. 🚶 Process Flowchart (Pipeline Stack), 2. 🛡️ Key Features of Version 2.0, 3. 🏗️ MinIO S3 Artifacts Tree Layout, 4. ⚙️ Configuration Templates, 🐳 Building and Deployment, 📜 Changelog & Version History, Input Configuration (`base_config.yaml`), Output Results (`results.json`) (+19 more)

### Community 30 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 31 - "log/__init__.py"
Cohesion: 0.33
Nodes (5): Logging utilities for wpipe., new_logger(), Any, Logging utilities for wpipe., Create and configure a new logger instance. Args: process_name: Name for the…

### Community 32 - "pipe.py"
Cohesion: 0.11
Nodes (16): API Client module for pipeline tracking and communication., Background, For, Parallel, Logic control blocks for WPipe pipelines. This module provides classes for…, A loop block in the pipeline. Attributes: steps (List[Any]): The steps to be…, A background task that executes without blocking the pipeline. The step runs in…, Represents a parallel execution block in the pipeline. Attributes: steps… (+8 more)

### Community 33 - "config/config.py"
Cohesion: 0.50
Nodes (3): crear_archivo(), obtener_usuario(), Obtiene el nombre de usuario usando 'whoami'.

### Community 35 - "run_training.py"
Cohesion: 0.50
Nodes (3): main(), Worker Executor Script. This script runs inside a container and performs the…, Main execution entry point for the training trial. Reads configuration,…

### Community 36 - "Step Decorators"
Cohesion: 0.09
Nodes (22): Advanced Patterns, API Reference, Auto-Register All, AutoRegister, Best Practices, Decorator Parameters, Dependency Chains, Error Handling (+14 more)

### Community 37 - "constants.py"
Cohesion: 0.50
Nodes (3): Codes, Error codes and execution constants for WPipe., Standard task and process status codes.

### Community 56 - "_wpipe/__init__.py"
Cohesion: 0.16
Nodes (10): _patched_return(), WPipe - Pipeline Orchestration Library. A high-performance library for building…, Patched return_connection that releases semaphore but avoids rollback., PostTrainContext, Shared context for the post-training pipeline. Attributes: model: Trained YOLO…, LlmAnalyzer, PostTrain, Read absolute image dirs (train/val/test) from a detection dataset YAML.… (+2 more)

### Community 57 - "wpipe - Core Package"
Cohesion: 0.10
Nodes (19): Architecture, Basic Usage, Code Quality, Conditional Pipeline, Documentation, Exception Handling, Features, File Structure (+11 more)

### Community 58 - "Parallel Execution"
Cohesion: 0.11
Nodes (18): `add_step(name, func, mode=IO_BOUND, timeout=None, depends_on=None)`, `add_step(step: StepDependency)`, API, DAGScheduler, `execute(context: Dict) -> Dict`, ExecutionMode, Features, `get_parallel_groups() -> List[List[StepDependency]]` (+10 more)

### Community 59 - "Pipeline Composition"
Cohesion: 0.11
Nodes (17): Advanced Usage, API, Best Practices, CompositionHelper, Context Filtering, Context Management, Context Transformation, Features (+9 more)

### Community 60 - "Resource Monitoring"
Cohesion: 0.11
Nodes (17): `add(task_name: str, monitor: ResourceMonitor)`, API, Features, `get_peak_ram() -> float`, `get_summary() -> Dict`, `get_summary() -> Dict`, `get_total_cpu_time() -> float`, `__init__(task_name: str, db_path: Optional[str] = None)` (+9 more)

### Community 61 - "train_model"
Cohesion: 0.24
Nodes (13): load_yaml(), BaseModel, UserInput, BaseModel, train_model(), UserInput, get_base_config(), get_complete_config() (+5 more)

### Community 62 - "🚀 Wyolo - Professional YOLO Training Library"
Cohesion: 0.12
Nodes (15): 🌟 Acknowledgments, 🏗️ Architecture Components, Code Style, 🤝 Contributing, 🚶 Diagram Walkthrough (High-Level Process Flow), 📂 File-by-File Guide, 📁 File Structure, Key Directory Functions (+7 more)

### Community 63 - "Any"
Cohesion: 0.18
Nodes (10): Any, setter, Sets the details data and saves state., Gets the error data dictionary., Sets the error data and saves state., Converts non-serializable objects to strings within a dictionary. Args: data…, Context manager exit point, shuts down the executor., Gets the output data dictionary. (+2 more)

### Community 64 - "Export & Analytics"
Cohesion: 0.14
Nodes (13): API, CSV, Export & Analytics, `export_metrics(pipeline_id, format, output_path) -> str`, `export_pipeline_logs(pipeline_id, format, output_path) -> str`, `export_statistics(pipeline_id, format, output_path) -> str`, Features, `__init__(db_path: str)` (+5 more)

### Community 65 - "CheckpointManager"
Cohesion: 0.15
Nodes (12): API, `can_resume(pipeline_id) -> bool`, Checkpointing & Resume, CheckpointManager, `clear_checkpoints(pipeline_id)`, Features, `get_checkpoint_stats(pipeline_id) -> Dict`, `get_last_checkpoint(pipeline_id) -> Optional[Dict]` (+4 more)

### Community 66 - "Sqlite.py"
Cohesion: 0.20
Nodes (9): SQLite database module for storing pipeline execution records., BaseModel, DTO models for LogGestor record entries in SQLite., Data Transfer Object for the LogGestor records table in SQLite. Attributes: id…, WsqliteModel, BaseModel, DTO models for pipeline record entries in SQLite., Data Transfer Object for the records table in SQLite. Attributes: id… (+1 more)

### Community 67 - "Type Hinting & Validation"
Cohesion: 0.17
Nodes (11): API, Exceptions, Features, GenericPipeline, PipelineContext, Quick Start, Type Hinting & Validation, TypeValidator (+3 more)

### Community 68 - "TrainingReportAnalyzer"
Cohesion: 0.23
Nodes (7): Path, Generate AI-assisted training analysis using OpenCode with fallback., Safely convert a CSV cell to float, returning default on failure., Generate a basic report from CSV data when OpenCode fails., Generate a professional training report. Args: results_file: Path to YOLO…, Attempt to generate report using OpenCode with timeout., TrainingReportAnalyzer

### Community 69 - "Task Timeouts"
Cohesion: 0.18
Nodes (10): API, Decorators, Exceptions, Features, Quick Start, Task Timeouts, TaskTimer, `@timeout_async(seconds: Optional[float])` (+2 more)

### Community 70 - "Wsqlite"
Cohesion: 0.22
Nodes (6): Saves or updates the current state in the database. Uses an Upsert-like logic…, Counts the total number of records in the database. Returns: int: Total record…, Context manager entry point., Context manager exit point, ensures state is saved., Simplified SQLite wrapper for pipeline records (LogGestor). Attributes: db_name…, Wsqlite

### Community 71 - "Configuration File Structure"
Cohesion: 0.20
Nodes (10): Basic Configuration, Classification Dataset, Configuration File Structure, ⚙️ Configuration & Setup, Dataset Configuration, Detection Dataset, Environment Variables, Hyperparameter Tuning (+2 more)

### Community 72 - ".input"
Cohesion: 0.28
Nodes (5): cargar_datos_a_influx(), seleccionar_archivo(), seleccionar_columnas(), Gets the input data dictionary., Sets the input data and saves state.

### Community 73 - "WYOLO Service 2 Executor"
Cohesion: 0.22
Nodes (8): Changelog, Configuration, Development & Usage, Features, Project Structure, Running Locally, v2.2.14 (2026-08-05), WYOLO Service 2 Executor

### Community 74 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 75 - "Celery ML Cluster - Distributed, Isolated, and Priority-Driven Architecture"
Cohesion: 0.22
Nodes (8): 1. Central Server (API + Manager + Redis + Postgres), 2. Remote Workers (GPU Nodes), Celery ML Cluster - Distributed, Isolated, and Priority-Driven Architecture, 🚀 Deployment Guide, Key Components, ⚖️ Priority Management and Routing, 🏗️ System Architecture (Distributed Scenario C), 🛠️ User Workflow

### Community 76 - "patched_get_shared_connection"
Cohesion: 0.25
Nodes (8): __getattr__(), patched_get_shared_connection(), patched_insert(), Any, Connection, Handle lazy loading of modules., Obtain a shared database connection to improve performance., Insert a new record and return the generated ID without committing immediately.

### Community 77 - "Metric"
Cohesion: 0.25
Nodes (5): Pipeline tracking module for storing execution data. Provides functionality to…, Metric, Record an event. Args: pipeline_id: Unique pipeline identifier. event_type:…, Constants for alert metrics and utility for recording numeric data., Record a numeric metric in the current pipeline execution. Args: name: Name of…

### Community 78 - "🔄 Diagram Walkthrough"
Cohesion: 0.25
Nodes (8): 1. **Configuration Phase**, 2. **GPU Detection & Setup**, 3. **Trainer Initialization**, 4. **Training Execution**, 5. **Hyperparameter Optimization** (Optional), 6. **Results Management**, 7. **Model Registration**, 🔄 Diagram Walkthrough

### Community 79 - "🛠️ Getting Started"
Cohesion: 0.25
Nodes (8): CLI Usage, Development Installation, Full Installation (with all dependencies), 🛠️ Getting Started, Installation, Prerequisites, Quick Start, Standard Installation

### Community 80 - "Docker Image Naming Convention"
Cohesion: 0.29
Nodes (6): Build and Push Commands, Docker Compose References, Docker Image Naming Convention, Environment Variables, Image Registry, Pattern

### Community 81 - "publish_request_redis"
Cohesion: 0.62
Nodes (6): _get_study_id(), _make_key(), publish_request_redis(), publish_results_redis(), _to_dict(), _to_dict_deep()

### Community 82 - "PatchedWSQLite"
Cohesion: 0.29
Nodes (5): PatchedWSQLite, Internal WSQLite with performance tuning. This class provides a thread-safe…, Initializes the SQLite instance. Args: db_name (str): Name of the database…, Initializes the Wsqlite instance. Args: db_name (str): Name of the database…, WSQLiteBase

### Community 83 - "wyoloservice2_worker — codegraph + graphify"
Cohesion: 0.33
Nodes (5): 🐳 Docker Deployment & Test Verification Workflow, Estado, graphify, Sync automático, wyoloservice2_worker — codegraph + graphify

### Community 84 - "check_dataset.py"
Cohesion: 0.53
Nodes (5): check_dataset(), Dataset, DatasetConfig, BaseModel, UserInput

### Community 85 - "✨ Key Features"
Cohesion: 0.33
Nodes (6): 🚀 **Advanced Training Capabilities**, 🧬 **Hyperparameter Optimization**, ✨ Key Features, 🔧 **MLOps Integration**, 📊 **Monitoring & Visualization**, 🎯 **Multi-Task Support**

### Community 86 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 87 - "Worker Executor"
Cohesion: 0.40
Nodes (4): Customization, Features, Worker Executor, Workflow

### Community 88 - "decorators/__init__.py"
Cohesion: 0.40
Nodes (4): __getattr__(), Any, Decorators module for WPipe. Provides @step decorator for inline step…, Handle lazy loading of modules.

### Community 89 - "💡 Usage Examples"
Cohesion: 0.40
Nodes (5): Basic Object Detection Training, GPU-Optimized Training, Image Classification with Genetic Optimization, MLflow Integration Example, 💡 Usage Examples

### Community 91 - "🧪 Development & Testing"
Cohesion: 0.50
Nodes (4): Code Quality, Development Setup, 🧪 Development & Testing, Running Tests

### Community 92 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 93 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 94 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 95 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 96 - "_close_connections"
Cohesion: 0.67
Nodes (3): _close_connections(), Cleanup connections and threads on exit., register

### Community 97 - "⚙️ Container Lifecycle"
Cohesion: 0.67
Nodes (3): Build Process, ⚙️ Container Lifecycle, Runtime Process

## Knowledge Gaps
- **251 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `graphState`, `translations`, `pipelineTabs` (+246 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PipelineTracker` connect `PipelineTracker` to `.__init__`, `Pipeline`, `tracker.py`, `PipelineAsync`, `Metric`, `AnalysisManager`, `QueryManager`, `SystemMetricsCollector`, `AlertManager`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `Pipeline` connect `Pipeline` to `pipe.py`, `.__init__`, `.set_steps`, `.link_to_pipeline`, `.steps`, `APIClient`, `TaskError`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `to_obj()` connect `app/states/__init__.py` to `publish_request_redis`, `CheckpointManager`, `check_dataset.py`, `_wpipe/__init__.py`, `train_model`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `PipelineTracker` (e.g. with `.__init__()` and `.__init__()`) actually correct?**
  _`PipelineTracker` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Pipeline` (e.g. with `PipelineAsync` and `APIClient`) actually correct?**
  _`Pipeline` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `QueryManager` (e.g. with `alerts_config` and `alerts_fired`) actually correct?**
  _`QueryManager` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `AlertManager` (e.g. with `AlertConfigModel` and `AlertFiredModel`) actually correct?**
  _`AlertManager` has 15 INFERRED edges - model-reasoned connections that need verification._