# Graph Report - /home/william.rodriguez/Documents/w_libraries/train_service2/wyoloservice2_worker  (2026-07-30)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1023 nodes · 1805 edges · 56 communities (44 shown, 12 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 213 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- states/__init__.py
- Wsqlite
- QueryManager
- Pipeline
- dashboard.js
- StepRegistry
- ResourceMonitor
- trainer/trainer_wrapper.py
- ParallelExecutor
- pipe.py
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
- Any
- ConditionAsync
- dashboard/main.py
- SystemMetricsCollector
- ProgressManager
- tracker_models.py
- AlertManager
- util/__init__.py
- utils.py
- log/__init__.py
- pipe/__init__.py
- config/config.py
- .__init__
- run_training.py
- exporter.py
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

## God Nodes (most connected - your core abstractions)
1. `PipelineTracker` - 45 edges
2. `Pipeline` - 42 edges
3. `QueryManager` - 25 edges
4. `AlertManager` - 24 edges
5. `AnalysisManager` - 24 edges
6. `PipelineAsync` - 20 edges
7. `AlertFiredModel` - 19 edges
8. `ResourceMonitor` - 18 edges
9. `PipelineModel` - 18 edges
10. `AlertConfigModel` - 18 edges

## Surprising Connections (you probably didn't know these)
- `PipelineAsync` --uses--> `APIClient`  [INFERRED]
  executor_v2.0/wtrain/app/_wpipe/pipe/pipe_async.py → executor_v2.0/wtrain/app/_wpipe/api_client/api_client.py
- `Pipeline` --uses--> `APIClient`  [INFERRED]
  executor_v2.0/wtrain/app/_wpipe/pipe/pipe.py → executor_v2.0/wtrain/app/_wpipe/api_client/api_client.py
- `CheckpointManager` --uses--> `CheckpointModel`  [INFERRED]
  executor_v2.0/wtrain/app/_wpipe/checkpoint/checkpoint.py → executor_v2.0/wtrain/app/_wpipe/sqlite/tables_dto/tracker_models.py
- `MyContext` --uses--> `ProcessError`  [INFERRED]
  executor_v2.0/wtrain/lib/src/wyolo/trainer/post_train.py → executor_v2.0/wtrain/app/_wpipe/exception/api_error.py
- `StepClass` --uses--> `ProcessError`  [INFERRED]
  executor_v2.0/wtrain/lib/src/wyolo/trainer/post_train.py → executor_v2.0/wtrain/app/_wpipe/exception/api_error.py

## Import Cycles
- None detected.

## Communities (56 total, 12 thin omitted)

### Community 0 - "states/__init__.py"
Cohesion: 0.06
Nodes (58): config_pipeline(), main(), check_dataset(), Dataset, DatasetConfig, BaseModel, UserInput, check_gpu_available() (+50 more)

### Community 1 - "Wsqlite"
Cohesion: 0.05
Nodes (39): cargar_datos_a_influx(), seleccionar_archivo(), seleccionar_columnas(), SQLite database module for storing pipeline execution records., PatchedWSQLite, Any, Connection, setter (+31 more)

### Community 2 - "QueryManager"
Cohesion: 0.12
Nodes (44): ComparisonModel, EventModel, PerformanceStatsModel, PipelineModel, PipelineRelationModel, Data Transfer Object for the performance_stats table. Attributes: id…, Data Transfer Object for the pipelines table. Attributes: id (str): Primary Key…, Data Transfer Object for the system_metrics table. Attributes: id… (+36 more)

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
Cohesion: 0.07
Nodes (21): Wyolo - Professional YOLO Training Library with MLOps integration., Elemental, MLflowYOLOModel, gpu_compatibility_check(), obtener_info_gpu_json(), print_gpu_report(), Obtiene información detallada sobre las GPUs disponibles y la devuelve en…, create_trainer() (+13 more)

### Community 8 - "ParallelExecutor"
Cohesion: 0.07
Nodes (25): Enum, ContextMerger, DAGScheduler, ExecutionMode, ParallelExecutor, Any, Parallel execution engine for WPipe pipelines. Enables execution of multiple…, Executes pipeline steps in parallel with dependency resolution. (+17 more)

### Community 9 - "pipe.py"
Cohesion: 0.09
Nodes (24): API Client module for pipeline tracking and communication., Background, Condition, For, Parallel, Any, Logic control blocks for WPipe pipelines. This module provides classes for…, Get the steps for the chosen branch based on the evaluation result. Args: data:… (+16 more)

### Community 10 - "PipelineAsync"
Cohesion: 0.11
Nodes (21): _is_async_callable(), PipelineAsync, Any, Add an event or annotation to the pipeline. Args: event_type: Category of the…, Add a checkpoint to the pipeline. Args: checkpoint_name: Unique name for the…, Add steps to be executed when an error occurs. Args: steps: List of callables…, Evaluate and fire checkpoints based on current data. Args: data: Current…, Check if a callable is async (handles both functions and callable objects).… (+13 more)

### Community 11 - "PipelineTracker"
Cohesion: 0.10
Nodes (16): PipelineTracker, Any, Unified Pipeline Tracker. Orchestrates registration, step tracking, alerts, and…, Delegate to alerts manager., Delegate to queries manager., Delegate to analysis manager., Register a pipeline and its steps. Args: name: Pipeline name. pipeline_steps:…, Mark a pipeline as completed or failed. Args: pipeline_id: Unique pipeline… (+8 more)

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
Cohesion: 0.10
Nodes (19): ApiError, Codes, ProcessError, Exception, Exception module for pipeline errors. Defines custom exception classes for…, Error codes for pipeline exceptions. Attributes: UPDATE_TASK: Code for task…, Exception for API-related errors. Attributes: message: Descriptive error…, Initialize ApiError. Args: message: Error message. error_code: Error code from… (+11 more)

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
Cohesion: 0.12
Nodes (12): AnalysisManager, Any, Statistical analysis and trend calculation for the dashboard., Handles statistical aggregations and trend calculations., Identify slowest steps across all executions. Args: limit: Maximum number of…, Get comprehensive analysis of all states/steps. Returns: Dictionary with states…, Initialize the AnalysisManager with database accessors. Args: db_pipelines:…, Get comprehensive analysis of all pipelines. Returns: Dictionary with pipelines… (+4 more)

### Community 20 - "CheckpointManager"
Cohesion: 0.13
Nodes (11): CheckpointManager, Any, Checkpoint manager for pipeline resumption after interruptions. Allows…, Clear all checkpoints for a pipeline. Args: pipeline_id: Unique pipeline…, Get statistics about checkpoints for a pipeline. Args: pipeline_id: Unique…, Manages pipeline checkpoints for resume functionality., Initialize checkpoint manager. Args: db_path: Path to the tracking database, Save a checkpoint at a specific step. Args: pipeline_id: Unique pipeline… (+3 more)

### Community 21 - "PipelineExporter"
Cohesion: 0.19
Nodes (10): PipelineExporter, Any, Exports calculated pipeline statistics. Args: pipeline_id (Optional[str]): ID…, Exports data as a JSON string or file. Args: data (List[Dict[str, Any]]): Data…, Export pipeline execution data to various formats. Attributes: db_path (str):…, Calculates pipeline execution statistics using WSQLite. Args: pipeline_id…, Initializes the pipeline exporter. Args: db_path (str): Path to the tracking…, Exports pipeline execution logs. Args: pipeline_id (Optional[str]): ID of the… (+2 more)

### Community 22 - "Any"
Cohesion: 0.13
Nodes (9): Any, Get all executions of a pipeline by name. Args: name: Name of the pipeline.…, Get recent fired alerts. Args: limit: Maximum number of alerts to return.…, Get alert configurations. Returns: A list of alert configuration dictionaries., Get pipeline events. Args: pipeline_id: Optional filter by pipeline ID. limit:…, Initialize the QueryManager with database accessors. Args: db_pipelines:…, Parse JSON strings in specified fields of a dictionary. Args: data: The…, Get list of pipelines for the dashboard. Args: limit: Maximum number of… (+1 more)

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

### Community 27 - "tracker_models.py"
Cohesion: 0.18
Nodes (10): AlertConfigModel, CheckpointModel, BaseModel, DTO models for pipeline tracking, performance, and monitoring in SQLite., Data Transfer Object for the alerts_config table. Attributes: id…, Data Transfer Object for the resource_metrics table. Attributes: id…, Data Transfer Object for the checkpoints table. Attributes: id (Optional[int]):…, ResourceMetricsModel (+2 more)

### Community 28 - "AlertManager"
Cohesion: 0.22
Nodes (9): AlertFiredModel, Data Transfer Object for the alerts_fired table. Attributes: id…, AlertManager, Any, Check and fire step-level alerts. Args: pipeline_id: Unique identifier for the…, Handles alert threshold configuration and firing logic., Check and fire pipeline-level alerts. Args: pipeline_id: Unique identifier for…, Initialize the AlertManager. Args: db_alerts_config: Database accessor for… (+1 more)

### Community 29 - "util/__init__.py"
Cohesion: 0.25
Nodes (9): Utility functions and decorators for pipeline steps. Transform decorators: -…, auto_dict_input(), dict_to_sns(), object_to_dict(), Any, Transform decorators for pipeline steps. Provides decorators for converting…, Decorator that converts any object arguments to dicts. Args: func: Function to…, Recursively convert a dictionary to SimpleNamespace. Args: data: Data to… (+1 more)

### Community 30 - "utils.py"
Cohesion: 0.28
Nodes (8): clean_for_json(), Any, YAML utilities for reading and writing configuration files., Recursively convert non-serializable objects to strings for JSON compatibility.…, Read a YAML file. Args: file_path: Path to the YAML file. verbose: Enable…, Write data to a YAML file. Args: file_path: Path to the output file. data:…, read_yaml(), write_yaml()

### Community 31 - "log/__init__.py"
Cohesion: 0.33
Nodes (5): Logging utilities for wpipe., new_logger(), Any, Logging utilities for wpipe., Create and configure a new logger instance. Args: process_name: Name for the…

### Community 32 - "pipe/__init__.py"
Cohesion: 0.33
Nodes (4): Pipeline module for orchestrating task execution., PipelineAsync, Minimal PipelineAsync for Phase 1., Async wrapper for Pipeline.

### Community 33 - "config/config.py"
Cohesion: 0.50
Nodes (3): crear_archivo(), obtener_usuario(), Obtiene el nombre de usuario usando 'whoami'.

### Community 35 - "run_training.py"
Cohesion: 0.50
Nodes (3): main(), Worker Executor Script. This script runs inside a container and performs the…, Main execution entry point for the training trial. Reads configuration,…

### Community 37 - "constants.py"
Cohesion: 0.50
Nodes (3): Codes, Error codes and execution constants for WPipe., Standard task and process status codes.

## Knowledge Gaps
- **10 isolated node(s):** `graphState`, `translations`, `pipelineTabs`, `currentAlerts`, `mount-cifs.sh script` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pipeline` connect `Pipeline` to `states/__init__.py`, `pipe/__init__.py`, `.__init__`, `.set_steps`, `pipe.py`, `.link_to_pipeline`, `.steps`, `APIClient`, `TaskError`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Why does `PipelineTracker` connect `PipelineTracker` to `QueryManager`, `Pipeline`, `.__init__`, `PipelineAsync`, `AnalysisManager`, `SystemMetricsCollector`, `tracker_models.py`, `AlertManager`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `config_pipeline()` connect `states/__init__.py` to `pipe.py`, `Pipeline`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `PipelineTracker` (e.g. with `.__init__()` and `.__init__()`) actually correct?**
  _`PipelineTracker` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Pipeline` (e.g. with `config_pipeline()` and `PipelineAsync`) actually correct?**
  _`Pipeline` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `QueryManager` (e.g. with `alerts_config` and `alerts_fired`) actually correct?**
  _`QueryManager` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `AlertManager` (e.g. with `AlertConfigModel` and `AlertFiredModel`) actually correct?**
  _`AlertManager` has 15 INFERRED edges - model-reasoned connections that need verification._