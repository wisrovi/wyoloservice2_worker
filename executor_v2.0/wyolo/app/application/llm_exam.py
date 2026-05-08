import mlflow
import pandas as pd
import os
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def analyze_yolov8_training(results_csv_path=None, confusion_matrix_path=None):
    """Analyze YOLOv8 training results using Ollama"""
    
    llm = OllamaLLM(base_url="http://192.168.20.52:11434", model="llama3.1:8b")
    
    analysis_prompt = PromptTemplate.from_template("""
    Eres un experto en análisis de entrenamiento de YOLOv8. Analiza los siguientes resultados de entrenamiento y proporciona:
    
    1. Evaluación general del rendimiento (malo/regular/bueno/excelente)
    2. Métricas clave: mAP, precision, recall, loss
    3. Problemas detectados (overfitting, underfitting, etc.)
    4. Recomendaciones para mejorar
    
    Resultados del entrenamiento:
    {training_data}
    
    Matriz de confusión (si disponible):
    {confusion_data}
    
    Proporciona un análisis detallado y conciso:
    """)
    
    # Load training results from CSV if path provided
    training_data = "No se proporcionaron datos de entrenamiento"
    if results_csv_path and os.path.exists(results_csv_path):
        df = pd.read_csv(results_csv_path)
        training_data = df.to_string()
    
    # Load confusion matrix info if path provided  
    confusion_data = "No se proporcionó matriz de confusión"
    if confusion_matrix_path and os.path.exists(confusion_matrix_path):
        confusion_data = f"Matriz de confusión disponible en: {confusion_matrix_path}"
    
    chain = analysis_prompt | llm
    
    with mlflow.start_run(run_name="yolov8_training_analysis"):
        mlflow.log_param("model_type", "YOLOv8")
        mlflow.log_param("analysis_method", "Ollama_LLM")
        
        if results_csv_path:
            mlflow.log_artifact(results_csv_path, "training_results")
        if confusion_matrix_path:
            mlflow.log_artifact(confusion_matrix_path, "confusion_matrix")
        
        result = chain.invoke({
            "training_data": training_data,
            "confusion_data": confusion_data
        })
        
        mlflow.log_text(result, "analysis_report.txt")
        
        return result

def analyze_mlflow_yolov8_experiments():
    """Analyze YOLOv8 experiments from MLflow"""
    
    # Get all experiments
    experiments = mlflow.search_experiments()
    
    llm = OllamaLLM(base_url="http://192.168.20.52:11434", model="llama3.1:8b")
    
    prompt = PromptTemplate.from_template("""
    Analiza los siguientes experimentos de YOLOv8 en MLflow:
    
    Experimentos disponibles:
    {experiment_info}
    
    Proporciona:
    1. Resumen de los experimentos
    2. Comparación de métricas principales
    3. Mejores prácticas identificadas
    4. Sugerencias para próximos entrenamientos
    """)
    
    experiment_info = []
    for exp in experiments:
        experiment_info.append(f"Experimento: {exp.name} (ID: {exp.experiment_id})")
        try:
            runs = mlflow.search_runs(exp.experiment_id)
            if len(runs) > 0:
                for run in runs:
                    if hasattr(run, 'data'):
                        metrics = run.data.metrics if hasattr(run.data, 'metrics') else {}
                        params = run.data.params if hasattr(run.data, 'params') else {}
                        run_id = run.info.run_id if hasattr(run.info, 'run_id') else str(run.get('run_id', 'N/A'))
                        status = run.info.status if hasattr(run.info, 'status') else str(run.get('status', 'N/A'))
                        
                        experiment_info.append(f"  Run ID: {run_id}")
                        experiment_info.append(f"  Estado: {status}")
                        if metrics:
                            experiment_info.append(f"  Métricas: {dict(list(metrics.items())[:5])}")
                        if params:
                            experiment_info.append(f"  Parámetros: {dict(list(params.items())[:3])}")
            else:
                experiment_info.append("  No runs found")
        except Exception as e:
            experiment_info.append(f"  Error getting runs: {e}")
    
    chain = prompt | llm
    
    with mlflow.start_run(run_name="yolov8_experiments_overview"):
        result = chain.invoke({
            "experiment_info": "\n".join(experiment_info)
        })
        
        mlflow.log_text(result, "experiments_analysis.txt")
        return result

if __name__ == "__main__":
    # Example usage
    print("=== Análisis de Experimentos MLflow ===")
    try:
        mlflow_analysis = analyze_mlflow_yolov8_experiments()
        print(mlflow_analysis)
    except Exception as e:
        print(f"Error analizando MLflow: {e}")
    
    # Example with specific files (uncomment when you have the files)
    # csv_path = "path/to/your/results.csv"
    # confusion_path = "path/to/your/confusion_matrix.png" 
    # analysis = analyze_yolov8_training(csv_path, confusion_path)
    # print(analysis)