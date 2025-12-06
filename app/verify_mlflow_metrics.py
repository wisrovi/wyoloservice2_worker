#!/usr/bin/env python3
"""
Verificar que las métricas del sistema se registraron en MLflow
"""

import mlflow


def check_system_metrics():
    """Verificar métricas del sistema en el último run"""
    try:
        # Conectar al MLflow local
        mlflow.set_tracking_uri("http://192.168.1.137:23435")

        # Obtener el experimento más reciente
        experiment = mlflow.get_experiment_by_name("example_clasification")
        if not experiment:
            print("❌ No se encontró el experimento")
            return

        # Obtener los runs del experimento
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            print("❌ No se encontraron runs")
            return

        run = runs[0]
        print(f"✅ Run encontrado: {run.info.run_id}")
        print(f"   Estado: {run.info.status}")
        print(f"   Duración: {run.info.end_time - run.info.start_time}")

        # Verificar métricas del sistema
        client = mlflow.MlflowClient()

        # Buscar métricas con prefijo 'system/'
        system_metrics = {}
        for key in run.data.metrics.keys():
            if key.startswith("system/"):
                system_metrics[key] = run.data.metrics[key]

        if system_metrics:
            print(f"✅ Métricas del sistema encontradas: {len(system_metrics)}")
            for name, value in list(system_metrics.items())[:5]:
                print(f"   - {name}: {value}")
        else:
            print("⚠️ No se encontraron métricas del sistema en el run")

        # Verificar artifacts del modelo
        artifacts = client.list_artifacts(run.info.run_id)
        model_artifacts = [a for a in artifacts if "model" in a.path.lower()]

        if model_artifacts:
            print(f"✅ Artifacts del modelo encontrados: {len(model_artifacts)}")
            for artifact in model_artifacts:
                print(f"   - {artifact.path}")
        else:
            print("⚠️ No se encontraron artifacts del modelo")

    except Exception as e:
        print(f"❌ Error verificando métricas: {e}")


if __name__ == "__main__":
    check_system_metrics()
