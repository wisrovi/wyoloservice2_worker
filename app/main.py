from application.executor import train_run

if __name__ == "__main__":
    resultado = train_run(
        config_path="/app/config.yaml",
        trial_number=1,
        verbose=False,
        # script_path="/app/lib/src/wyolo/core/",
    )
    if resultado is not None:
        print(f"Resultado del entrenamiento: {resultado}")
