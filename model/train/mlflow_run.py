import mlflow

mlflow.set_experiment("malicious-url-prediction")
run = mlflow.start_run()
mlflow.end_run()
print(run.info.run_id)