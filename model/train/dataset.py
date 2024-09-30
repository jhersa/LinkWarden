import os
import subprocess
import time
import mlflow

MLFLOW_RUN_ID = os.environ["MLFLOW_RUN_ID"]

# Download Kaggle dataset
directory = "data/datasets"
dataset_ref = "sid321axn/malicious-urls-dataset"

with mlflow.start_run(run_id=MLFLOW_RUN_ID):
    mlflow.log_param("dataset_ref", dataset_ref)
    t0 = time.time()

    # Check if the directory does not exist
    if not os.path.isdir(directory):
        # Clone the repository with a shallow copy
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_ref, "-p", directory])
        # Extract the contents of the zip file
        subprocess.run(["unzip", f"{directory}/{dataset_ref.split('/')[1]}.zip", "-d", directory])
    else:
        # Directory exists, print a message
        print(f"The dataset already exists in the `{directory}` folder.")

    t1 = time.time()

    mlflow.log_metric("dataset_download_time", t1 - t0)
    mlflow.log_artifact(directory)
    mlflow.end_run()
