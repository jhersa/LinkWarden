#!/bin/bash

export MLFLOW_TRACKING_URI=http://localhost:5555

MLFLOW_RUN_ID=$(python model/train/mlflow_run.py)

export MLFLOW_RUN_ID=$MLFLOW_RUN_ID

python model/train/download_dataset.py

python model/train/preprocessing.py

python model/train/train_model.py