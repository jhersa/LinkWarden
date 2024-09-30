import os

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

MLFLOW_RUN_ID = os.environ["MLFLOW_RUN_ID"]

with mlflow.start_run(run_id=MLFLOW_RUN_ID):
    client = MlflowClient()
    data = client.download_artifacts(MLFLOW_RUN_ID, "malicious_phish_preprocessed.csv")

    data = pd.read_csv(data)

    X = data.drop(['url','type','Category','domain'],axis=1)
    y = data['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    accuracy = []
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(pred, y_test)
    accuracy.append(acc)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 2)
    mlflow.log_param("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()