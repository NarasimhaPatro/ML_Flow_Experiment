import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

# Logging setup
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    np.random.seed(40)

    # Set MLflow tracking URI BEFORE run
    remote_server_uri = "https://dagshub.com/suraj.cse.28/ML_Flow_Experiment.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    # Read dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download dataset. Error: %s", e)

    # Train-test split
    train, test = train_test_split(data, random_state=40)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Parameters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Start MLflow run
    with mlflow.start_run():

        # Model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Predictions
        predicted_qualities = lr.predict(test_x)

        # Metrics
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Log params
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Explicit pip requirements (removes warning)
        pip_requirements = [
            "mlflow",
            "scikit-learn==1.8.0",
            "skops==0.13.0",
            "pandas",
            "numpy"
        ]

        # Detect tracking type
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Log model - specify signature and input example if desired
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model=lr,
                name="model",
                registered_model_name="ElasticnetWineModel",
                serialization_format="skops",
                pip_requirements=pip_requirements
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=lr,
                name="model",
                serialization_format="skops",
                pip_requirements=pip_requirements
            )