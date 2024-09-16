import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from mlflow import get_tracking_uri
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from pathlib import Path    #to access path to store aritfact location
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


#evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
def register_model(model_name, model_uri):
    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except Exception as e:
        print(f"Model already exists or error occurred: {e}")
    client.create_model_version(name=model_name, source=model_uri, run_id=mlflow.active_run().info.run_id)

def promote_model_to_production(model_name):
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )
    print(f"Model {model_name} version {latest_version} promoted to production.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("red-wine-quality.csv")
    #os.mkdir("data/")
    data.to_csv("data/red-wine-quality.csv", index=False)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="./mlops_exp")

    print("The set tracking uri is ", mlflow.get_tracking_uri())
    exp = mlflow.set_experiment(experiment_name="experiment_mlops")
    #get_exp = mlflow.get_experiment(exp_id)

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    with mlflow.start_run() as run:

        tags = {
            "engineering": "ML platform",
            "release.candidate":"RC1",
            "release.version": "2.0"
        }


        mlflow.set_tags(tags)
        mlflow.autolog(
            log_input_examples=True
        )
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_artifact("red-wine-quality.csv")

        artifacts_uri=mlflow.get_artifact_uri()
        print("The artifact path is",artifacts_uri )
        # Register the model
        register_model(model_name="ElasticNet_Wine_Model",
                       model_uri=f"mlruns/{run.info.experiment_id}/{run.info.run_id}/artifacts/ElasticNet_Wine_Model")

        # Promote the model to production
        promote_model_to_production(model_name="ElasticNet_Wine_Model")
        mlflow.end_run()
        run = mlflow.last_active_run()
        print("Active run id is {}".format(run.info.run_id))
        print("Active run name is {}".format(run.info.run_name))







