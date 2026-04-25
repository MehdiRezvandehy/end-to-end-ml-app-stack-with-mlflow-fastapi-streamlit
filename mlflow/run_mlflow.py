import argparse

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix
import matplotlib
import pylab as plt
import numpy as np
import pandas as pd
import joblib
import mlflow
import yaml
import logging
from mlflow.tracking import MlflowClient
import platform
import sklearn

print("------------------")
# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Load trained model parameters and performance from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()


# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']

    if args.mlflow_tracking_uri:
        logger.info(f"Load mlflow_tracking_uri: {args.mlflow_tracking_uri}")
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg['name'])

    # Start MLflow run
    with mlflow.start_run(run_name="final_trained_model"):
        logger.info(f"The best model: {model_cfg['best_model']}")

        # Log params and metrics
        mlflow.log_params(model_cfg['parameters'])
        accuracy = model_cfg['accuracy']
        mlflow.log_metrics({'accuracy': accuracy})
        precision = model_cfg['precision']
        mlflow.log_metrics({'precision': accuracy})
        recall = model_cfg['recall']
        mlflow.log_metrics({'recall': accuracy})
        f1_score = model_cfg['f1_score']
        mlflow.log_metrics({'f1_score': accuracy})

        # Log and register model
        model_name = model_cfg['name']
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"

        logger.info("Registering model to MLflow Model Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  # already exists

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )

        # Transition model to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        # Add a human-readable description
        description = (
            f"Model for predicting buidling energy load.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All features in the dataset except the target variable\n"
            f"Target variable: {model_cfg['target_variable']}\n"
            f"Features: {model_cfg['features']}\n"
            f"Model saved at: {args.models_dir}/model.pkl\n"
            f"Performance metrics:\n"
            f"  - accuracy: {accuracy:.2f}\n"
            f"  - precision: {precision:.2f}\n"
            f"  - recall: {recall:.2f}\n"
            f"  - f1_score: {f1_score:.2f}\n"                        
        )
        client.update_registered_model(name=model_name, description=description)

        # Add tags for better organization
        client.set_registered_model_tag(model_name, "algorithm", model_cfg['best_model'])
        client.set_registered_model_tag(model_name, "hyperparameters", str(model_cfg['parameters']))
        client.set_registered_model_tag(model_name, "training features", model_cfg['features'])
        client.set_registered_model_tag(model_name, "target_variable", model_cfg['target_variable'])
        #client.set_registered_model_tag(model_name, "training_dataset", args.data)
        client.set_registered_model_tag(model_name, "model_path", f"{args.models_dir}/model.pkl")

        # Add dependency tags
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in deps.items():
            client.set_registered_model_tag(model_name, k, v)


if __name__ == "__main__":
    args = parse_args()
    main(args)
