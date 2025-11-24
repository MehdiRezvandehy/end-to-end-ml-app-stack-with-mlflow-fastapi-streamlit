import argparse
import sys
print(sys.executable)
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib
import pylab as plt
import numpy as np
import pickle
import yaml
import logging

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    return parser.parse_args()

# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # Load data
    df = pd.read_csv(args.data, na_values=['NA', '?', ' '])
    logger.info(f"Sample Energy Efficiency Dataset:\n{df.head()}")

    np.random.seed(32)
    df = df.reindex(np.random.permutation(df.index))

    df['Binary Classes'] = df['Binary Classes'].replace({'Low Level': 0, 'High Level': 1})

    # Training and Test
    spt = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in spt.split(df, df['Multi-Classes']):
        train_set_strat = df.loc[train_idx].reset_index(drop=True)
        test_set_strat = df.loc[test_idx].reset_index(drop=True)

    train_set_strat.drop(['Heating Load', 'Multi-Classes'], axis=1, inplace=True)
    test_set_strat.drop(['Heating Load', 'Multi-Classes'], axis=1, inplace=True)

    clmns = list(train_set_strat.drop(['Binary Classes'], axis=1).columns)
    logger.info(f"Features for training: {clmns}")

    # Standardization
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(train_set_strat.drop(['Binary Classes'], axis=1))
    y_train = train_set_strat['Binary Classes']

    x_test_std = scaler.transform(test_set_strat.drop(['Binary Classes'], axis=1))
    y_test = test_set_strat['Binary Classes']

    # Fine-tune RandomForest
    rf = RandomForestClassifier(random_state=42)

    param_dist = {
        'n_estimators': [50, 100, 200, 300, 400],
        'max_depth': [10, 20, 40, 60, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy', 'log_loss']
    }

    rf_search_cv = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=2,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    # Fit model
    rf_search_cv.fit(x_train_std, y_train)

    best_model = rf_search_cv.best_estimator_
    best_params = rf_search_cv.best_params_
    best_score = rf_search_cv.best_score_

    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best Cross-Validation Score: {best_score}")

    # Create YAML-style features block
    yaml_features_block = "\n" + "\n".join([f"  - {col}" for col in clmns])

    rf_fine_tuned = RandomForestClassifier(random_state=42, **best_params)
    rf_fine_tuned.fit(x_train_std, y_train)

    y_pred = rf_fine_tuned.predict(x_test_std)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1_score = f1_score(y_test, y_pred)

    # Build YAML dictionary
    model_info = {
        "model": {
            "best_model": best_model.__class__.__name__,
            "name": "energy_load_classifier",
            "accuracy": float(np.round(test_accuracy, 2)),
            "precision": float(np.round(test_precision, 2)),
            "recall": float(np.round(test_recall, 2)),
            "f1_score": float(np.round(test_f1_score, 2)),
            "parameters": best_params,
            "target_variable": "energy_load_category",
            "features": yaml_features_block
        }
    }

    logger.info(f"Model yaml file: {args.config}")

    # Write YAML file
    with open(args.config, "w") as file:
        yaml.dump(model_info, file, sort_keys=False)

    # Save model and scaler
    model_pickle = f"{args.models_dir}/model.pkl"
    logger.info(f"Model pickle file: {model_pickle}")

    with open(model_pickle, "wb") as f:
        pickle.dump(rf_search_cv, f)

    scaler_pickle = f"{args.models_dir}/scaler.pkl"
    logger.info(f"Scaler pickle file: {scaler_pickle}")

    with open(scaler_pickle, "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
