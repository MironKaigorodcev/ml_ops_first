# src/train_model.py
import argparse, joblib, pandas as pd, numpy as np
from pathlib import Path
import mlflow, mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss

def load_processed(processed_dir, artifacts_dir):
    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)
    X_train = pd.read_parquet(processed_dir / "X_train.parquet")
    X_test  = pd.read_parquet(processed_dir / "X_test.parquet")
    y_train = pd.read_parquet(processed_dir / "y_train.parquet")["target"].values
    y_test  = pd.read_parquet(processed_dir / "y_test.parquet")["target"].values
    feature_names = pd.read_parquet(artifacts_dir / "feature_names.parquet")["feature"].tolist()
    X_train, X_test = X_train[feature_names], X_test[feature_names]
    return X_train, X_test, y_train, y_test, feature_names

def build_model(cfg):
    name = cfg["model_name"]
    if name == "logreg":
        return LogisticRegression(max_iter=cfg.get("max_iter", 1000), C=cfg.get("C", 1.0))
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", None),
            random_state=cfg.get("seed", 42),
        )
    if name == "gb":
        return GradientBoostingClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth", 3),
            random_state=cfg.get("seed", 42),
        )
    raise ValueError(f"unknown model: {name}")

def eval_and_log(y_true, y_prob, y_pred):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
    mlflow.log_metrics(metrics)
    return metrics

def run_one_experiment(X_train, X_test, y_train, y_test, cfg, args):
    models_dir = Path(args.models_dir)
    artifacts_dir = Path(args.artifacts_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=cfg["model_name"], nested=False):
        mlflow.log_params({**cfg, "n_features": X_train.shape[1]})
        clf = build_model(cfg)
        clf.fit(X_train, y_train)

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            z = clf.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-z))
        else:
            y_prob = None

        y_pred = clf.predict(X_test) if y_prob is None else (y_prob >= 0.5).astype(int)
        metrics = eval_and_log(y_test, y_prob, y_pred)

        # сохраняем как pickle в models_dir и логируем в MLflow
        pkl_path = models_dir / f"model_{cfg['model_name']}.pkl"
        joblib.dump(clf, pkl_path)
        mlflow.log_artifact(str(pkl_path))

        # дублируем в формате MLflow (удобно для деплоя)
        try:
            mlflow.sklearn.log_model(clf, artifact_path=f"model_{cfg['model_name']}")
        except Exception:
            pass

        print(f"{cfg['model_name']} -> {metrics}")

def main(args):
    mlflow.set_experiment(args.experiment)
    X_train, X_test, y_train, y_test, _ = load_processed(args.processed_dir, args.artifacts_dir)

    grids = [
        {"model_name": "logreg", "max_iter": 1000, "C": 1.0, "seed": args.seed},
        {"model_name": "logreg", "max_iter": 1000, "C": 0.5, "seed": args.seed},
        {"model_name": "rf", "n_estimators": 300, "max_depth": None, "seed": args.seed},
        {"model_name": "rf", "n_estimators": 500, "max_depth": 5, "seed": args.seed},
        {"model_name": "gb", "n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "seed": args.seed},
        {"model_name": "gb", "n_estimators": 400, "learning_rate": 0.05, "max_depth": 3, "seed": args.seed},
    ]
    for cfg in grids:
        run_one_experiment(X_train, X_test, y_train, y_test, cfg, args)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--models_dir", type=str, default="models")  # <-- добавили
    p.add_argument("--experiment", type=str, default="HeartDisease")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
