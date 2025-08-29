import os
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def main(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.artifacts, exist_ok=True)

    # 1) читаем
    df = pd.read_csv(args.input)

    # 2) нормализация только числовых (кроме target)
    numeric_cols = df.drop(columns=[args.target]).select_dtypes(include="number").columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 3) X, y
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # 4) train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 5) сохраняем сплиты (parquet удобно; если нет pyarrow — можно заменить на csv)
    xtr_p = os.path.join(args.output, "X_train.parquet")
    xts_p = os.path.join(args.output, "X_test.parquet")
    ytr_p = os.path.join(args.output, "y_train.parquet")
    yts_p = os.path.join(args.output, "y_test.parquet")
    feats_p = os.path.join(args.artifacts, "feature_names.parquet")
    scaler_p = os.path.join(args.artifacts, "scaler.joblib")

    X_train.to_parquet(xtr_p, index=False)
    X_test.to_parquet(xts_p, index=False)
    pd.DataFrame({"target": y_train.values}).to_parquet(ytr_p, index=False)
    pd.DataFrame({"target": y_test.values}).to_parquet(yts_p, index=False)
    pd.DataFrame({"feature": X.columns}).to_parquet(feats_p, index=False)

    joblib.dump({"scaler": scaler, "numeric_cols": list(numeric_cols)}, scaler_p)

    # 6) paths
    print({
        "X_train": xtr_p,
        "X_test": xts_p,
        "y_train": ytr_p,
        "y_test": yts_p,
        "feature_names": feats_p,
        "scaler": scaler_p,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/heart.csv")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    main(args)
