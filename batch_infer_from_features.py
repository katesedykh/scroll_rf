"""
Batch inference from features 

Usage:
  python batch_infer_from_features.py \
    --features processed_data/features.parquet \
    --model runs/model.pkl \
    --feature-names runs/feature_names.pkl \
    --label-map runs/label_map.pkl \
    --threshold runs/threshold.json \
    --out predictions.csv
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

def load_pickle(p): 
    with open(p, "rb") as f: 
        return pickle.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--feature-names", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--threshold", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.features) if args.features.endswith(".parquet") else pd.read_csv(args.features)
    pipe = load_pickle(Path(args.model))
    feature_names = load_pickle(Path(args.feature_names))
    print("feature_names", feature_names)
    label_map = load_pickle(Path(args.label_map))
    inv_label_map = {v:k for k,v in label_map.items()}
    th = json.loads(Path(args.threshold).read_text())

    X = df[feature_names].copy()
    ids = df["id"].astype(str)
    has_labels = "label" in df.columns
    y = df["label"].map(label_map).astype(int).to_numpy() if has_labels else None

    proba = pipe.predict_proba(X)
    classes = list(getattr(pipe, "classes_", []))
    pos_class = int(th["pos_class"])
    pos_idx = classes.index(pos_class)
    p_pos = proba[:, pos_idx]

    y_pred_raw = pipe.predict(X).astype(int)

    thr = float(th["threshold"])
    y_pred_thr = (p_pos >= thr).astype(int)

    out = pd.DataFrame({
        "id": ids,
        "p_pos": p_pos,
        "pred_raw_id": y_pred_raw,
        "pred_raw_label": [inv_label_map.get(int(c), str(c)) for c in y_pred_raw],
        "pred_thr_id": y_pred_thr,
        "pred_thr_label": [inv_label_map.get(int(c), str(c)) for c in y_pred_thr],
    })
    if has_labels:
        out["true_label"] = df["label"]
        out["true_id"] = y

    out.to_csv(args.out, index=False)
    print(f"✓ Wrote {args.out}")

    if has_labels:
        print("\n== Metrics (raw class) ==")
        print("Accuracy:", accuracy_score(y, y_pred_raw))
        print("F1 (weighted):", f1_score(y, y_pred_raw, average="weighted"))
        print(classification_report(y, y_pred_raw, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y, y_pred_raw))

        print("\n== Metrics (thresholded on pos='bot') ==")
        print("Accuracy:", accuracy_score(y, y_pred_thr))
        print("F1 (weighted):", f1_score(y, y_pred_thr, average="weighted"))
        print(classification_report(y, y_pred_thr, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y, y_pred_thr))

if __name__ == "__main__":
    main()
