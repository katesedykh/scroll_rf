"""
- Uses StandardScaler + RandomForest in a Pipeline.
- Handles imbalance via class_weight="balanced_subsample".
- Calibrates probabilities (isotonic) and tunes threshold for the positive class ("bot" if present).
- Saves: model.pkl, feature_names.pkl, label_map.pkl, metrics.json, threshold.json
"""

import json, argparse, logging, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
L = logging.getLogger("train")

def save_pickle(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f: pickle.dump(o, f)

def load_pickle(p: Path):
    with open(p, "rb") as f: return pickle.load(f)

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "item") and callable(obj.item):  # catches np scalar subclasses
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="out/.../features.parquet or .csv")
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--balanced", action="store_true", default=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    fp = Path(args.features)
    df = pd.read_parquet(fp) if fp.suffix.lower()==".parquet" else pd.read_csv(fp)
    if "label" not in df.columns:
        raise ValueError("No 'label' column. Ensure extractor labeled from folders or provide a labeled dataset.")

    feat_names_p = fp.parent / "feature_names.pkl"
    label_map_p = fp.parent / "label_map.pkl"
    feature_cols = load_pickle(feat_names_p) if feat_names_p.exists() else [c for c in df.columns if c not in ("id","label")]
    label_map = load_pickle(label_map_p) if label_map_p.exists() else {lbl:i for i,lbl in enumerate(sorted(df["label"].unique()))}

    X = df[feature_cols].copy()
    y = df["label"].map(label_map).astype(int).to_numpy()

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    pre = ColumnTransformer([("num", StandardScaler(), feature_cols)], remainder="drop", verbose_feature_names_out=False)
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight=("balanced_subsample" if args.balanced else None),
    )
    base = Pipeline([("pre", pre), ("clf", rf)])

    L.info("Fitting calibrated classifier (isotonic)...")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(Xtr, ytr)

    def eval_split(X_, y_) -> dict:
        y_pred = clf.predict(X_)
        proba = clf.predict_proba(X_) if hasattr(clf, "predict_proba") else None
        out = {
            "accuracy": float(accuracy_score(y_, y_pred)),
            "f1_weighted": float(f1_score(y_, y_pred, average="weighted")),
            "report": classification_report(y_, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_, y_pred).tolist(),
        }
        if proba is not None:
            out["classes"] = list(getattr(clf, "classes_", []))
        return out

    metrics = {"train": eval_split(Xtr, ytr), "val": eval_split(Xval, yval)}
    print(metrics)
    (run_dir / "metrics.json").write_text(json.dumps(make_json_safe(metrics), indent=2, ensure_ascii=False))

    # Tune threshold for positive class (bot)
    classes = list(getattr(clf, "classes_", []))
    inv_label_map = {v:k for k,v in label_map.items()}
    pos_class = None
    for c in classes:
        if inv_label_map.get(int(c), "").lower() == "bot":
            pos_class = int(c)
            break
    if pos_class is None and classes:
        pos_class = int(max(classes)) 

    if classes:
        pos_idx = classes.index(pos_class)
        p_val = clf.predict_proba(Xval)[:, pos_idx]
        prec, rec, thr = precision_recall_curve(yval, p_val, pos_label=pos_class)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        best_idx = int(np.argmax(f1))
        best_thr = float(thr[max(0, min(best_idx, len(thr)-1))])
    else:
        best_thr = 0.5
        pos_class = 1

    save_pickle(clf, run_dir / "model.pkl")
    save_pickle(feature_cols, run_dir / "feature_names.pkl")
    save_pickle(label_map, run_dir / "label_map.pkl")
    (run_dir / "threshold.json").write_text(json.dumps({"pos_class": int(pos_class), "threshold": float(best_thr)}, indent=2))

    L.info(f"Saved model to {run_dir}. Tuned threshold for class '{inv_label_map.get(pos_class, pos_class)}' = {best_thr:.4f}")

if __name__ == "__main__":
    main()
