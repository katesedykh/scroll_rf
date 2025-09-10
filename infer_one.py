"""
Usage:
  python infer_one.py \
    --model runs/rf_v1/model.pkl \
    --file sample.json \
    --feature-names runs/rf_v1/feature_names.pkl \
    --label-map runs/rf_v1/label_map.pkl \
    --constants out/fe_v1/preproc_constants.json \
    --threshold runs/rf_v1/threshold.json
"""

import json, argparse, logging, pickle
from pathlib import Path
import numpy as np
import pandas as pd

from extract_features import _read_json, _locate_scroll_list, _to_frame, _compute_features_leakage_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
L = logging.getLogger("infer")

def load_pickle(p: Path):
    with open(p, "rb") as f: return pickle.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--feature-names", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--constants", required=True, help="preproc_constants.json from extractor")
    ap.add_argument("--threshold", required=True, help="threshold.json from training")
    ap.add_argument("--min-events", type=int, default=8)
    args = ap.parse_args()

    pipe = load_pickle(Path(args.model))
    feature_names = load_pickle(Path(args.feature_names))
    label_map = load_pickle(Path(args.label_map))
    inv_label_map = {v:k for k,v in label_map.items()}
    constants = json.loads(Path(args.constants).read_text())
    threshold = json.loads(Path(args.threshold).read_text())

    obj = _read_json(Path(args.file))
    scroll = _locate_scroll_list(obj)
    df = _to_frame(scroll)
    feats = _compute_features_leakage_safe(
        df,
        micro_scroll_px=float(constants["MICRO_SCROLL_PX"]),
        fast_vel_px_per_ms=float(constants["FAST_VEL_PX_PER_MS"]),
        use_abs_velocity_for_fast=bool(constants["USE_ABS_VELOCITY_FOR_FAST"]),
        timestamps_in_seconds=bool(constants["TIMESTAMPS_IN_SECONDS"]),
    )

    X = pd.DataFrame([{k: feats.get(k, np.nan) for k in feature_names}], columns=feature_names).fillna(0.0)

    proba = pipe.predict_proba(X)[0]
    classes = list(getattr(pipe, "classes_", []))
    pred_raw = int(pipe.predict(X)[0])
    pred_lbl = inv_label_map.get(pred_raw, str(pred_raw))

    # Thresholded decision for positive class
    pos_class = int(threshold["pos_class"])
    thr = float(threshold["threshold"])
    pos_idx = classes.index(pos_class)
    p_pos = float(proba[pos_idx])
    decision = inv_label_map.get(pos_class, "pos") if p_pos >= thr else [
        inv_label_map.get(c, "neg") for c in classes if c != pos_class
    ][0]

    print(f"Prediction(raw): {pred_lbl}")
    print(f"Thresholded decision: {decision} (p_pos={p_pos:.4f} >= {thr:.4f} ?)")
    prob_map = {inv_label_map.get(int(c), str(c)): float(p) for c,p in zip(classes, proba)}
    print("Probabilities:", json.dumps(prob_map, indent=2))

if __name__ == "__main__":
    main()
