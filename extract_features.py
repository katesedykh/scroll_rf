"""
Outputs in --out-dir:
  - features.parquet / .csv
  - feature_names.pkl
  - label_map.pkl            # {"bot":1,"human":0}
  - preproc_constants.json
"""

import os, json, argparse, logging, pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
L = logging.getLogger("extract")

DEFAULT_MICRO_SCROLL_PX = 5.0
DEFAULT_FAST_VEL_PX_PER_MS = 0.8
USE_ABS_VELOCITY_FOR_FAST = True
TIMESTAMPS_IN_SECONDS = False

def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _read_json(path: Path) -> Any:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return json.loads(txt)

def _locate_scroll_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Supports either:
      - [ { "incoming_data": { "scrollData": [...] }, ... } ]
      - [ { "scroll_data": [...] , ... } ]
      - { "scroll_data": [...] } or { "incoming_data": { "scrollData": [...] } }
    Returns: list of events dicts with keys including ts,y,(scrollHeight,viewportHeight)
    """
    root = obj[0] if isinstance(obj, list) and obj else obj
    if not isinstance(root, dict):
        raise ValueError("Unsupported JSON shape")

    # nested camelCase
    inc = root.get("incoming_data") or root.get("incomingData")
    if isinstance(inc, dict) and isinstance(inc.get("scrollData"), list):
        return inc["scrollData"]

    # top-level snake_case
    if isinstance(root.get("scroll_data"), list):
        return root["scroll_data"]

    if isinstance(root.get("scrollData"), list):
        return root["scrollData"]

    raise ValueError("Could not find scroll list (incoming_data.scrollData / scroll_data)")

def _to_frame(scroll_list: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(scroll_list)
    # normalize common variants
    rename_map = {}
    for c in list(df.columns):
        cl = c.lower()
        if cl in ("ts_ms",):
            rename_map[c] = "ts"
        elif cl in ("timestamp","time","t"):
            rename_map[c] = "ts"
        elif cl in ("deltay","delta_y"):
            rename_map[c] = "deltaY"
        elif cl in ("scrollheight","docheight","documentheight","sh"):
            rename_map[c] = "scrollHeight"
        elif cl in ("viewportheight","innerheight","vh"):
            rename_map[c] = "viewportHeight"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    if "ts" not in df.columns:
        raise ValueError("Missing 'ts' in events")
    if "y" not in df.columns and "scrollTop" in df.columns:
        df["y"] = df["scrollTop"]
    if "y" not in df.columns:
        raise ValueError("Missing 'y' (or 'scrollTop') in events")

    if "scrollHeight" not in df.columns:
        df["scrollHeight"] = np.nan
    if "viewportHeight" not in df.columns:
        df["viewportHeight"] = np.nan

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["scrollHeight"] = pd.to_numeric(df["scrollHeight"], errors="coerce")
    df["viewportHeight"] = pd.to_numeric(df["viewportHeight"], errors="coerce")
    df = df.dropna(subset=["ts","y"]).reset_index(drop=True)
    return df[["ts","y","scrollHeight","viewportHeight"]]

def _compute_features_leakage_safe(
    df: pd.DataFrame,
    *,
    micro_scroll_px: float,
    fast_vel_px_per_ms: float,
    use_abs_velocity_for_fast: bool,
    timestamps_in_seconds: bool,
) -> Dict[str, float]:

    ts = df["ts"].to_numpy(dtype=np.float64)
    if timestamps_in_seconds:
        ts = ts * 1000.0
    y = df["y"].to_numpy(dtype=np.float64)
    sh = df["scrollHeight"].to_numpy(dtype=np.float64)
    vh = df["viewportHeight"].to_numpy(dtype=np.float64)
    n = len(ts)

    dy = np.diff(y, prepend=y[0]) if n else np.array([], dtype=np.float64)
    duration_ms = float(np.nanmax(ts) - np.nanmin(ts)) if n else 0.0

    if n > 1:
        dt = np.diff(ts)
        dt[dt <= 0] = np.nan
        vel = np.divide(np.diff(y), dt, out=np.zeros_like(dt), where=np.isfinite(dt) & (dt != 0))
    else:
        dt = np.array([], dtype=np.float64)
        vel = np.array([], dtype=np.float64)

    def _iqr(x):
        if x.size == 0:
            return 0.0
        q75, q25 = np.nanpercentile(x, [75, 25])
        return float(q75 - q25)

    vh_med = float(np.nanmedian(vh)) if np.isfinite(vh).any() else np.nan
    sh_med = float(np.nanmedian(sh)) if np.isfinite(sh).any() else np.nan
    vh_over_sh = float(vh_med / sh_med) if sh_med and np.isfinite(sh_med) and sh_med != 0 else np.nan

    feats = {
        #"n_events": float(n),
        "duration_ms": float(duration_ms),
        "delta_y_mean": float(np.nanmean(dy)) if n else 0.0,
        "delta_y_std": float(np.nanstd(dy, ddof=0)) if n else 0.0,
        "delta_y_med": float(np.nanmedian(dy)) if n else 0.0,
        "delta_y_iqr": _iqr(dy),
        "time_diff_mean_ms": float(np.nanmean(dt)) if dt.size else 0.0,
        "time_diff_std_ms": float(np.nanstd(dt, ddof=0)) if dt.size else 0.0,
        "time_diff_med_ms": float(np.nanmedian(dt)) if dt.size else 0.0,
        "time_diff_iqr_ms": _iqr(dt),
        "velocity_mean_px_per_ms": float(np.nanmean(vel)) if vel.size else 0.0,
        "velocity_std_px_per_ms": float(np.nanstd(vel, ddof=0)) if vel.size else 0.0,
        "velocity_med_px_per_ms": float(np.nanmedian(vel)) if vel.size else 0.0,
        "total_abs_scroll_px": float(np.nansum(np.abs(dy))),
        "net_scroll_px": float(np.nansum(dy)),
        "prop_micro_scrolls": float(np.mean(np.abs(dy) < micro_scroll_px)) if n else 0.0,
        "prop_fast_intervals": float(
            np.mean((np.abs(vel) if use_abs_velocity_for_fast else vel) > fast_vel_px_per_ms)
        ) if vel.size else 0.0,
        "direction_changes": float(np.nansum(np.diff(np.sign(dy)) != 0)) if n > 1 else 0.0,
        "vh_over_sh_med": vh_over_sh,
    }
    return feats

def _iter_json(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower()==".json" else []
    return [p for p in root.rglob("*.json") if p.is_file()]

def _label_from_parent(p: Path, human_name: str, bot_name: str) -> str:
    parent = p.parent.name.lower()
    if parent == human_name.lower():
        return "human"
    if parent == bot_name.lower():
        return "bot"
    return None 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dir or single .json file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--human-folder-name", default="human")
    ap.add_argument("--bot-folder-name", default="bot")
    ap.add_argument("--micro-scroll-px", type=float, default=DEFAULT_MICRO_SCROLL_PX)
    ap.add_argument("--fast-vel-px-per-ms", type=float, default=DEFAULT_FAST_VEL_PX_PER_MS)
    ap.add_argument("--use-abs-velocity-for-fast", action="store_true", default=USE_ABS_VELOCITY_FOR_FAST)
    ap.add_argument("--timestamps-in-seconds", action="store_true", default=TIMESTAMPS_IN_SECONDS)
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    constants = {
        "MICRO_SCROLL_PX": float(args.micro_scroll_px),
        "FAST_VEL_PX_PER_MS": float(args.fast_vel_px_per_ms),
        "USE_ABS_VELOCITY_FOR_FAST": bool(args.use_abs_velocity_for_fast),
        "TIMESTAMPS_IN_SECONDS": bool(args.timestamps_in_seconds),
    }
    (out_dir / "preproc_constants.json").write_text(json.dumps(constants, indent=2))

    rows: List[Dict[str, Any]] = []
    files = _iter_json(src)
    if not files:
        L.error("No JSON files found."); return

    for f in files:
        try:
            obj = _read_json(f)
            scroll = _locate_scroll_list(obj)
            df = _to_frame(scroll)
            feats = _compute_features_leakage_safe(
                df,
                micro_scroll_px=constants["MICRO_SCROLL_PX"],
                fast_vel_px_per_ms=constants["FAST_VEL_PX_PER_MS"],
                use_abs_velocity_for_fast=constants["USE_ABS_VELOCITY_FOR_FAST"],
                timestamps_in_seconds=constants["TIMESTAMPS_IN_SECONDS"],
            )
            row = {"id": str(f)}
            lbl = _label_from_parent(f, args.human_folder_name, args.bot_folder_name)
            if lbl is not None:
                row["label"] = lbl
            row.update(feats)
            rows.append(row)
        except Exception as e:
            L.warning(f"Skip {f}: {e}")

    if not rows:
        L.error("No rows extracted after parsing."); return

    feat_df = pd.DataFrame(rows)
    print(feat_df.head())
    print(feat_df.tail())

    front = ["id"] + (["label"] if "label" in feat_df.columns else [])
    feature_cols = [c for c in feat_df.columns if c not in front]
    feat_df = feat_df[front + feature_cols]

    feat_df.to_parquet(out_dir / "features.parquet", index=False)
    feat_df.to_csv(out_dir / "features.csv", index=False)

    save_pickle(feature_cols, out_dir / "feature_names.pkl")

    if "label" in feat_df.columns:
        uniq = sorted(feat_df["label"].unique().tolist())
        if set(uniq) == {"bot","human"}:
            label_map = {"human":0, "bot":1}
        else:
            label_map = {lbl:i for i,lbl in enumerate(uniq)}
        save_pickle(label_map, out_dir / "label_map.pkl")

    L.info(f"Saved features to {out_dir}")

if __name__ == "__main__":
    main()
