

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET = "observed_recovered"

FEATURES = [
    "amount",
    "payment_method",
    "failure_reason",
    "device",
    "location",
    "previous_success_rate",
    "days_since_last_payment",
    "subscription_age_days",
    "historical_retries",
    "time_since_failure_mins",
    "customer_value",
    "merchant_category",
    "failure_hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_salary_window",
    "network_quality",
    "customer_fatigue",
    "observed_intervention",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "failure_reason",
    "device",
    "location",
    "customer_value",
    "merchant_category",
    "observed_intervention",
]

NUMERIC_FEATURES = [
    c for c in FEATURES if c not in CATEGORICAL_FEATURES
]

ACTION_ORDER = [
    "retry_30m",
    "retry_evening",
    "payment_link",
    "whatsapp_reminder",
    "alternate_method",
    "stop",
]

# Training rows are deliberately capped only for the optional fast
# development mode. Final runs should use the complete training set.
DEFAULT_GPU_IDS = [0, 1, 2]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(
        f"[PID {os.getpid()}] "
        f"{msg}",
        flush=True,
    )


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(
        json.dumps(obj, indent=2, default=str),
        encoding="utf-8",
    )


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce CPU RAM footprint without changing semantics."""
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col],
                downcast="float",
            )
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col],
                downcast="integer",
            )
    return df


def load_raw_split(
    path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load only the columns needed for training."""
    cols = FEATURES + [TARGET]

    log(f"Loading {path.name}")

    df = pd.read_parquet(
        path,
        columns=cols,
    )

    if max_rows is not None and len(df) > max_rows:
        # Deterministic sampling for development mode.
        df = df.sample(
            n=max_rows,
            random_state=20260903,
        ).reset_index(drop=True)

    # Convert categorical columns to strings.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string")

    df = downcast_dataframe(df)

    return df


def split_xy(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[FEATURES].copy()
    y = df[TARGET].astype(np.int8).to_numpy()
    return X, y


def make_sklearn_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    One-hot encode the small categorical feature set once per worker.

    Since customer_id / merchant_id are excluded, categorical cardinality
    remains small and the resulting dense matrix is manageable.
    """
    train_x = train_df[FEATURES].copy()
    val_x = val_df[FEATURES].copy()

    # Convert categories to a stable common universe.
    for col in CATEGORICAL_FEATURES:
        train_x[col] = train_x[col].astype("string")
        val_x[col] = val_x[col].astype("string")

        categories = pd.Index(
            pd.concat(
                [
                    train_x[col],
                    val_x[col],
                ],
                ignore_index=True,
            ).dropna().unique()
        )

        train_x[col] = pd.Categorical(
            train_x[col],
            categories=categories,
        )
        val_x[col] = pd.Categorical(
            val_x[col],
            categories=categories,
        )

    train_x = pd.get_dummies(
        train_x,
        columns=CATEGORICAL_FEATURES,
        dtype=np.float32,
    )

    val_x = pd.get_dummies(
        val_x,
        columns=CATEGORICAL_FEATURES,
        dtype=np.float32,
    )

    val_x = val_x.reindex(
        columns=train_x.columns,
        fill_value=0,
    )

    feature_names = train_x.columns.tolist()

    X_train = np.ascontiguousarray(
        train_x.to_numpy(dtype=np.float32)
    )

    X_val = np.ascontiguousarray(
        val_x.to_numpy(dtype=np.float32)
    )

    y_train = train_df[TARGET].astype(np.int8).to_numpy()
    y_val = val_df[TARGET].astype(np.int8).to_numpy()

    del train_x, val_x
    gc.collect()

    return (
        X_train,
        X_val,
        y_train,
        feature_names,
    )


def metrics_from_probabilities(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Dict[str, float]:
    """Compute useful probability-quality metrics."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    probability = np.clip(
        probability,
        1e-7,
        1 - 1e-7,
    )

    prediction = (
        probability >= 0.5
    ).astype(np.int8)

    accuracy = float(
        np.mean(prediction == y_true)
    )

    result = {
        "roc_auc": float(
            roc_auc_score(y_true, probability)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probability)
        ),
        "log_loss": float(
            log_loss(y_true, probability)
        ),
        "brier": float(
            brier_score_loss(y_true, probability)
        ),
        "accuracy_at_0_5": accuracy,
        "positive_rate": float(np.mean(y_true)),
    }

    return result


# ---------------------------------------------------------------------
# Model 1 - CatBoost
# ---------------------------------------------------------------------

def train_catboost(
    gpu_id: int,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    threads: int,
    max_train_rows: int | None,
    max_val_rows: int | None,
) -> Dict:

    start = time.time()

    log(
        f"CATBOOST -> GPU {gpu_id} | "
        f"loading data"
    )

    from catboost import CatBoostClassifier, Pool

    train_df = load_raw_split(
        train_path,
        max_rows=max_train_rows,
    )
    val_df = load_raw_split(
        val_path,
        max_rows=max_val_rows,
    )

    X_train = train_df[FEATURES].copy()
    X_val = val_df[FEATURES].copy()

    y_train = train_df[TARGET].astype(np.int8).to_numpy()
    y_val = val_df[TARGET].astype(np.int8).to_numpy()

    for col in CATEGORICAL_FEATURES:
        X_train[col] = X_train[col].fillna("__MISSING__").astype(str)
        X_val[col] = X_val[col].fillna("__MISSING__").astype(str)

    cat_indices = [
        FEATURES.index(c)
        for c in CATEGORICAL_FEATURES
    ]

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_indices,
    )

    val_pool = Pool(
        X_val,
        y_val,
        cat_features=cat_indices,
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2500,
        learning_rate=0.08,
        depth=10,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        border_count=128,
        task_type="GPU",
        devices=str(gpu_id),
        gpu_ram_part=0.92,
        thread_count=threads,
        random_seed=42,
        allow_writing_files=False,
        verbose=100,
        od_type="Iter",
        od_wait=120,
    )

    log(
        f"CATBOOST -> GPU {gpu_id} | "
        f"training {len(train_df):,} rows"
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
    )

    probabilities = model.predict_proba(
        val_pool
    )[:, 1]

    metrics = metrics_from_probabilities(
        y_val,
        probabilities,
    )

    model_path = out_dir / "catboost_gpu0.cbm"
    model.save_model(str(model_path))

    meta = {
        "model": "catboost",
        "gpu_id": gpu_id,
        "rows_train": int(len(train_df)),
        "rows_validation": int(len(val_df)),
        "best_iteration": int(
            model.get_best_iteration()
        ),
        "metrics": metrics,
        "runtime_seconds": time.time() - start,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    save_json(
        out_dir / "catboost_gpu0.json",
        meta,
    )

    del train_pool, val_pool, X_train, X_val
    del train_df, val_df, model
    gc.collect()

    log(
        f"CATBOOST -> GPU {gpu_id} DONE | "
        f"AUC={metrics['roc_auc']:.6f} | "
        f"PR-AUC={metrics['pr_auc']:.6f} | "
        f"time={meta['runtime_seconds']/60:.2f} min"
    )

    return meta


# ---------------------------------------------------------------------
# Model 2 - XGBoost
# ---------------------------------------------------------------------

def train_xgboost(
    gpu_id: int,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    threads: int,
    max_train_rows: int | None,
    max_val_rows: int | None,
) -> Dict:

    start = time.time()

    log(
        f"XGBOOST -> GPU {gpu_id} | "
        f"loading data"
    )

    import xgboost as xgb

    train_df = load_raw_split(
        train_path,
        max_rows=max_train_rows,
    )

    val_df = load_raw_split(
        val_path,
        max_rows=max_val_rows,
    )

    X_train, X_val, y_train, feature_names = make_sklearn_matrix(
        train_df,
        val_df,
    )
    y_val = val_df[TARGET].astype(np.int8).to_numpy()

    log(
        f"XGBOOST -> GPU {gpu_id} | "
        f"matrix {X_train.shape}"
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=2500,
        learning_rate=0.06,
        max_depth=10,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=4.0,
        gamma=0.0,
        tree_method="hist",
        device=f"cuda:{gpu_id}",
        max_bin=256,
        n_jobs=threads,
        random_state=42,
        early_stopping_rounds=120,
    )

    log(
        f"XGBOOST -> GPU {gpu_id} | "
        f"training {len(train_df):,} rows"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_val,
                y_val,
            )
        ],
        verbose=100,
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    metrics = metrics_from_probabilities(
        y_val,
        probabilities,
    )

    model_path = out_dir / "xgboost_gpu1.json"
    model.save_model(str(model_path))

    joblib.dump(
        feature_names,
        out_dir / "xgboost_gpu1_features.joblib",
        compress=3,
    )

    meta = {
        "model": "xgboost",
        "gpu_id": gpu_id,
        "rows_train": int(len(train_df)),
        "rows_validation": int(len(val_df)),
        "best_iteration": int(
            model.best_iteration
            if model.best_iteration is not None
            else model.n_estimators - 1
        ),
        "metrics": metrics,
        "runtime_seconds": time.time() - start,
        "feature_count": len(feature_names),
        "features": feature_names,
    }

    save_json(
        out_dir / "xgboost_gpu1_metadata.json",
        meta,
    )

    del X_train, X_val, y_train, y_val
    del train_df, val_df, model
    gc.collect()

    log(
        f"XGBOOST -> GPU {gpu_id} DONE | "
        f"AUC={metrics['roc_auc']:.6f} | "
        f"PR-AUC={metrics['pr_auc']:.6f} | "
        f"time={meta['runtime_seconds']/60:.2f} min"
    )

    return meta


# ---------------------------------------------------------------------
# Model 3 - LightGBM
# ---------------------------------------------------------------------

def train_lightgbm(
    gpu_id: int,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    threads: int,
    max_train_rows: int | None,
    max_val_rows: int | None,
    allow_cpu_fallback: bool,
) -> Dict:

    start = time.time()

    log(
        f"LIGHTGBM -> GPU {gpu_id} | "
        f"loading data"
    )

    import lightgbm as lgb

    train_df = load_raw_split(
        train_path,
        max_rows=max_train_rows,
    )

    val_df = load_raw_split(
        val_path,
        max_rows=max_val_rows,
    )

    X_train, X_val, y_train, feature_names = make_sklearn_matrix(
        train_df,
        val_df,
    )
    y_val = val_df[TARGET].astype(np.int8).to_numpy()

    log(
        f"LIGHTGBM -> GPU {gpu_id} | "
        f"matrix {X_train.shape}"
    )

    base_params = dict(
        objective="binary",
        metric="auc",
        n_estimators=2500,
        learning_rate=0.055,
        num_leaves=255,
        max_depth=-1,
        min_child_samples=1000,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=4.0,
        max_bin=255,
        random_state=42,
        n_jobs=threads,
        verbosity=-1,
    )

    gpu_mode = "cuda"

    model = None
    last_error = None

    # First try CUDA backend.
    try:
        log(
            f"LIGHTGBM -> GPU {gpu_id} | "
            f"trying device_type='cuda'"
        )

        model = lgb.LGBMClassifier(
            **base_params,
            device_type="cuda",
            gpu_device_id=gpu_id,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_val,
                    y_val,
                )
            ],
            eval_names=["valid_0"],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(
                    120,
                    verbose=True,
                ),
                lgb.log_evaluation(100),
            ],
        )

    except Exception as exc:
        last_error = repr(exc)
        model = None

        log(
            "LightGBM CUDA backend failed. "
            f"Reason: {exc}"
        )

    # Some Windows builds expose OpenCL 'gpu' instead.
    if model is None:
        try:
            log(
                f"LIGHTGBM -> GPU {gpu_id} | "
                f"trying device_type='gpu'"
            )

            gpu_mode = "gpu"

            model = lgb.LGBMClassifier(
                **base_params,
                device_type="gpu",
                gpu_device_id=gpu_id,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (
                        X_val,
                        y_val,
                    )
                ],
                eval_names=["valid_0"],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(
                        120,
                        verbose=True,
                    ),
                    lgb.log_evaluation(100),
                ],
            )

        except Exception as exc:
            last_error = repr(exc)
            model = None

            log(
                "LightGBM GPU backend failed. "
                f"Reason: {exc}"
            )

    # Optional emergency CPU fallback.
    if model is None:
        if not allow_cpu_fallback:
            raise RuntimeError(
                "LightGBM could not initialize GPU training.\n"
                f"Last error: {last_error}\n"
                "Install a CUDA/OpenCL-enabled LightGBM build, "
                "or rerun with --allow-cpu-fallback."
            )

        log(
            "LIGHTGBM -> falling back to CPU. "
            "This will leave GPU 2 mostly unused."
        )

        gpu_mode = "cpu"

        model = lgb.LGBMClassifier(
            **base_params,
            device_type="cpu",
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_val,
                    y_val,
                )
            ],
            eval_names=["valid_0"],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(
                    120,
                    verbose=True,
                ),
                lgb.log_evaluation(100),
            ],
        )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    metrics = metrics_from_probabilities(
        y_val,
        probabilities,
    )

    model_path = out_dir / "lightgbm_gpu2.txt"
    model.booster_.save_model(
        str(model_path)
    )

    joblib.dump(
        feature_names,
        out_dir / "lightgbm_gpu2_features.joblib",
        compress=3,
    )

    meta = {
        "model": "lightgbm",
        "gpu_id": gpu_id,
        "device_mode": gpu_mode,
        "rows_train": int(len(train_df)),
        "rows_validation": int(len(val_df)),
        "best_iteration": int(
            model.best_iteration_
            if model.best_iteration_ is not None
            else model.n_estimators - 1
        ),
        "metrics": metrics,
        "runtime_seconds": time.time() - start,
        "feature_count": len(feature_names),
        "features": feature_names,
    }

    save_json(
        out_dir / "lightgbm_gpu2_metadata.json",
        meta,
    )

    del X_train, X_val, y_train, y_val
    del train_df, val_df, model
    gc.collect()

    log(
        f"LIGHTGBM -> GPU {gpu_id} DONE | "
        f"mode={gpu_mode} | "
        f"AUC={metrics['roc_auc']:.6f} | "
        f"PR-AUC={metrics['pr_auc']:.6f} | "
        f"time={meta['runtime_seconds']/60:.2f} min"
    )

    return meta


# ---------------------------------------------------------------------
# Worker wrapper
# ---------------------------------------------------------------------

def worker(
    model_name: str,
    gpu_id: int,
    train_path: str,
    val_path: str,
    out_dir: str,
    threads: int,
    max_train_rows: int | None,
    max_val_rows: int | None,
    allow_cpu_fallback: bool,
    result_queue,
) -> None:

    # Do NOT mask CUDA_VISIBLE_DEVICES here.
    #
    # We pass the physical GPU ordinal explicitly to each library:
    #   CatBoost -> devices="0"/"1"/"2"
    #   XGBoost  -> device="cuda:0"/"cuda:1"/"cuda:2"
    #   LightGBM -> gpu_device_id=0/1/2
    #
    # Masking CUDA_VISIBLE_DEVICES would renumber the visible device list
    # and could accidentally turn GPU 1/2 into an invalid ordinal.

    # Avoid CPU oversubscription. Each model owns a separate GPU and a
    # controlled number of CPU threads for input preprocessing.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    try:
        train = Path(train_path)
        val = Path(val_path)
        out = Path(out_dir)

        if model_name == "catboost":
            # CatBoost sees this process's visible GPU as device 0 in many
            # setups; to avoid ambiguity, use device 0 after masking.
            #
            # However, if visibility is honored differently by the installed
            # CUDA runtime, the original ordinal is still accepted by many
            # CatBoost installations. We use the original explicit ordinal.
            meta = train_catboost(
                gpu_id=gpu_id,
                train_path=train,
                val_path=val,
                out_dir=out,
                threads=threads,
                max_train_rows=max_train_rows,
                max_val_rows=max_val_rows,
            )

        elif model_name == "xgboost":
            meta = train_xgboost(
                gpu_id=gpu_id,
                train_path=train,
                val_path=val,
                out_dir=out,
                threads=threads,
                max_train_rows=max_train_rows,
                max_val_rows=max_val_rows,
            )

        elif model_name == "lightgbm":
            meta = train_lightgbm(
                gpu_id=gpu_id,
                train_path=train,
                val_path=val,
                out_dir=out,
                threads=threads,
                max_train_rows=max_train_rows,
                max_val_rows=max_val_rows,
                allow_cpu_fallback=allow_cpu_fallback,
            )

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        result_queue.put(
            {
                "status": "ok",
                "model": model_name,
                "gpu_id": gpu_id,
                "meta": meta,
            }
        )

    except Exception:
        tb = traceback.format_exc()

        result_queue.put(
            {
                "status": "error",
                "model": model_name,
                "gpu_id": gpu_id,
                "error": tb,
            }
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CatBoost + XGBoost + LightGBM "
            "concurrently on GPUs 0/1/2."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Directory containing recovery_train.parquet "
            "and recovery_validation.parquet."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="recovery_models_parallel",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=12,
        help=(
            "CPU threads per model process. "
            "With 3 concurrent jobs, 12 = about 36 CPU threads."
        ),
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help=(
            "Optional development cap. "
            "Default = all 16M training rows."
        ),
    )

    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help=(
            "Optional development cap. "
            "Default = all validation rows."
        ),
    )

    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help=(
            "Allow LightGBM to fall back to CPU if the installed "
            "LightGBM build has no working GPU backend."
        ),
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    safe_mkdir(out_dir)

    train_path = data_dir / "recovery_train.parquet"
    val_path = data_dir / "recovery_validation.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing: {train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Missing: {val_path}"
        )

    # ---------------------------------------------------------------
    # Save run configuration
    # ---------------------------------------------------------------

    config = {
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "threads_per_model": args.threads,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "gpu_assignment": {
            "catboost": 0,
            "xgboost": 1,
            "lightgbm": 2,
        },
        "gpu_3_reserved": True,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
    }

    save_json(
        out_dir / "training_config.json",
        config,
    )

    print()
    print("=" * 88)
    print("RecoveryOS PARALLEL GPU TRAINING")
    print("=" * 88)
    print()
    print("GPU 0 -> CatBoost")
    print("GPU 1 -> XGBoost")
    print("GPU 2 -> LightGBM")
    print("GPU 3 -> RESERVED")
    print()
    print(f"Train: {train_path}")
    print(f"Val  : {val_path}")
    print(f"CPU threads/model: {args.threads}")
    print(
        "Rows/model: "
        + (
            f"{args.max_train_rows:,}"
            if args.max_train_rows
            else "ALL"
        )
    )
    print()

    # ---------------------------------------------------------------
    # Windows-safe multiprocessing.
    # ---------------------------------------------------------------

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    jobs = [
        (
            "catboost",
            0,
        ),
        (
            "xgboost",
            1,
        ),
        (
            "lightgbm",
            2,
        ),
    ]

    processes = []

    global_start = time.time()

    for model_name, gpu_id in jobs:

        p = ctx.Process(
            target=worker,
            args=(
                model_name,
                gpu_id,
                str(train_path),
                str(val_path),
                str(out_dir),
                args.threads,
                args.max_train_rows,
                args.max_val_rows,
                args.allow_cpu_fallback,
                result_queue,
            ),
            name=f"{model_name}-gpu{gpu_id}",
        )

        p.start()
        processes.append(p)

        print(
            f"Started {model_name} on GPU {gpu_id} "
            f"(PID {p.pid})"
        )

    # ---------------------------------------------------------------
    # Wait for all workers.
    # ---------------------------------------------------------------

    results = []

    for p in processes:
        p.join()

    for _ in processes:
        try:
            results.append(
                result_queue.get(
                    timeout=10
                )
            )
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------------

    runtime = time.time() - global_start

    errors = [
        r for r in results
        if r.get("status") != "ok"
    ]

    successes = [
        r for r in results
        if r.get("status") == "ok"
    ]

    summary = {
        "total_runtime_seconds": runtime,
        "successful_models": successes,
        "failed_models": errors,
    }

    save_json(
        out_dir / "parallel_training_summary.json",
        summary,
    )

    print()
    print("=" * 88)
    print("PARALLEL TRAINING COMPLETE")
    print("=" * 88)
    print(
        f"Wall time: {runtime / 60:.2f} minutes"
    )
    print()

    if successes:
        print(
            f"{'MODEL':<12}"
            f"{'GPU':<6}"
            f"{'AUC':<12}"
            f"{'PR-AUC':<12}"
            f"{'Brier':<12}"
            f"{'TIME(min)':<12}"
        )

        print("-" * 68)

        for item in sorted(
            successes,
            key=lambda x: x["model"],
        ):
            meta = item["meta"]
            m = meta["metrics"]

            print(
                f"{item['model']:<12}"
                f"{item['gpu_id']:<6}"
                f"{m['roc_auc']:<12.6f}"
                f"{m['pr_auc']:<12.6f}"
                f"{m['brier']:<12.6f}"
                f"{meta['runtime_seconds']/60:<12.2f}"
            )

    if errors:
        print()
        print("FAILED MODELS")
        print("-" * 88)

        for item in errors:
            print(
                f"\n[{item['model']} / GPU {item['gpu_id']}]\n"
                f"{item.get('error', 'unknown error')}"
            )

    # ---------------------------------------------------------------
    # Rank models by PR-AUC first, then Brier.
    # ---------------------------------------------------------------

    if successes:

        ranked = sorted(
            successes,
            key=lambda x: (
                -x["meta"]["metrics"]["pr_auc"],
                x["meta"]["metrics"]["brier"],
            ),
        )

        winner = ranked[0]

        save_json(
            out_dir / "recommended_model.json",
            {
                "winner": winner,
                "ranking": ranked,
                "selection_rule": (
                    "highest validation PR-AUC; "
                    "Brier score used as secondary criterion"
                ),
            },
        )

        print()
        print(
            "Current probability-model winner: "
            f"{winner['model'].upper()} "
            f"(GPU {winner['gpu_id']})"
        )

    print()
    print(
        "NEXT STEP: run policy generation/evaluation using the "
        "three models and the preserved V2 counterfactual columns."
    )

    return 0 if len(errors) == 0 else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
