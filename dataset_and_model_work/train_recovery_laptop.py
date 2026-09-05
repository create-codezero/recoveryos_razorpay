

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "observed_recovered"

ACTION_ORDER = [
    "retry_30m",
    "retry_evening",
    "payment_link",
    "whatsapp_reminder",
    "alternate_method",
    "stop",
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


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train laptop-friendly CatBoost recovery model."
    )

    parser.add_argument(
        "--train",
        default="recovery_prepared/recovery_train.parquet",
        help="Training parquet file",
    )

    parser.add_argument(
        "--validation",
        default="recovery_prepared/recovery_validation.parquet",
        help="Validation parquet file",
    )

    parser.add_argument(
        "--output-dir",
        default="recovery_models",
        help="Directory for model artifacts",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help=(
            "Maximum training rows. "
            "Use 2000000 or 4000000 for development. "
            "Default: all rows."
        ),
    )

    parser.add_argument(
        "--val-rows",
        type=int,
        default=None,
        help="Maximum validation rows. Default: all rows.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1800,
        help="Maximum CatBoost iterations",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="CatBoost tree depth",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.07,
        help="CatBoost learning rate",
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=100,
        help="Early stopping rounds",
    )

    parser.add_argument(
        "--thread-count",
        type=int,
        default=None,
        help="CPU threads. Default: automatic conservative value.",
    )

    parser.add_argument(
        "--task-type",
        choices=["GPU", "CPU"],
        default="GPU",
        help="CatBoost device",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID",
    )

    parser.add_argument(
        "--gpu-ram-part",
        type=float,
        default=0.75,
        help="Fraction of GPU memory CatBoost may use",
    )

    parser.add_argument(
        "--development",
        action="store_true",
        help=(
            "Development mode. If --rows is omitted, uses 2M train "
            "and 250k validation rows."
        ),
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle sampled rows.",
    )

    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_data(
    path: str,
    max_rows: int | None,
    seed: int,
    shuffle: bool,
) -> pd.DataFrame:

    log(f"Loading: {path}")

    columns = FEATURES + [TARGET]

    start = time.time()

    df = pd.read_parquet(
        path,
        columns=columns,
    )

    elapsed = time.time() - start

    log(
        f"Loaded {len(df):,} rows "
        f"in {elapsed:.1f}s"
    )

    if max_rows is not None and len(df) > max_rows:

        log(
            f"Sampling {max_rows:,} rows "
            f"from {len(df):,} rows"
        )

        if shuffle:
            df = df.sample(
                n=max_rows,
                random_state=seed,
            )
        else:
            df = df.iloc[:max_rows].copy()

    else:
        df = df.copy()

    # --------------------------------------------------------
    # Ensure categorical columns are strings.
    # --------------------------------------------------------

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("__MISSING__").astype(str)

    # --------------------------------------------------------
    # Numeric columns.
    # --------------------------------------------------------

    numeric_features = [
        c for c in FEATURES
        if c not in CATEGORICAL_FEATURES
    ]

    for col in numeric_features:

        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
                downcast="float",
            )

        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
                downcast="integer",
            )

    # --------------------------------------------------------
    # Target.
    # --------------------------------------------------------

    df[TARGET] = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )
        .fillna(0)
        .astype(np.int8)
    )

    # --------------------------------------------------------
    # Validate actions.
    # --------------------------------------------------------

    actions = set(
        df["observed_intervention"].unique()
    )

    unknown_actions = actions.difference(
        ACTION_ORDER
    )

    if unknown_actions:
        raise ValueError(
            f"Unknown intervention(s): "
            f"{sorted(unknown_actions)}"
        )

    # --------------------------------------------------------
    # Validate columns.
    # --------------------------------------------------------

    missing = [
        c for c in FEATURES + [TARGET]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    log(
        f"Final dataframe: "
        f"{len(df):,} rows × {len(FEATURES)} features"
    )

    return df


# ============================================================
# DATA QUALITY
# ============================================================

def print_dataset_stats(
    name: str,
    df: pd.DataFrame,
) -> None:

    log(f"--- {name} statistics ---")

    positive_rate = float(
        df[TARGET].mean()
    )

    log(
        f"Recovery rate: "
        f"{positive_rate * 100:.4f}%"
    )

    log(
        f"Memory usage: "
        f"{df.memory_usage(deep=True).sum() / 1024**3:.2f} GB"
    )

    log("Interventions:")

    counts = (
        df["observed_intervention"]
        .value_counts()
        .reindex(ACTION_ORDER)
        .fillna(0)
        .astype(int)
    )

    for action, count in counts.items():

        rate = (
            df.loc[
                df["observed_intervention"] == action,
                TARGET,
            ].mean()
        )

        log(
            f"  {action:<20} "
            f"{count:>10,} "
            f"recovery={rate * 100:.3f}%"
        )


# ============================================================
# GPU CHECK
# ============================================================

def print_gpu_environment() -> None:

    log("Checking CatBoost GPU environment...")

    try:
        import catboost

        log(
            f"CatBoost version: "
            f"{catboost.__version__}"
        )

    except Exception as exc:
        log(f"Could not determine CatBoost version: {exc}")

    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            log("NVIDIA GPU:")
            print(result.stdout.strip(), flush=True)
        else:
            log(
                "nvidia-smi unavailable. "
                "CatBoost GPU training may fail."
            )

    except Exception as exc:
        log(
            f"GPU check unavailable: {exc}"
        )


# ============================================================
# TRAINING
# ============================================================

def train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args,
):

    log("=" * 70)
    log("TRAINING CATBOOST RECOVERY MODEL")
    log("=" * 70)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_val = val_df[FEATURES]
    y_val = val_df[TARGET]

    cat_indices = [
        FEATURES.index(col)
        for col in CATEGORICAL_FEATURES
    ]

    if args.thread_count is None:

        # Conservative choice for a laptop.
        cpu_count = os.cpu_count() or 8

        thread_count = max(
            2,
            min(8, cpu_count // 2),
        )

    else:
        thread_count = args.thread_count

    log(
        f"CatBoost device: {args.task_type}"
    )

    log(
        f"GPU ID: {args.gpu_id}"
    )

    log(
        f"Threads: {thread_count}"
    )

    log(
        f"Iterations: {args.iterations}"
    )

    log(
        f"Depth: {args.depth}"
    )

    log(
        f"Learning rate: {args.learning_rate}"
    )

    log(
        f"Early stopping: "
        f"{args.early_stopping}"
    )

    # --------------------------------------------------------
    # CatBoost parameters.
    #
    # Compared with the old server configuration:
    #
    #   depth 10 -> 8
    #   iterations 2500 -> 1800
    #   GPU RAM 92% -> 75%
    #
    # This is intentional for the 4 GB RTX 3050.
    # --------------------------------------------------------

    model_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": 6,
        "random_strength": 0.5,
        "border_count": 128,

        "random_seed": args.seed,

        "thread_count": thread_count,

        "cat_features": cat_indices,

        "verbose": 100,

        "allow_writing_files": False,

        "od_type": "Iter",
        "od_wait": args.early_stopping,
    }

    if args.task_type == "GPU":

        model_params.update({
            "task_type": "GPU",
            "devices": str(args.gpu_id),
            "gpu_ram_part": args.gpu_ram_part,
        })

    else:

        model_params.update({
            "task_type": "CPU",
        })

    model = CatBoostClassifier(
        **model_params
    )

    log("Starting CatBoost training...")

    start = time.time()

    model.fit(
        X_train,
        y_train,

        eval_set=(
            X_val,
            y_val,
        ),

        verbose=100,
    )

    elapsed = time.time() - start

    log(
        f"Training finished in "
        f"{elapsed / 60:.2f} minutes"
    )

    return model, elapsed


# ============================================================
# VALIDATION METRICS
# ============================================================

def calculate_validation_metrics(
    model,
    val_df: pd.DataFrame,
) -> dict:

    log("=" * 70)
    log("VALIDATION")
    log("=" * 70)

    X_val = val_df[FEATURES]
    y_val = val_df[TARGET].to_numpy()

    start = time.time()

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    elapsed = time.time() - start

    log(
        f"Prediction time: "
        f"{elapsed:.1f}s"
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int8)

    # --------------------------------------------------------
    # Basic classification metrics.
    # --------------------------------------------------------

    try:

        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            brier_score_loss,
            log_loss,
        )

        auc = roc_auc_score(
            y_val,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_val,
            probabilities,
        )

        brier = brier_score_loss(
            y_val,
            probabilities,
        )

        loss = log_loss(
            y_val,
            probabilities,
        )

    except Exception as exc:

        log(
            f"Could not calculate sklearn "
            f"metrics: {exc}"
        )

        auc = None
        pr_auc = None
        brier = None
        loss = None

    accuracy = float(
        (predictions == y_val).mean()
    )

    result = {
        "validation_rows": int(len(val_df)),
        "recovery_rate": float(y_val.mean()),
        "auc": None if auc is None else float(auc),
        "pr_auc": None if pr_auc is None else float(pr_auc),
        "brier": None if brier is None else float(brier),
        "log_loss": None if loss is None else float(loss),
        "accuracy_at_0_5": accuracy,
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
    }

    log(
        f"AUC:       "
        f"{result['auc']:.6f}"
        if result["auc"] is not None
        else "AUC: unavailable"
    )

    log(
        f"PR-AUC:    "
        f"{result['pr_auc']:.6f}"
        if result["pr_auc"] is not None
        else "PR-AUC: unavailable"
    )

    log(
        f"Brier:     "
        f"{result['brier']:.6f}"
        if result["brier"] is not None
        else "Brier: unavailable"
    )

    log(
        f"Log loss:  "
        f"{result['log_loss']:.6f}"
        if result["log_loss"] is not None
        else "Log loss: unavailable"
    )

    log(
        f"Accuracy:  "
        f"{result['accuracy_at_0_5']:.6f}"
    )

    return result


# ============================================================
# MODEL ARTIFACTS
# ============================================================

def save_artifacts(
    model,
    output_dir: Path,
    args,
    validation_metrics: dict,
    train_rows: int,
    val_rows: int,
    training_seconds: float,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir /
        "catboost_recovery_laptop.cbm"
    )

    metadata_path = (
        output_dir /
        "catboost_recovery_laptop_metadata.json"
    )

    feature_path = (
        output_dir /
        "catboost_recovery_features.json"
    )

    log(
        f"Saving model: {model_path}"
    )

    model.save_model(
        str(model_path)
    )

    metadata = {
        "model_type": "CatBoostClassifier",
        "model_role": (
            "action_conditioned_recovery_model"
        ),

        "target": TARGET,

        "features": FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "actions": ACTION_ORDER,

        "train_rows": int(train_rows),

        "validation_rows": int(val_rows),

        "training_seconds":
            float(training_seconds),

        "training_minutes":
            float(training_seconds / 60),

        "seed": int(args.seed),

        "iterations": int(args.iterations),

        "depth": int(args.depth),

        "learning_rate":
            float(args.learning_rate),

        "early_stopping":
            int(args.early_stopping),

        "task_type":
            args.task_type,

        "gpu_id":
            int(args.gpu_id),

        "gpu_ram_part":
            float(args.gpu_ram_part),

        "validation_metrics":
            validation_metrics,

        "notes": [
            (
                "observed_intervention is an input feature."
            ),
            (
                "The model estimates "
                "P(observed_recovered | context, action)."
            ),
            (
                "Policy evaluation should score "
                "all candidate actions and choose "
                "the action with highest expected "
                "recovered revenue."
            ),
            (
                "Do not select the final policy using "
                "PR-AUC alone."
            ),
        ],
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    with open(
        feature_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "features": FEATURES,
                "categorical_features":
                    CATEGORICAL_FEATURES,
                "actions": ACTION_ORDER,
                "target": TARGET,
            },
            f,
            indent=2,
        )

    log(
        f"Saved metadata: {metadata_path}"
    )

    log(
        f"Saved feature manifest: "
        f"{feature_path}"
    )

    return model_path, metadata_path


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Development defaults.
    # --------------------------------------------------------

    if args.development:

        if args.rows is None:
            args.rows = 2_000_000

        if args.val_rows is None:
            args.val_rows = 250_000

        log(
            "DEVELOPMENT MODE ENABLED"
        )

        log(
            f"Training rows: "
            f"{args.rows:,}"
        )

        log(
            f"Validation rows: "
            f"{args.val_rows:,}"
        )

    # --------------------------------------------------------
    # Output directory.
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Environment.
    # --------------------------------------------------------

    log("=" * 70)
    log("RECOVERYOS LAPTOP TRAINING")
    log("=" * 70)

    log(
        f"Python: {sys.version.split()[0]}"
    )

    log(
        f"Train file: {args.train}"
    )

    log(
        f"Validation file: {args.validation}"
    )

    log(
        f"Output directory: {output_dir}"
    )

    print_gpu_environment()

    # --------------------------------------------------------
    # Load training data.
    # --------------------------------------------------------

    train_df = load_data(
        path=args.train,
        max_rows=args.rows,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    print_dataset_stats(
        "TRAIN",
        train_df,
    )

    # --------------------------------------------------------
    # Load validation data.
    # --------------------------------------------------------

    val_df = load_data(
        path=args.validation,
        max_rows=args.val_rows,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    print_dataset_stats(
        "VALIDATION",
        val_df,
    )

    # --------------------------------------------------------
    # Make sure every action is represented in training.
    # --------------------------------------------------------

    train_actions = set(
        train_df[
            "observed_intervention"
        ].unique()
    )

    missing_actions = set(
        ACTION_ORDER
    ).difference(train_actions)

    if missing_actions:

        raise RuntimeError(
            "Training data does not contain "
            f"these actions: {sorted(missing_actions)}"
        )

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------

    model, training_seconds = train_catboost(
        train_df=train_df,
        val_df=val_df,
        args=args,
    )

    # --------------------------------------------------------
    # Validation.
    # --------------------------------------------------------

    validation_metrics = (
        calculate_validation_metrics(
            model=model,
            val_df=val_df,
        )
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    model_path, metadata_path = (
        save_artifacts(
            model=model,
            output_dir=output_dir,
            args=args,
            validation_metrics=validation_metrics,
            train_rows=len(train_df),
            val_rows=len(val_df),
            training_seconds=training_seconds,
        )
    )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    log("=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)

    log(
        f"Model: {model_path}"
    )

    log(
        f"Metadata: {metadata_path}"
    )

    if validation_metrics["auc"] is not None:

        log(
            f"Validation AUC: "
            f"{validation_metrics['auc']:.6f}"
        )

    if validation_metrics["pr_auc"] is not None:

        log(
            f"Validation PR-AUC: "
            f"{validation_metrics['pr_auc']:.6f}"
        )

    log("")
    log(
        "NEXT STEP:"
    )

    log(
        "Run policy evaluation on the HELD-OUT TEST SET."
    )

    log(
        "The evaluator should score all six interventions "
        "for every test transaction."
    )

    log(
        "Do NOT choose the winning model by PR-AUC alone."
    )

    # --------------------------------------------------------
    # Explicit cleanup.
    # --------------------------------------------------------

    del train_df
    del val_df
    del model

    gc.collect()


if __name__ == "__main__":
    main()
