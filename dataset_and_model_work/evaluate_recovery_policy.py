import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ACTIONS = [
    "retry_30m",
    "retry_evening",
    "payment_link",
    "whatsapp_reminder",
    "alternate_method",
    "stop",
]

PROB_COLUMNS = {
    "retry_30m": "p_retry_30m",
    "retry_evening": "p_retry_evening",
    "payment_link": "p_payment_link",
    "whatsapp_reminder": "p_whatsapp_reminder",
    "alternate_method": "p_alternate_method",
    "stop": "p_stop",
}

DEFAULT_COSTS = {
    "retry_30m": 0.0,
    "retry_evening": 0.0,
    "payment_link": 0.0,
    "whatsapp_reminder": 0.0,
    "alternate_method": 0.0,
    "stop": 0.0,
}

def log(msg):
    print(f"[RecoveryOS] {msg}", flush=True)


def money(x):
    """Format INR value."""
    if pd.isna(x):
        return "NaN"

    x = float(x)

    if abs(x) >= 1e9:
        return f"₹{x / 1e9:.3f}B"
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.3f}Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.3f}L"

    return f"₹{x:,.2f}"


def pct(x):
    return f"{100.0 * float(x):.3f}%"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path, columns=None):
    log(f"Loading: {path}")

    start = time.time()

    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path, columns=columns)
    elif str(path).endswith(".csv"):
        df = pd.read_csv(path, usecols=columns)
    else:
        raise ValueError(
            f"Unsupported file format: {path}. "
            "Use .parquet or .csv"
        )

    log(
        f"Loaded {len(df):,} rows "
        f"in {time.time() - start:.2f}s"
    )

    return df


# ============================================================
# GROUND TRUTH
# ============================================================

def validate_ground_truth(df):
    missing = []

    for action, col in PROB_COLUMNS.items():
        if col not in df.columns:
            missing.append(col)

    if "amount" not in df.columns:
        missing.append("amount")

    if "optimal_intervention" not in df.columns:
        missing.append("optimal_intervention")

    if "oracle_recovery_probability" not in df.columns:
        missing.append("oracle_recovery_probability")

    if "oracle_expected_revenue" not in df.columns:
        missing.append("oracle_expected_revenue")

    if missing:
        raise ValueError(
            "Dataset is missing required evaluation columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )


# ============================================================
# GROUND-TRUTH POLICY
# ============================================================

def build_oracle_policy(df):
    """
    Build the synthetic oracle policy from p_* columns.

    This is NOT used as a model feature.

    Oracle action:
        argmax_a P(recovery | context, a)
    """

    prob_matrix = np.column_stack(
        [
            df[PROB_COLUMNS[action]].to_numpy(dtype=np.float64)
            for action in ACTIONS
        ]
    )

    oracle_indices = np.argmax(prob_matrix, axis=1)

    oracle_actions = np.asarray(ACTIONS, dtype=object)[oracle_indices]

    oracle_probability = np.max(prob_matrix, axis=1)

    return oracle_actions, oracle_probability


# ============================================================
# HISTORICAL POLICY
# ============================================================

def historical_policy(df):
    """
    Observed intervention in the synthetic dataset.
    """

    if "observed_intervention" not in df.columns:
        raise ValueError(
            "observed_intervention is required "
            "for historical-policy evaluation."
        )

    return df["observed_intervention"].astype(str).to_numpy()


# ============================================================
# SIMPLE BASELINES
# ============================================================

def constant_policy(df, action):
    return np.full(len(df), action, dtype=object)


def simple_rule_policy(df):
    """
    A deliberately simple business-rule baseline.

    This is NOT intended to be optimal.

    Rules:
        bank_timeout / network_error -> retry_evening
        expired_card -> payment_link
        insufficient_funds -> whatsapp_reminder
        authentication_failed -> alternate_method
        limit_exceeded -> alternate_method
        user_cancelled -> whatsapp_reminder

    Unknown -> retry_evening
    """

    result = np.full(
        len(df),
        "retry_evening",
        dtype=object,
    )

    reason = df["failure_reason"].astype(str).to_numpy()

    result[
        np.isin(
            reason,
            ["bank_timeout", "network_error"],
        )
    ] = "retry_evening"

    result[
        reason == "expired_card"
    ] = "payment_link"

    result[
        reason == "insufficient_funds"
    ] = "whatsapp_reminder"

    result[
        reason == "authentication_failed"
    ] = "alternate_method"

    result[
        reason == "limit_exceeded"
    ] = "alternate_method"

    result[
        reason == "user_cancelled"
    ] = "whatsapp_reminder"

    return result


# ============================================================
# POLICY SCORING
# ============================================================

def score_policy(
    df,
    policy_actions,
    policy_name,
    costs,
):
    """
    Evaluate a policy using synthetic counterfactual probabilities.

    For each transaction:

        expected_revenue =
            selected_action_probability * amount
            - intervention_cost
    """

    amounts = df["amount"].to_numpy(dtype=np.float64)

    selected_probabilities = np.zeros(len(df), dtype=np.float64)
    selected_costs = np.zeros(len(df), dtype=np.float64)

    valid = np.zeros(len(df), dtype=bool)

    for action in ACTIONS:

        mask = policy_actions == action

        if not np.any(mask):
            continue

        prob_col = PROB_COLUMNS[action]

        selected_probabilities[mask] = (
            df.loc[mask, prob_col]
            .to_numpy(dtype=np.float64)
        )

        selected_costs[mask] = costs.get(
            action,
            0.0,
        )

        valid[mask] = True

    expected_revenue = (
        selected_probabilities * amounts
        - selected_costs
    )

    result = {
        "policy": policy_name,
        "rows": len(df),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "expected_recovery_probability": float(
            selected_probabilities.mean()
        ),
        "expected_revenue": float(
            expected_revenue.sum()
        ),
        "expected_revenue_per_transaction": float(
            expected_revenue.mean()
        ),
        "selected_gross_revenue": float(
            (selected_probabilities * amounts).sum()
        ),
        "intervention_cost": float(
            selected_costs.sum()
        ),
        "action_agreement_with_oracle": np.nan,
        "revenue_efficiency_vs_oracle": np.nan,
        "revenue_regret_vs_oracle": np.nan,
    }

    return result, selected_probabilities, expected_revenue


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

def action_distribution(actions):
    counts = pd.Series(actions).value_counts()

    total = len(actions)

    rows = []

    for action in ACTIONS:

        count = int(counts.get(action, 0))

        rows.append(
            {
                "action": action,
                "count": count,
                "percentage": (
                    count / total
                    if total > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MODEL PREDICTION
# ============================================================

def load_model(model_path):
    """
    Automatically detect and load:

        CatBoost
        XGBoost
        LightGBM
    """

    path = str(model_path)

    suffix = Path(path).suffix.lower()

    log(f"Loading model: {path}")

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    if suffix in [".cbm", ".catboost"]:

        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise RuntimeError(
                "catboost is not installed."
            )

        model = CatBoostClassifier()
        model.load_model(path)

        return model, "catboost"

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if suffix in [".json", ".ubj", ".model"]:

        try:
            import xgboost as xgb
        except ImportError:
            raise RuntimeError(
                "xgboost is not installed."
            )

        model = xgb.XGBClassifier()
        model.load_model(path)

        return model, "xgboost"

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    if suffix in [".txt", ".lgb"]:

        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError(
                "lightgbm is not installed."
            )

        booster = lgb.Booster(
            model_file=path
        )

        return booster, "lightgbm"

    raise ValueError(
        f"Could not infer model type from: {path}\n"
        "Supported examples:\n"
        "  model.cbm\n"
        "  model.json\n"
        "  model.txt"
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

EXCLUDED_COLUMNS = {
    # IDs
    "transaction_id",
    "customer_id",
    "merchant_id",

    # Timestamp
    "timestamp",

    # Observed outcomes
    "observed_intervention",
    "observed_recovered",
    "observed_recovery_probability",
    "observed_expected_revenue",

    # Counterfactual probabilities
    "p_retry_30m",
    "p_retry_evening",
    "p_payment_link",
    "p_whatsapp_reminder",
    "p_alternate_method",
    "p_stop",

    # Oracle
    "optimal_intervention",
    "oracle_recovery_probability",
    "oracle_expected_revenue",
}


def prepare_features(df):
    """
    Prepare observable transaction features.

    IMPORTANT:
        No outcome / oracle / counterfactual columns
        are passed to the model.
    """

    feature_columns = [
        col
        for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    X = df[feature_columns].copy()

    return X, feature_columns


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def prepare_model_input(X, model_type):
    """
    Prepare model input.

    CatBoost:
        keeps categorical columns as strings.

    XGBoost / LightGBM:
        converts object/category columns to integer codes.

    NOTE:
        This assumes the training script used compatible
        categorical handling.

    If your train_parallel_gpu.py saves an explicit feature
        preprocessing pipeline, use that instead.
    """

    X = X.copy()

    if model_type == "catboost":

        categorical_columns = list(
            X.select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            ).columns
        )

        for col in categorical_columns:
            X[col] = X[col].fillna("__MISSING__").astype(str)

        return X

    # --------------------------------------------------------
    # Tree models requiring numeric matrices
    # --------------------------------------------------------

    for col in X.columns:

        if (
            X[col].dtype == "object"
            or str(X[col].dtype) == "category"
        ):
            X[col] = (
                X[col]
                .fillna("__MISSING__")
                .astype("category")
                .cat.codes
                .astype(np.int32)
            )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    return X


# ============================================================
# MODEL PREDICTION STRATEGY
# ============================================================

def predict_probability(
    model,
    X,
    model_type,
):
    """
    Get P(recovery).

    IMPORTANT:
    Your current training setup needs to be checked here.

    This function assumes each trained model predicts:

        P(observed recovery | context)

    To obtain true action-conditioned probabilities:

        P(recovery | context, action)

    the training dataset must contain an action/intervention
    feature.

    If your current models were trained WITHOUT action as a
    feature, they cannot independently estimate six different
    action probabilities.

    In that case, this function raises an error rather than
    pretending that one probability applies to all actions.
    """

    if model_type == "catboost":

        pred = model.predict_proba(X)[:, 1]

    elif model_type == "xgboost":

        pred = model.predict_proba(X)[:, 1]

    elif model_type == "lightgbm":

        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(X)[:, 1]
        else:
            pred = model.predict(X)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    return np.asarray(
        pred,
        dtype=np.float64,
    )


# ============================================================
# POLICY FROM ACTION-CONDITIONED MODEL
# ============================================================

def build_model_policy(
    model,
    model_type,
    X,
    amounts,
    costs,
):
    """
    Generate action-conditioned policy.

    The model is evaluated once per action by injecting
    the action into the input.

    EXPECTED REQUIREMENT:
        Training data must contain an `intervention`
        feature, or equivalent action column.

    The exact action feature name can be configured using
        --action-column
    """

    raise RuntimeError(
        "\n"
        "Your trained models cannot be safely evaluated as "
        "six-action policies without knowing how the training "
        "script encoded the intervention/action.\n\n"
        "Expected architecture:\n"
        "    context + candidate_action\n"
        "            ↓\n"
        "       recovery model\n"
        "            ↓\n"
        "    P(recovery | context, action)\n"
        "            ↓\n"
        "    amount × probability\n"
        "            ↓\n"
        "        best action\n\n"
        "Please make sure your train_parallel_gpu.py includes "
        "the intervention/action as a model feature."
    )


# ============================================================
# METRICS
# ============================================================

def calculate_policy_metrics(
    df,
    policy_actions,
    oracle_actions,
    oracle_probabilities,
    costs,
):
    amounts = df["amount"].to_numpy(
        dtype=np.float64
    )

    selected_probabilities = np.zeros(
        len(df),
        dtype=np.float64,
    )

    selected_costs = np.zeros(
        len(df),
        dtype=np.float64,
    )

    for action in ACTIONS:

        mask = policy_actions == action

        if not np.any(mask):
            continue

        selected_probabilities[mask] = (
            df.loc[
                mask,
                PROB_COLUMNS[action]
            ]
            .to_numpy(dtype=np.float64)
        )

        selected_costs[mask] = costs.get(
            action,
            0.0,
        )

    policy_revenue = (
        selected_probabilities * amounts
        - selected_costs
    )

    oracle_revenue = (
        oracle_probabilities * amounts
        - np.array(
            [
                costs.get(a, 0.0)
                for a in oracle_actions
            ],
            dtype=np.float64,
        )
    )

    agreement = np.mean(
        policy_actions == oracle_actions
    )

    efficiency = (
        policy_revenue.sum()
        / oracle_revenue.sum()
        if oracle_revenue.sum() != 0
        else np.nan
    )

    regret = (
        oracle_revenue.sum()
        - policy_revenue.sum()
    )

    return {
        "expected_recovery_probability": float(
            selected_probabilities.mean()
        ),
        "expected_revenue": float(
            policy_revenue.sum()
        ),
        "gross_expected_revenue": float(
            (selected_probabilities * amounts).sum()
        ),
        "intervention_cost": float(
            selected_costs.sum()
        ),
        "oracle_expected_revenue": float(
            oracle_revenue.sum()
        ),
        "oracle_recovery_probability": float(
            oracle_probabilities.mean()
        ),
        "oracle_action_agreement": float(
            agreement
        ),
        "revenue_efficiency_vs_oracle": float(
            efficiency
        ),
        "revenue_regret_vs_oracle": float(
            regret
        ),
    }


# ============================================================
# SEGMENT ANALYSIS
# ============================================================

def segment_analysis(
    df,
    policy_actions,
    oracle_actions,
    oracle_probabilities,
    costs,
    column,
):
    if column not in df.columns:
        return pd.DataFrame()

    rows = []

    for value, group_idx in df.groupby(
        column,
        sort=False,
    ).groups.items():

        idx = np.asarray(group_idx)

        group_df = df.iloc[idx]

        metrics = calculate_policy_metrics(
            group_df,
            policy_actions[idx],
            oracle_actions[idx],
            oracle_probabilities[idx],
            costs,
        )

        row = {
            "segment": column,
            "value": value,
            "rows": len(idx),
            **metrics,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_baselines(
    df,
    output_dir,
    costs,
):
    """
    Evaluate policies for which action selection is already
    defined:

        historical
        always_retry_evening
        always_retry_30m
        simple_rules
        oracle
    """

    ensure_dir(output_dir)

    oracle_actions, oracle_probabilities = (
        build_oracle_policy(df)
    )

    policies = {}

    policies["historical"] = historical_policy(df)

    policies["always_retry_evening"] = (
        constant_policy(
            df,
            "retry_evening",
        )
    )

    policies["always_retry_30m"] = (
        constant_policy(
            df,
            "retry_30m",
        )
    )

    policies["simple_rules"] = (
        simple_rule_policy(df)
    )

    policies["oracle"] = oracle_actions

    summary_rows = []

    for name, actions in policies.items():

        log(f"Evaluating policy: {name}")

        metrics = calculate_policy_metrics(
            df,
            actions,
            oracle_actions,
            oracle_probabilities,
            costs,
        )

        metrics["policy"] = name
        metrics["rows"] = len(df)

        summary_rows.append(metrics)

        # Action distribution
        dist = action_distribution(actions)

        dist.to_csv(
            Path(output_dir)
            / f"{name}_action_distribution.csv",
            index=False,
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # Reorder
    preferred = [
        "policy",
        "rows",
        "expected_recovery_probability",
        "expected_revenue",
        "gross_expected_revenue",
        "intervention_cost",
        "oracle_expected_revenue",
        "oracle_recovery_probability",
        "oracle_action_agreement",
        "revenue_efficiency_vs_oracle",
        "revenue_regret_vs_oracle",
    ]

    summary = summary[
        [
            c
            for c in preferred
            if c in summary.columns
        ]
    ]

    summary.to_csv(
        Path(output_dir)
        / "policy_summary.csv",
        index=False,
    )

    return summary, policies, oracle_actions, oracle_probabilities


# ============================================================
# DETAILED POLICY ANALYSIS
# ============================================================

def detailed_analysis(
    df,
    policies,
    oracle_actions,
    oracle_probabilities,
    output_dir,
    costs,
):
    log("Running detailed segment analysis...")

    segment_columns = [
        "failure_reason",
        "customer_value",
        "payment_method",
        "device",
        "merchant_category",
        "location",
    ]

    for policy_name, actions in policies.items():

        if policy_name == "oracle":
            continue

        for column in segment_columns:

            result = segment_analysis(
                df,
                actions,
                oracle_actions,
                oracle_probabilities,
                costs,
                column,
            )

            if len(result) == 0:
                continue

            result.to_csv(
                Path(output_dir)
                / f"{policy_name}_by_{column}.csv",
                index=False,
            )


# ============================================================
# SAVE POLICY OUTPUT
# ============================================================

def save_policy_assignments(
    df,
    policies,
    oracle_actions,
    oracle_probabilities,
    output_dir,
):
    """
    Save transaction-level policy decisions.

    This is useful for later dashboard/demo work.
    """

    columns = []

    for col in [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "amount",
        "failure_reason",
        "customer_value",
    ]:
        if col in df.columns:
            columns.append(col)

    result = df[columns].copy()

    for name, actions in policies.items():

        result[
            f"{name}_action"
        ] = actions

    result[
        "oracle_probability"
    ] = oracle_probabilities

    result[
        "oracle_action"
    ] = oracle_actions

    output_path = (
        Path(output_dir)
        / "policy_assignments.parquet"
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    log(
        f"Saved policy assignments: "
        f"{output_path}"
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(summary):
    print()
    print("=" * 100)
    print("RECOVERYOS POLICY EVALUATION")
    print("=" * 100)
    print()

    for _, row in summary.iterrows():

        print(
            f"{str(row['policy']):25s} | "
            f"Recovery: "
            f"{pct(row['expected_recovery_probability']):>9s} | "
            f"Revenue: "
            f"{money(row['expected_revenue']):>14s} | "
            f"Efficiency: "
            f"{pct(row['revenue_efficiency_vs_oracle']):>9s} | "
            f"Regret: "
            f"{money(row['revenue_regret_vs_oracle']):>14s}"
        )

    print()
    print("=" * 100)


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RecoveryOS recovery policies "
            "using synthetic counterfactual ground truth."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Evaluation dataset. Recommended: "
            "razorpay_recovery_v2_20m.parquet"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="policy_evaluation",
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--retry-30m-cost",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--retry-evening-cost",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--payment-link-cost",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--whatsapp-cost",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--alternate-method-cost",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--stop-cost",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--no-segments",
        action="store_true",
        help="Skip detailed segment analysis.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    ensure_dir(output_dir)

    costs = {
        "retry_30m": args.retry_30m_cost,
        "retry_evening": args.retry_evening_cost,
        "payment_link": args.payment_link_cost,
        "whatsapp_reminder": args.whatsapp_cost,
        "alternate_method": args.alternate_method_cost,
        "stop": args.stop_cost,
    }

    # --------------------------------------------------------
    # Required evaluation columns
    # --------------------------------------------------------

    required_columns = [
        "amount",
        "failure_reason",
        "observed_intervention",
        "optimal_intervention",
        "oracle_recovery_probability",
        "oracle_expected_revenue",
    ]

    required_columns.extend(
        PROB_COLUMNS.values()
    )

    # Add useful segment columns if present.
    optional_columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "customer_value",
        "payment_method",
        "device",
        "merchant_category",
        "location",
    ]

    columns = list(
        dict.fromkeys(
            required_columns
            + optional_columns
        )
    )

    df = load_data(
        args.input,
        columns=columns,
    )

    validate_ground_truth(df)

    # --------------------------------------------------------
    # Basic dataset information
    # --------------------------------------------------------

    log(
        f"Rows: {len(df):,}"
    )

    log(
        f"Total failed amount: "
        f"{money(df['amount'].sum())}"
    )

    # --------------------------------------------------------
    # Baseline policies
    # --------------------------------------------------------

    (
        summary,
        policies,
        oracle_actions,
        oracle_probabilities,
    ) = evaluate_baselines(
        df,
        output_dir,
        costs,
    )

    # --------------------------------------------------------
    # Detailed segment analysis
    # --------------------------------------------------------

    if not args.no_segments:

        detailed_analysis(
            df,
            policies,
            oracle_actions,
            oracle_probabilities,
            output_dir,
            costs,
        )

    # --------------------------------------------------------
    # Save assignments
    # --------------------------------------------------------

    save_policy_assignments(
        df,
        policies,
        oracle_actions,
        oracle_probabilities,
        output_dir,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "input": str(args.input),
        "rows": len(df),
        "actions": ACTIONS,
        "counterfactual_probability_columns": PROB_COLUMNS,
        "costs": costs,
        "notes": [
            "p_* columns are used only for synthetic evaluation.",
            "p_* columns are not model features.",
            "oracle is synthetic ground truth.",
            "revenue values are synthetic and must not be presented as real Razorpay revenue.",
        ],
    }

    with open(
        output_dir / "evaluation_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(summary)

    log(
        f"Results written to: "
        f"{output_dir.resolve()}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "The next step is to evaluate the trained "
        "CatBoost/XGBoost/LightGBM models as "
        "action-conditioned policies."
    )
    print(
        "That requires the exact action/intervention "
        "encoding used by train_parallel_gpu.py."
    )


if __name__ == "__main__":
    main()
