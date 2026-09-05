

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ACTION_ORDER = [
    "retry_30m",
    "retry_evening",
    "payment_link",
    "whatsapp_reminder",
    "alternate_method",
    "stop",
]

# Exact feature order used during model training.
MODEL_FEATURES = [
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

MODEL_CAT_FEATURES = [
    "payment_method",
    "failure_reason",
    "device",
    "location",
    "customer_value",
    "merchant_category",
    "observed_intervention",
]

# Columns that must NEVER enter the ML model.
# These contain outcomes, counterfactuals, IDs, or oracle information.
EXCLUDED_FROM_MODEL = {
    "transaction_id",
    "customer_id",
    "merchant_id",
    "timestamp",

    # Observed outcome / policy information
    "observed_recovered",
    "observed_recovery_probability",
    "observed_expected_revenue",

    # Counterfactual ground truth
    "p_retry_30m",
    "p_retry_evening",
    "p_payment_link",
    "p_whatsapp_reminder",
    "p_alternate_method",
    "p_stop",

    # Oracle information
    "optimal_intervention",
    "oracle_recovery_probability",
    "oracle_expected_revenue",
}


# Default action costs.
# Currently zero, matching the previous benchmark.
ACTION_COSTS = {
    "retry_30m": 0.0,
    "retry_evening": 0.0,
    "payment_link": 0.0,
    "whatsapp_reminder": 0.0,
    "alternate_method": 0.0,
    "stop": 0.0,
}


# ============================================================
# HELPERS
# ============================================================

def log(msg):
    print(f"[RecoveryOS] {msg}")


def format_rupees(value):
    """
    Format INR into crore/billion style for readability.
    """
    value = float(value)

    if abs(value) >= 1e9:
        return f"₹{value / 1e9:.3f}B"
    elif abs(value) >= 1e7:
        return f"₹{value / 1e7:.2f}Cr"
    elif abs(value) >= 1e5:
        return f"₹{value / 1e5:.2f}L"
    else:
        return f"₹{value:,.2f}"


def prepare_features(df):
    """
    Prepare exact CatBoost input.

    Critical:
    - Exact 20 features
    - Exact training order
    - observed_intervention is action-conditioned
    """

    missing = [c for c in MODEL_FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required model features:\n"
            + "\n".join(missing)
        )

    X = df[MODEL_FEATURES].copy()

    # CatBoost categorical columns should be strings.
    for col in MODEL_CAT_FEATURES:
        X[col] = X[col].astype(str)

    return X


def load_model(model_path):
    log(f"Loading model: {model_path}")

    model = CatBoostClassifier()
    model.load_model(model_path)

    model_names = list(model.feature_names_)
    feature_count = len(model_names)

    log(f"Model feature count: {feature_count}")

    if feature_count != len(MODEL_FEATURES):
        raise ValueError(
            f"Model expects {feature_count} features, "
            f"but evaluator expects {len(MODEL_FEATURES)}."
        )

    # Validate feature names when available.
    if model_names:
        if model_names != MODEL_FEATURES:
            print("\n[ERROR] Model feature order mismatch!")
            print("Model:")
            for i, name in enumerate(model_names):
                print(f"  {i}: {name}")

            print("\nEvaluator:")
            for i, name in enumerate(MODEL_FEATURES):
                print(f"  {i}: {name}")

            raise ValueError(
                "CatBoost model feature order does not match evaluator."
            )

    log("Model feature order validated.")

    return model


def predict_probability(model, df):
    """
    Predict recovery probability.
    """

    X = prepare_features(df)

    probabilities = model.predict_proba(X)[:, 1]

    return probabilities


# ============================================================
# BASELINE POLICIES
# ============================================================

def evaluate_historical(df):
    recovered = df["observed_recovered"].astype(float).values
    amount = df["amount"].astype(float).values

    revenue = np.sum(amount * recovered)

    total_amount = np.sum(amount)

    recovery_rate = np.sum(recovered) / len(df)

    return {
        "policy": "historical",
        "recovery_rate": recovery_rate,
        "revenue": revenue,
        "recovered_amount": revenue,
        "total_amount": total_amount,
    }


def evaluate_fixed_action(df, action):
    """
    Evaluate a fixed intervention using the synthetic counterfactual
    recovery probability associated with that action.
    """

    probability_column = f"p_{action}"

    if probability_column not in df.columns:
        raise ValueError(
            f"Missing counterfactual column: {probability_column}"
        )

    amount = df["amount"].astype(float).values
    probability = df[probability_column].astype(float).values

    cost = ACTION_COSTS.get(action, 0.0)

    expected_revenue = probability * amount - cost

    revenue = np.sum(expected_revenue)

    recovery_rate = np.sum(probability) / len(df)

    return {
        "policy": f"always_{action}",
        "recovery_rate": recovery_rate,
        "revenue": revenue,
        "recovered_amount": revenue,
        "total_amount": np.sum(amount),
    }


def evaluate_simple_rules(df):
    """
    Same simple rule policy used in the previous benchmark.

    Rule:
    - bank_timeout / network_error -> retry_evening
    - insufficient_funds -> retry_evening
    - expired_card / authentication_failed / limit_exceeded
      -> alternate_method
    - user_cancelled -> whatsapp_reminder
    """

    actions = np.full(len(df), "retry_evening", dtype=object)

    failure = df["failure_reason"].astype(str).values

    actions[
        np.isin(
            failure,
            [
                "expired_card",
                "authentication_failed",
                "limit_exceeded",
            ],
        )
    ] = "alternate_method"

    actions[failure == "user_cancelled"] = "whatsapp_reminder"

    revenue = np.zeros(len(df), dtype=float)
    probability = np.zeros(len(df), dtype=float)

    amount = df["amount"].astype(float).values

    for action in ACTION_ORDER:
        mask = actions == action

        if not np.any(mask):
            continue

        p_col = f"p_{action}"

        if p_col in df.columns:
            p = df.loc[mask, p_col].astype(float).values
        else:
            p = np.zeros(np.sum(mask))

        probability[mask] = p

        cost = ACTION_COSTS.get(action, 0.0)

        revenue[mask] = p * amount[mask] - cost

    return {
        "policy": "simple_rules",
        "recovery_rate": np.mean(probability),
        "revenue": np.sum(revenue),
        "recovered_amount": np.sum(revenue),
        "total_amount": np.sum(amount),
    }


def evaluate_oracle(df):
    probability = df["oracle_recovery_probability"].astype(float).values
    amount = df["amount"].astype(float).values

    action = df["optimal_intervention"].astype(str).values

    revenue = np.zeros(len(df), dtype=float)

    for candidate in ACTION_ORDER:
        mask = action == candidate

        if not np.any(mask):
            continue

        cost = ACTION_COSTS.get(candidate, 0.0)

        revenue[mask] = (
            probability[mask] * amount[mask] - cost
        )

    return {
        "policy": "oracle",
        "recovery_rate": np.mean(probability),
        "revenue": np.sum(revenue),
        "recovered_amount": np.sum(revenue),
        "total_amount": np.sum(amount),
    }


# ============================================================
# AI POLICY
# ============================================================

def generate_ai_policy(
    df,
    model,
    chunk_size=100000,
):
    """
    Evaluate all candidate interventions.

    For each transaction:

        predicted recovery probability(action)
        × transaction amount
        - intervention cost

    The action with the highest expected revenue is selected.
    """

    n = len(df)

    selected_action = np.empty(n, dtype=object)
    selected_probability = np.zeros(n, dtype=np.float32)
    selected_expected_revenue = np.zeros(n, dtype=np.float64)

    amount_all = df["amount"].astype(float).values

    log(
        f"Generating AI policy for {n:,} held-out rows "
        f"({len(ACTION_ORDER)} candidate actions, "
        f"chunk={chunk_size:,})..."
    )

    start = time.time()

    for start_idx in range(0, n, chunk_size):

        end_idx = min(start_idx + chunk_size, n)

        chunk = df.iloc[start_idx:end_idx].copy()

        amount = amount_all[start_idx:end_idx]

        best_action = np.empty(len(chunk), dtype=object)
        best_probability = np.zeros(len(chunk), dtype=np.float32)

        best_revenue = np.full(
            len(chunk),
            -np.inf,
            dtype=np.float64,
        )

        for action in ACTION_ORDER:

            # Action-conditioned model input.
            chunk_action = chunk.copy()
            chunk_action["observed_intervention"] = action

            probability = predict_probability(
                model,
                chunk_action,
            )

            cost = ACTION_COSTS.get(action, 0.0)

            expected_revenue = (
                probability * amount - cost
            )

            better = expected_revenue > best_revenue

            best_revenue[better] = expected_revenue[better]
            best_probability[better] = probability[better]
            best_action[better] = action

        selected_action[start_idx:end_idx] = best_action
        selected_probability[start_idx:end_idx] = best_probability
        selected_expected_revenue[start_idx:end_idx] = best_revenue

        if (
            start_idx == 0
            or end_idx == n
            or (start_idx // chunk_size) % 10 == 0
        ):
            elapsed = time.time() - start

            log(
                f"  AI policy progress: "
                f"{end_idx:,}/{n:,} "
                f"({100 * end_idx / n:.1f}%) "
                f"| elapsed {elapsed / 60:.1f} min"
            )

    return (
        selected_action,
        selected_probability,
        selected_expected_revenue,
    )


def evaluate_ai_policy(
    df,
    model,
    chunk_size,
):
    (
        selected_action,
        selected_probability,
        selected_expected_revenue,
    ) = generate_ai_policy(
        df,
        model,
        chunk_size,
    )

    return {
        "policy": "ai_catboost",
        "recovery_rate": np.mean(selected_probability),
        "revenue": np.sum(selected_expected_revenue),
        "recovered_amount": np.sum(selected_expected_revenue),
        "total_amount": np.sum(df["amount"].astype(float)),
        "selected_action": selected_action,
        "selected_probability": selected_probability,
        "selected_expected_revenue": selected_expected_revenue,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Held-out recovery_test.parquet",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Trained CatBoost .cbm model",
    )

    parser.add_argument(
        "--output-dir",
        default="policy_evaluation_ai_heldout",
    )

    parser.add_argument(
        "--prediction-chunk-size",
        type=int,
        default=100000,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    log(f"Loading STRICT HELD-OUT TEST DATA: {args.input}")

    load_start = time.time()

    df = pd.read_parquet(args.input)

    log(
        f"Loaded {len(df):,} rows in "
        f"{time.time() - load_start:.2f}s"
    )

    log(f"Test rows: {len(df):,}")

    if "customer_id" in df.columns:
        customers = df["customer_id"].nunique()
        log(f"Unique test customers: {customers:,}")

    total_amount = df["amount"].astype(float).sum()

    log(
        f"Total held-out failed amount: "
        f"{format_rupees(total_amount)}"
    )

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    required_columns = [
        "amount",
        "observed_recovered",
        "observed_intervention",
        "oracle_recovery_probability",
        "oracle_expected_revenue",
        "optimal_intervention",
    ]

    required_columns += [
        f"p_{action}"
        for action in ACTION_ORDER
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Test dataset is missing required evaluation columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model(args.model)

    log(f"AI model type: catboost")
    log(f"AI feature count: {len(MODEL_FEATURES)}")
    log(f"Action feature: observed_intervention")
    log(
        f"Prediction chunk size: "
        f"{args.prediction_chunk_size:,}"
    )

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------

    results = []

    log("Evaluating policy: historical")

    historical = evaluate_historical(df)
    results.append(historical)

    log("Evaluating policy: always_retry_evening")

    retry_evening = evaluate_fixed_action(
        df,
        "retry_evening",
    )
    results.append(retry_evening)

    log("Evaluating policy: always_retry_30m")

    retry_30m = evaluate_fixed_action(
        df,
        "retry_30m",
    )
    results.append(retry_30m)

    log("Evaluating policy: simple_rules")

    simple_rules = evaluate_simple_rules(df)
    results.append(simple_rules)

    log("Evaluating policy: oracle")

    oracle = evaluate_oracle(df)
    results.append(oracle)

    # --------------------------------------------------------
    # AI POLICY
    # --------------------------------------------------------

    print()
    log("=" * 70)
    log("EVALUATING AI POLICY ON STRICT HELD-OUT CUSTOMERS")
    log("=" * 70)

    ai = evaluate_ai_policy(
        df,
        model,
        args.prediction_chunk_size,
    )

    results.append(ai)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    oracle_revenue = oracle["revenue"]

    for result in results:

        result["oracle_efficiency"] = (
            result["revenue"] / oracle_revenue
            if oracle_revenue > 0
            else 0.0
        )

        result["regret"] = (
            oracle_revenue - result["revenue"]
        )

    # --------------------------------------------------------
    # SAVE AI ASSIGNMENTS
    # --------------------------------------------------------

    assignments = pd.DataFrame({
        "customer_id": (
            df["customer_id"].values
            if "customer_id" in df.columns
            else np.arange(len(df))
        ),
        "transaction_id": (
            df["transaction_id"].values
            if "transaction_id" in df.columns
            else np.arange(len(df))
        ),
        "amount": df["amount"].values,
        "failure_reason": df["failure_reason"].values,
        "observed_intervention": df["observed_intervention"].values,
        "ai_selected_intervention": ai["selected_action"],
        "ai_predicted_recovery_probability": (
            ai["selected_probability"]
        ),
        "ai_expected_revenue": (
            ai["selected_expected_revenue"]
        ),
        "optimal_intervention": (
            df["optimal_intervention"].values
        ),
        "oracle_recovery_probability": (
            df["oracle_recovery_probability"].values
        ),
        "oracle_expected_revenue": (
            df["oracle_expected_revenue"].values
        ),
    })

    assignment_path = os.path.join(
        args.output_dir,
        "heldout_policy_assignments.parquet",
    )

    assignments.to_parquet(
        assignment_path,
        index=False,
    )

    log(
        f"Saved held-out policy assignments: "
        f"{assignment_path}"
    )

    # --------------------------------------------------------
    # SAVE SUMMARY CSV
    # --------------------------------------------------------

    summary_rows = []

    for result in results:

        summary_rows.append({
            "policy": result["policy"],
            "recovery_rate": result["recovery_rate"],
            "revenue": result["revenue"],
            "oracle_efficiency": result["oracle_efficiency"],
            "regret": result["regret"],
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        args.output_dir,
        "heldout_policy_summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    json_results = []

    for result in results:

        json_results.append({
            "policy": result["policy"],
            "recovery_rate": float(
                result["recovery_rate"]
            ),
            "revenue": float(
                result["revenue"]
            ),
            "oracle_efficiency": float(
                result["oracle_efficiency"]
            ),
            "regret": float(
                result["regret"]
            ),
        })

    metadata = {
        "evaluation_type": "strict_customer_held_out",
        "input": os.path.abspath(args.input),
        "model": os.path.abspath(args.model),
        "rows": int(len(df)),
        "unique_customers": (
            int(df["customer_id"].nunique())
            if "customer_id" in df.columns
            else None
        ),
        "model_features": MODEL_FEATURES,
        "categorical_features": MODEL_CAT_FEATURES,
        "candidate_actions": ACTION_ORDER,
        "results": json_results,
    }

    json_path = os.path.join(
        args.output_dir,
        "heldout_policy_evaluation.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("RECOVERYOS — STRICT CUSTOMER-HELD-OUT POLICY EVALUATION")
    print("=" * 100)
    print()

    for result in results:

        print(
            f"{result['policy']:<26} | "
            f"Recovery: "
            f"{result['recovery_rate'] * 100:7.3f}% | "
            f"Revenue: "
            f"{format_rupees(result['revenue']):>14} | "
            f"Efficiency: "
            f"{result['oracle_efficiency'] * 100:7.3f}% | "
            f"Regret: "
            f"{format_rupees(result['regret']):>14}"
        )

    print()
    print("=" * 100)

    print()
    print(
        f"Held-out customers: "
        f"{df['customer_id'].nunique():,}"
        if "customer_id" in df.columns
        else f"Held-out rows: {len(df):,}"
    )

    print(
        f"Held-out transactions: "
        f"{len(df):,}"
    )

    print(
        f"Oracle revenue: "
        f"{format_rupees(oracle_revenue)}"
    )

    print(
        f"AI revenue: "
        f"{format_rupees(ai['revenue'])}"
    )

    print(
        f"AI vs oracle: "
        f"{ai['oracle_efficiency'] * 100:.3f}%"
    )

    print(
        f"AI regret: "
        f"{format_rupees(ai['regret'])}"
    )

    print()
    print(
        f"[RecoveryOS] Results written to: "
        f"{os.path.abspath(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
