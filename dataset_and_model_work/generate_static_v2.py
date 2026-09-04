#!/usr/bin/env python3
"""
Razorpay RecoveryOS - V2 Synthetic Causal Dataset Generator

Purpose
-------
Generate a large, realistic synthetic payment-recovery dataset for
training/evaluating an AI revenue recovery system.

Important properties
--------------------
1. Repeated customer identities with latent customer behavior.
2. Repeated merchant identities with latent merchant behavior.
3. Realistic payment failure patterns.
4. Observational intervention policy (NOT random uniform treatment).
5. Potential outcomes for ALL possible interventions.
6. Observed outcome for the intervention actually selected.
7. Oracle / ground-truth optimal intervention.
8. Expected recovery revenue for every action.
9. Chunked generation so 10M+ rows can be generated without
   holding the complete dataset in memory.

Output
------
Parquet is recommended.

Example
-------
python generate_v2.py \
    --rows 10000000 \
    --customers 1000000 \
    --merchants 10000 \
    --output razorpay_recovery_v2_10m.parquet

Dependencies
------------
pip install numpy pandas pyarrow
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ACTIONS: List[str] = [
    "retry_30m",
    "retry_evening",
    "payment_link",
    "whatsapp_reminder",
    "alternate_method",
    "stop",
]

MERCHANT_CATEGORIES: List[str] = [
    "SaaS",
    "E-commerce",
    "EdTech",
    "Gaming",
    "Financial Services",
    "Health & Fitness",
    "OTT / Media",
]

PAYMENT_METHODS: List[str] = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Netbanking",
    "BNPL",
    "Wallet",
]

FAILURE_REASONS: List[str] = [
    "insufficient_funds",
    "bank_timeout",
    "authentication_failed",
    "expired_card",
    "limit_exceeded",
    "user_cancelled",
    "network_error",
]

DEVICES: List[str] = [
    "Android",
    "iOS",
    "Web-Desktop",
    "Web-Mobile",
]

LOCATIONS: List[str] = [
    "Mumbai, MH",
    "Bengaluru, KA",
    "Delhi, DL",
    "Hyderabad, TS",
    "Pune, MH",
    "Chennai, TN",
    "Kolkata, WB",
    "Ahmedabad, GJ",
    "Jaipur, RJ",
]

CUSTOMER_VALUES: List[str] = [
    "Low",
    "Medium",
    "High",
    "VIP",
]

# Approximate relative intervention cost.
# These aren't real Razorpay prices; they are simulation penalties
# so that the optimizer doesn't blindly maximize recovery probability.
ACTION_COST = {
    "retry_30m": 0.50,
    "retry_evening": 0.50,
    "payment_link": 0.20,
    "whatsapp_reminder": 0.15,
    "alternate_method": 0.35,
    "stop": 0.00,
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    """
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Row-wise softmax.
    """
    x = logits / temperature
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def weighted_choice(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    values: List[str],
) -> np.ndarray:
    """
    Vectorized categorical sampling.

    probabilities:
        shape = (N, K)
        each row sums to 1.
    """
    cumulative = np.cumsum(probabilities, axis=1)
    r = rng.random(probabilities.shape[0])[:, None]

    indices = (r > cumulative).sum(axis=1)
    indices = np.minimum(indices, len(values) - 1)

    return np.asarray(values, dtype=object)[indices]


def generate_ids(prefix: str, numbers: np.ndarray, width: int = 8) -> np.ndarray:
    """
    Convert integer IDs to strings.
    """
    return np.char.add(
        prefix,
        np.char.zfill(numbers.astype(str), width),
    )


# ============================================================
# CUSTOMER PROFILE
# ============================================================

def generate_customer_profiles(
    rng: np.random.Generator,
    num_customers: int,
) -> pd.DataFrame:
    """
    Generate latent customer properties.

    These profiles are reused across transactions, which gives us
    actual customer-level structure rather than independent random
    rows.
    """

    customer_id = np.arange(num_customers)

    # Latent payment reliability.
    # Beta(7, 2) => generally reliable customers, with variation.
    success_rate = rng.beta(
        7.0,
        2.0,
        size=num_customers,
    )

    # Latent customer monetary value.
    value_score = rng.beta(
        2.0,
        5.0,
        size=num_customers,
    )

    customer_value = np.select(
        [
            value_score >= 0.92,
            value_score >= 0.70,
            value_score >= 0.40,
        ],
        [
            "VIP",
            "High",
            "Medium",
        ],
        default="Low",
    )

    # Preferred payment hour.
    preferred_hour = rng.integers(
        low=0,
        high=24,
        size=num_customers,
    )

    # Customer average transaction amount.
    avg_amount = rng.lognormal(
        mean=7.2,
        sigma=0.65,
        size=num_customers,
    )

    avg_amount = np.clip(
        avg_amount,
        99.0,
        150000.0,
    )

    # Salary-cycle behavior.
    salary_sensitive = rng.beta(
        2.0,
        4.0,
        size=num_customers,
    )

    # How likely the customer is to tolerate payment reminders.
    contact_tolerance = rng.beta(
        5.0,
        2.0,
        size=num_customers,
    )

    # Historical retry tendency.
    retry_tendency = rng.poisson(
        lam=0.8,
        size=num_customers,
    )

    retry_tendency = np.clip(
        retry_tendency,
        0,
        5,
    )

    return pd.DataFrame(
        {
            "customer_num": customer_id,
            "customer_success_rate": success_rate,
            "customer_value_score": value_score,
            "customer_value": customer_value,
            "preferred_hour": preferred_hour,
            "customer_avg_amount": avg_amount,
            "salary_sensitive": salary_sensitive,
            "contact_tolerance": contact_tolerance,
            "retry_tendency": retry_tendency,
        }
    )


# ============================================================
# MERCHANT PROFILE
# ============================================================

def generate_merchant_profiles(
    rng: np.random.Generator,
    num_merchants: int,
) -> pd.DataFrame:
    """
    Generate merchant-level behavior.
    """

    merchant_num = np.arange(num_merchants)

    category = rng.choice(
        MERCHANT_CATEGORIES,
        size=num_merchants,
        p=[
            0.20,
            0.28,
            0.13,
            0.10,
            0.10,
            0.09,
            0.10,
        ],
    )

    merchant_quality = rng.beta(
        7.0,
        2.5,
        size=num_merchants,
    )

    avg_ticket = rng.lognormal(
        mean=7.0,
        sigma=0.7,
        size=num_merchants,
    )

    avg_ticket = np.clip(
        avg_ticket,
        99.0,
        100000.0,
    )

    return pd.DataFrame(
        {
            "merchant_num": merchant_num,
            "merchant_category": category,
            "merchant_quality": merchant_quality,
            "merchant_avg_ticket": avg_ticket,
        }
    )


# ============================================================
# GENERATE ONE CHUNK
# ============================================================

def generate_chunk(
    rng: np.random.Generator,
    customer_profiles: pd.DataFrame,
    merchant_profiles: pd.DataFrame,
    n: int,
    chunk_start: int,
) -> pd.DataFrame:

    customer_count = len(customer_profiles)
    merchant_count = len(merchant_profiles)

    # --------------------------------------------------------
    # 1. Select recurring customers and merchants
    # --------------------------------------------------------

    customer_idx = rng.integers(
        0,
        customer_count,
        size=n,
    )

    merchant_idx = rng.integers(
        0,
        merchant_count,
        size=n,
    )

    customer = customer_profiles.iloc[customer_idx].reset_index(drop=True)
    merchant = merchant_profiles.iloc[merchant_idx].reset_index(drop=True)

    # --------------------------------------------------------
    # 2. Payment timestamp
    # --------------------------------------------------------

    start_timestamp = np.datetime64("2026-01-01T00:00:00")
    end_timestamp = np.datetime64("2026-08-31T23:59:59")

    start_seconds = int(
        (start_timestamp - np.datetime64("1970-01-01T00:00:00"))
        / np.timedelta64(1, "s")
    )

    end_seconds = int(
        (end_timestamp - np.datetime64("1970-01-01T00:00:00"))
        / np.timedelta64(1, "s")
    )

    timestamp_seconds = rng.integers(
        start_seconds,
        end_seconds,
        size=n,
    )

    timestamps = (
        timestamp_seconds.astype("datetime64[s]")
    )

    dt = pd.to_datetime(timestamps)

    failure_hour = dt.hour.to_numpy()
    day_of_week = dt.dayofweek.to_numpy()
    day_of_month = dt.day.to_numpy()

    is_weekend = (day_of_week >= 5)

    # Salary window approximation.
    is_salary_window = (
        (day_of_month <= 5)
        | (day_of_month >= 28)
    )

    # --------------------------------------------------------
    # 3. Basic transaction properties
    # --------------------------------------------------------

    amounts = (
        rng.lognormal(
            mean=np.log(
                np.maximum(
                    merchant["merchant_avg_ticket"].to_numpy(),
                    100.0,
                )
            ),
            sigma=0.45,
        )
    )

    amounts = np.clip(
        amounts,
        99.0,
        150000.0,
    )

    # Payment method correlated with merchant category.
    payment_methods = rng.choice(
        PAYMENT_METHODS,
        size=n,
        p=[
            0.55,
            0.20,
            0.10,
            0.08,
            0.04,
            0.03,
        ],
    )

    failure_reasons = rng.choice(
        FAILURE_REASONS,
        size=n,
        p=[
            0.34,
            0.21,
            0.14,
            0.09,
            0.08,
            0.06,
            0.08,
        ],
    )

    devices = rng.choice(
        DEVICES,
        size=n,
        p=[
            0.62,
            0.16,
            0.14,
            0.08,
        ],
    )

    locations = rng.choice(
        LOCATIONS,
        size=n,
    )

    # --------------------------------------------------------
    # 4. Historical/customer features
    # --------------------------------------------------------

    previous_success_rate = np.clip(
        customer["customer_success_rate"].to_numpy()
        + rng.normal(0, 0.035, n),
        0.05,
        0.995,
    )

    days_since_last_payment = np.clip(
        rng.exponential(
            scale=18.0,
            size=n,
        ).astype(int),
        0,
        180,
    )

    subscription_age_days = rng.integers(
        7,
        1000,
        size=n,
    )

    historical_retries = np.clip(
        customer["retry_tendency"].to_numpy()
        + rng.poisson(0.35, n),
        0,
        5,
    )

    time_since_failure_mins = np.clip(
        rng.exponential(
            scale=120.0,
            size=n,
        ).astype(int) + 1,
        1,
        1440,
    )

    # --------------------------------------------------------
    # 5. Device/network context
    # --------------------------------------------------------

    network_quality = rng.beta(
        6.0,
        2.0,
        size=n,
    )

    # Fraud / abuse is NOT the goal here, but poor quality should
    # still make some transactions harder to recover.
    customer_fatigue = np.clip(
        historical_retries / 5.0
        + rng.normal(0, 0.08, n),
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 6. Base recovery score
    # --------------------------------------------------------

    base_logit = (
        -0.85
        + 2.5 * (previous_success_rate - 0.5)
        + 0.35 * merchant["merchant_quality"].to_numpy()
        + 0.30 * network_quality
        - 0.60 * customer_fatigue
    )

    # High-value customers receive slightly better recovery odds.
    base_logit += np.select(
        [
            customer["customer_value"].to_numpy() == "VIP",
            customer["customer_value"].to_numpy() == "High",
            customer["customer_value"].to_numpy() == "Medium",
        ],
        [
            0.35,
            0.20,
            0.05,
        ],
        default=-0.05,
    )

    # Salary window.
    base_logit += np.where(
        is_salary_window,
        0.25 * customer["salary_sensitive"].to_numpy(),
        -0.05,
    )

    # Weekend effect.
    base_logit += np.where(
        is_weekend,
        -0.08,
        0.0,
    )

    # --------------------------------------------------------
    # 7. Potential outcomes
    #
    # For each row we calculate P(recovery | action).
    # --------------------------------------------------------

    probabilities = {}

    for action in ACTIONS:

        logits = base_logit.copy()

        # ================================================
        # Insufficient funds
        # ================================================

        mask = failure_reasons == "insufficient_funds"

        if action == "retry_evening":
            # Strong effect around salary cycle / evening.
            logits += (
                1.30
                + np.where(
                    is_salary_window,
                    0.90,
                    0.0,
                )
            )

            evening_bonus = (
                ((failure_hour >= 18) & (failure_hour <= 22))
            )

            logits += np.where(
                evening_bonus,
                0.35,
                0.0,
            )

        elif action == "retry_30m":
            logits += np.where(
                is_salary_window,
                0.70,
                -0.90,
            )

        elif action == "payment_link":
            logits += 0.25

        elif action == "whatsapp_reminder":
            logits += (
                0.20
                + 0.40
                * customer["contact_tolerance"].to_numpy()
            )

        elif action == "alternate_method":
            logits += 0.15

        # ================================================
        # Bank timeout
        # ================================================

        mask_timeout = failure_reasons == "bank_timeout"

        if action == "retry_30m":
            logits += (
                mask_timeout
                * 1.80
            )

        elif action == "retry_evening":
            logits += (
                mask_timeout
                * 0.70
            )

        elif action == "stop":
            logits += (
                mask_timeout
                * -1.50
            )

        # ================================================
        # Authentication failure
        # ================================================

        mask_auth = (
            failure_reasons
            == "authentication_failed"
        )

        if action == "payment_link":
            logits += (
                mask_auth
                * 1.10
            )

        elif action == "whatsapp_reminder":
            logits += (
                mask_auth
                * 1.40
            )

        elif action == "retry_30m":
            logits += (
                mask_auth
                * -0.70
            )

        # ================================================
        # Expired card
        # ================================================

        mask_expired = (
            failure_reasons
            == "expired_card"
        )

        if action == "alternate_method":
            logits += (
                mask_expired
                * 2.10
            )

        elif action == "payment_link":
            logits += (
                mask_expired
                * 1.10
            )

        elif action == "retry_30m":
            logits += (
                mask_expired
                * -2.20
            )

        elif action == "retry_evening":
            logits += (
                mask_expired
                * -2.00
            )

        # ================================================
        # Limit exceeded
        # ================================================

        mask_limit = (
            failure_reasons
            == "limit_exceeded"
        )

        if action == "alternate_method":
            logits += (
                mask_limit
                * 1.35
            )

        elif action == "payment_link":
            logits += (
                mask_limit
                * 0.75
            )

        elif action == "retry_30m":
            logits += (
                mask_limit
                * -1.00
            )

        # ================================================
        # User cancelled
        # ================================================

        mask_cancelled = (
            failure_reasons
            == "user_cancelled"
        )

        if action == "whatsapp_reminder":
            logits += (
                mask_cancelled
                * 1.15
            )

        elif action == "payment_link":
            logits += (
                mask_cancelled
                * 1.00
            )

        elif action == "retry_30m":
            logits += (
                mask_cancelled
                * -1.10
            )

        # ================================================
        # Network error
        # ================================================

        mask_network = (
            failure_reasons
            == "network_error"
        )

        if action == "retry_30m":
            logits += (
                mask_network
                * 1.60
            )

        elif action == "retry_evening":
            logits += (
                mask_network
                * 0.60
            )

        # ================================================
        # Customer preferred time
        # ================================================

        hour_distance = np.minimum(
            np.abs(
                failure_hour
                - customer["preferred_hour"].to_numpy()
            ),
            24
            - np.abs(
                failure_hour
                - customer["preferred_hour"].to_numpy()
            ),
        )

        preferred_time_bonus = np.exp(
            -hour_distance / 5.0
        )

        if action == "retry_evening":
            logits += (
                0.55
                * preferred_time_bonus
            )

        # ================================================
        # Fatigue penalty
        # ================================================

        if action in (
            "retry_30m",
            "retry_evening",
        ):
            logits -= (
                0.70
                * customer_fatigue
            )

        if action in (
            "whatsapp_reminder",
            "payment_link",
        ):
            logits -= (
                0.20
                * customer_fatigue
            )

        # ================================================
        # STOP action
        # ================================================

        if action == "stop":
            logits = np.full(
                n,
                -10.0,
                dtype=np.float64,
            )

        probabilities[action] = np.clip(
            sigmoid(logits),
            0.01,
            0.98,
        )

    # --------------------------------------------------------
    # 8. Build potential outcome matrix
    # --------------------------------------------------------

    probability_matrix = np.column_stack(
        [probabilities[action] for action in ACTIONS]
    )

    # --------------------------------------------------------
    # 9. Expected revenue for every intervention
    # --------------------------------------------------------

    expected_revenue_matrix = np.zeros_like(
        probability_matrix
    )

    for idx, action in enumerate(ACTIONS):

        cost = ACTION_COST[action]

        expected_revenue_matrix[:, idx] = (
            probability_matrix[:, idx]
            * amounts
            - cost
        )

    # --------------------------------------------------------
    # 10. Oracle action
    # --------------------------------------------------------

    optimal_indices = np.argmax(
        expected_revenue_matrix,
        axis=1,
    )

    optimal_intervention = np.asarray(
        ACTIONS,
        dtype=object,
    )[optimal_indices]

    oracle_recovery_probability = (
        probability_matrix[
            np.arange(n),
            optimal_indices,
        ]
    )

    oracle_expected_revenue = (
        expected_revenue_matrix[
            np.arange(n),
            optimal_indices,
        ]
    )

    # --------------------------------------------------------
    # 11. Historical intervention policy
    #
    # IMPORTANT:
    # This is intentionally NOT uniformly random.
    #
    # It creates an observational dataset where the action choice
    # depends on customer/payment context.
    # --------------------------------------------------------

    intervention_logits = np.zeros(
        (n, len(ACTIONS)),
        dtype=np.float64,
    )

    for i, action in enumerate(ACTIONS):

        score = np.zeros(n)

        # Some baseline preference for common interventions.
        if action == "retry_30m":
            score += 0.50

        elif action == "retry_evening":
            score += 0.30

        elif action == "payment_link":
            score += 0.10

        elif action == "whatsapp_reminder":
            score += 0.05

        elif action == "alternate_method":
            score -= 0.10

        elif action == "stop":
            score -= 2.00

        # Operators historically make simple rule-based choices.
        score += np.where(
            (
                failure_reasons
                == "bank_timeout"
            )
            & (
                action == "retry_30m"
            ),
            1.40,
            0.0,
        )

        score += np.where(
            (
                failure_reasons
                == "expired_card"
            )
            & (
                action == "alternate_method"
            ),
            1.20,
            0.0,
        )

        score += np.where(
            (
                failure_reasons
                == "insufficient_funds"
            )
            & (
                action == "retry_evening"
            ),
            0.85,
            0.0,
        )

        score += np.where(
            (
                customer["customer_value"].to_numpy()
                == "VIP"
            )
            & (
                action
                in (
                    "payment_link",
                    "whatsapp_reminder",
                )
            ),
            0.50,
            0.0,
        )

        # More retries makes humans/system less likely to retry.
        if action in (
            "retry_30m",
            "retry_evening",
        ):
            score -= (
                0.45
                * historical_retries
            )

        # Make historical policy imperfect.
        score += rng.normal(
            0,
            0.15,
            size=n,
        )

        intervention_logits[:, i] = score

    intervention_probabilities = softmax(
        intervention_logits,
        temperature=0.85,
    )

    observed_intervention = weighted_choice(
        rng,
        intervention_probabilities,
        ACTIONS,
    )

    # --------------------------------------------------------
    # 12. Observed recovery
    #
    # Only the selected action actually happens.
    # --------------------------------------------------------

    observed_action_indices = np.asarray(
        [ACTIONS.index(x) for x in observed_intervention]
    )

    observed_probabilities = (
        probability_matrix[
            np.arange(n),
            observed_action_indices,
        ]
    )

    observed_recovered = (
        rng.random(n)
        < observed_probabilities
    ).astype(np.int8)

    # --------------------------------------------------------
    # 13. Observed / historical expected revenue
    # --------------------------------------------------------

    observed_expected_revenue = (
        observed_probabilities * amounts
    )

    # --------------------------------------------------------
    # 14. Identifiers
    # --------------------------------------------------------

    customer_numbers = customer["customer_num"].to_numpy()
    merchant_numbers = merchant["merchant_num"].to_numpy()

    customer_ids = generate_ids(
        "CUST_",
        customer_numbers,
    )

    merchant_ids = generate_ids(
        "MERCH_",
        merchant_numbers,
    )

    transaction_numbers = np.arange(
        chunk_start,
        chunk_start + n,
    )

    transaction_ids = generate_ids(
        "TXN_",
        transaction_numbers,
        width=12,
    )

    # --------------------------------------------------------
    # 15. Final DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": customer_ids,
            "merchant_id": merchant_ids,

            "timestamp": timestamps,

            "amount": np.round(
                amounts,
                2,
            ),

            "payment_method": payment_methods,
            "failure_reason": failure_reasons,
            "device": devices,
            "location": locations,

            "previous_success_rate": np.round(
                previous_success_rate,
                4,
            ),

            "days_since_last_payment": (
                days_since_last_payment
            ),

            "subscription_age_days": (
                subscription_age_days
            ),

            "historical_retries": (
                historical_retries
            ),

            "time_since_failure_mins": (
                time_since_failure_mins
            ),

            "customer_value": (
                customer["customer_value"].to_numpy()
            ),

            "merchant_category": (
                merchant["merchant_category"].to_numpy()
            ),

            "failure_hour": failure_hour,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,

            "is_weekend": is_weekend.astype(np.int8),
            "is_salary_window": (
                is_salary_window.astype(np.int8)
            ),

            "network_quality": np.round(
                network_quality,
                4,
            ),

            "customer_fatigue": np.round(
                customer_fatigue,
                4,
            ),

            # =============================================
            # Observed historical action/outcome
            # =============================================

            "observed_intervention": (
                observed_intervention
            ),

            "observed_recovery_probability": np.round(
                observed_probabilities,
                4,
            ),

            "observed_recovered": (
                observed_recovered
            ),

            "observed_expected_revenue": np.round(
                observed_expected_revenue,
                2,
            ),

            # =============================================
            # Potential outcomes
            # =============================================

            "p_retry_30m": np.round(
                probabilities["retry_30m"],
                4,
            ),

            "p_retry_evening": np.round(
                probabilities["retry_evening"],
                4,
            ),

            "p_payment_link": np.round(
                probabilities["payment_link"],
                4,
            ),

            "p_whatsapp_reminder": np.round(
                probabilities["whatsapp_reminder"],
                4,
            ),

            "p_alternate_method": np.round(
                probabilities["alternate_method"],
                4,
            ),

            "p_stop": np.round(
                probabilities["stop"],
                4,
            ),

            # =============================================
            # Oracle / ground truth
            # =============================================

            "optimal_intervention": (
                optimal_intervention
            ),

            "oracle_recovery_probability": np.round(
                oracle_recovery_probability,
                4,
            ),

            "oracle_expected_revenue": np.round(
                oracle_expected_revenue,
                2,
            ),
        }
    )

    return df


# ============================================================
# DATASET GENERATOR
# ============================================================

def generate_dataset(
    num_rows: int,
    num_customers: int,
    num_merchants: int,
    output_file: str,
    chunk_size: int,
    seed: int,
) -> None:

    if num_rows <= 0:
        raise ValueError(
            "num_rows must be > 0"
        )

    if num_customers <= 0:
        raise ValueError(
            "num_customers must be > 0"
        )

    if num_merchants <= 0:
        raise ValueError(
            "num_merchants must be > 0"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be > 0"
        )

    rng = np.random.default_rng(seed)

    print("=" * 72)
    print("RecoveryOS V2 Synthetic Causal Dataset Generator")
    print("=" * 72)

    print(f"Rows       : {num_rows:,}")
    print(f"Customers  : {num_customers:,}")
    print(f"Merchants  : {num_merchants:,}")
    print(f"Chunk size : {chunk_size:,}")
    print(f"Seed       : {seed}")
    print(f"Output     : {output_file}")
    print()

    # --------------------------------------------------------
    # Build latent profiles once.
    # --------------------------------------------------------

    profile_start = time.time()

    print("Generating customer profiles...")
    customer_profiles = generate_customer_profiles(
        rng,
        num_customers,
    )

    print("Generating merchant profiles...")
    merchant_profiles = generate_merchant_profiles(
        rng,
        num_merchants,
    )

    print(
        f"Profiles ready in "
        f"{time.time() - profile_start:.2f}s"
    )
    print()

    # --------------------------------------------------------
    # Remove existing output
    # --------------------------------------------------------

    if os.path.exists(output_file):
        print(
            f"Removing existing file: {output_file}"
        )
        os.remove(output_file)

    # --------------------------------------------------------
    # Parquet writer
    # --------------------------------------------------------

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None

    total_generated = 0
    start_time = time.time()

    try:

        while total_generated < num_rows:

            current_chunk = min(
                chunk_size,
                num_rows - total_generated,
            )

            df = generate_chunk(
                rng=rng,
                customer_profiles=customer_profiles,
                merchant_profiles=merchant_profiles,
                n=current_chunk,
                chunk_start=total_generated,
            )

            table = pa.Table.from_pandas(
                df,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression="zstd",
                    compression_level=3,
                )

            writer.write_table(table)

            total_generated += current_chunk

            elapsed = time.time() - start_time
            rows_per_second = (
                total_generated
                / max(elapsed, 1e-9)
            )

            print(
                f"[{total_generated:>12,} / "
                f"{num_rows:,}] "
                f"{total_generated / num_rows * 100:6.2f}% | "
                f"{rows_per_second:,.0f} rows/sec | "
                f"{elapsed / 60:.2f} min"
            )

            del df
            del table

    finally:

        if writer is not None:
            writer.close()

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DATASET GENERATION COMPLETE")
    print("=" * 72)

    print(
        f"Records generated : {total_generated:,}"
    )

    print(
        f"Runtime           : "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Throughput        : "
        f"{total_generated / max(total_time, 1e-9):,.0f} rows/sec"
    )

    print(
        f"Output file       : {output_file}"
    )

    print()

    # --------------------------------------------------------
    # Small sample / sanity check
    # --------------------------------------------------------

    sample = pd.read_parquet(
        output_file,
        columns=[
            "failure_reason",
            "observed_intervention",
            "observed_recovered",
            "optimal_intervention",
            "oracle_recovery_probability",
            "oracle_expected_revenue",
        ],
    ).head(100_000)

    print("Sanity check:")
    print()

    print(
        "Observed recovery rate:",
        f"{sample['observed_recovered'].mean() * 100:.2f}%"
    )

    print(
        "Oracle recovery probability:",
        f"{sample['oracle_recovery_probability'].mean() * 100:.2f}%"
    )

    print()
    print("Observed interventions:")
    print(
        sample["observed_intervention"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

    print()
    print("Oracle interventions:")
    print(
        sample["optimal_intervention"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

    print()
    print("Example rows:")
    print(
        sample.head(10).to_string(
            index=False
        )
    )


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate RecoveryOS V2 synthetic "
            "causal payment recovery data."
        )
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=1_000_000,
        help="Number of payment-failure rows.",
    )

    parser.add_argument(
        "--customers",
        type=int,
        default=200_000,
        help="Number of latent customers.",
    )

    parser.add_argument(
        "--merchants",
        type=int,
        default=10_000,
        help="Number of latent merchants.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows generated per chunk.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "razorpay_recovery_v2.parquet"
        ),
        help="Output Parquet file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    generate_dataset(
        num_rows=args.rows,
        num_customers=args.customers,
        num_merchants=args.merchants,
        output_file=args.output,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()