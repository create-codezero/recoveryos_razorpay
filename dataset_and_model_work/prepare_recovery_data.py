

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import duckdb


ORACLE_COLS = [
    "p_retry_30m",
    "p_retry_evening",
    "p_payment_link",
    "p_whatsapp_reminder",
    "p_alternate_method",
    "p_stop",
    "oracle_recovery_probability",
    "oracle_expected_revenue",
    "optimal_intervention",
]

# These describe what actually happened historically and are useful for
# observational / off-policy evaluation. They should NOT be used as features
# when predicting which action to take before acting.
OBSERVED_COLS = [
    "observed_intervention",
    "observed_recovery_probability",
    "observed_recovered",
    "observed_expected_revenue",
]

# Columns that should not become model features.
NON_FEATURE_COLS = set(
    [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "timestamp",
        "observed_intervention",
        "observed_recovery_probability",
        "observed_recovered",
        "observed_expected_revenue",
    ]
    + ORACLE_COLS
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input V2 parquet file")
    p.add_argument("--outdir", default="recovery_prepared", help="Output directory")
    return p.parse_args()


def q(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchall()


def table(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).df()


def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    started = time.time()

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4") # from 0 to 4
    con.execute("PRAGMA enable_progress_bar=false")

    # DuckDB reads parquet lazily; this avoids loading 20M rows into Python memory.
    source = input_path.as_posix().replace("'", "''")
    con.execute(
        f"CREATE OR REPLACE VIEW recovery AS SELECT * FROM read_parquet('{source}')"
    )

    # ------------------------------------------------------------------
    # 1. Basic audit
    # ------------------------------------------------------------------
    print_header("1. BASIC DATASET AUDIT")

    row_count = q(con, "SELECT COUNT(*) FROM recovery")[0][0]
    customer_count = q(con, "SELECT COUNT(DISTINCT customer_id) FROM recovery")[0][0]
    merchant_count = q(con, "SELECT COUNT(DISTINCT merchant_id) FROM recovery")[0][0]

    print(f"Rows                 : {row_count:,}")
    print(f"Unique customers     : {customer_count:,}")
    print(f"Unique merchants     : {merchant_count:,}")
    print(f"Rows/customer        : {row_count / max(customer_count, 1):.2f}")

    schema_df = table(con, "DESCRIBE recovery")
    print("\nSchema:")
    print(schema_df[["column_name", "column_type"]].to_string(index=False))

    # Null report for all columns.
    null_exprs = []
    for col in schema_df["column_name"].tolist():
        safe = '"' + col.replace('"', '""') + '"'
        null_exprs.append(
            f"SUM(CASE WHEN {safe} IS NULL THEN 1 ELSE 0 END) AS \"{col}\""
        )
    null_sql = "SELECT " + ", ".join(null_exprs) + " FROM recovery"
    null_row = q(con, null_sql)[0]
    null_report = []
    for col, nnull in zip(schema_df["column_name"].tolist(), null_row):
        if nnull:
            null_report.append((col, int(nnull), 100.0 * nnull / row_count))
    if null_report:
        print("\nNULLS:")
        for col, nnull, pct in null_report:
            print(f"  {col:36s} {nnull:12,d} ({pct:.4f}%)")
    else:
        print("\nNULLS: none")

    # ------------------------------------------------------------------
    # 2. Core outcome audit
    # ------------------------------------------------------------------
    print_header("2. RECOVERY / INTERVENTION AUDIT")

    overall = table(
        con,
        """
        SELECT
            AVG(observed_recovered) AS observed_recovery_rate,
            AVG(oracle_recovery_probability) AS oracle_recovery_probability,
            SUM(CASE WHEN observed_recovered = 1 THEN amount ELSE 0 END) AS recovered_gross_amount,
            SUM(amount) AS failed_amount,
            SUM(oracle_expected_revenue) AS oracle_expected_revenue
        FROM recovery
        """,
    )
    print(overall.to_string(index=False))

    print("\nObserved intervention distribution:")
    intervention_df = table(
        con,
        """
        SELECT
            observed_intervention,
            COUNT(*) AS rows,
            AVG(observed_recovered) AS recovery_rate,
            AVG(amount) AS avg_amount,
            SUM(CASE WHEN observed_recovered = 1 THEN amount ELSE 0 END) AS recovered_amount
        FROM recovery
        GROUP BY observed_intervention
        ORDER BY rows DESC
        """,
    )
    print(intervention_df.to_string(index=False))

    print("\nOracle intervention distribution:")
    oracle_df = table(
        con,
        """
        SELECT
            optimal_intervention,
            COUNT(*) AS rows,
            AVG(oracle_recovery_probability) AS oracle_recovery_prob,
            AVG(oracle_expected_revenue) AS oracle_expected_revenue
        FROM recovery
        GROUP BY optimal_intervention
        ORDER BY rows DESC
        """,
    )
    print(oracle_df.to_string(index=False))

    print("\nFailure reason × observed intervention:")
    cross_df = table(
        con,
        """
        SELECT
            failure_reason,
            observed_intervention,
            COUNT(*) AS rows,
            AVG(observed_recovered) AS recovery_rate
        FROM recovery
        GROUP BY failure_reason, observed_intervention
        ORDER BY failure_reason, rows DESC
        """,
    )
    print(cross_df.to_string(index=False))

    print("\nOracle recovery by failure reason:")
    failure_df = table(
        con,
        """
        SELECT
            failure_reason,
            COUNT(*) AS rows,
            AVG(observed_recovered) AS observed_rate,
            AVG(oracle_recovery_probability) AS oracle_rate,
            SUM(oracle_expected_revenue) AS oracle_expected_revenue
        FROM recovery
        GROUP BY failure_reason
        ORDER BY oracle_expected_revenue DESC
        """,
    )
    print(failure_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Dominance / leakage diagnostics
    # ------------------------------------------------------------------
    print_header("3. MODEL-DESIGN DIAGNOSTICS")

    top_oracle = q(
        con,
        """
        SELECT optimal_intervention, COUNT(*) AS n
        FROM recovery
        GROUP BY optimal_intervention
        ORDER BY n DESC
        LIMIT 1
        """,
    )[0]
    top_share = top_oracle[1] / row_count
    print(f"Dominant oracle action : {top_oracle[0]}")
    print(f"Dominant action share  : {top_share * 100:.2f}%")

    if top_share > 0.80:
        print("WARNING: oracle policy is highly concentrated; a naive majority-action baseline may look deceptively strong.")
    elif top_share > 0.70:
        print("CAUTION: oracle policy is noticeably concentrated; evaluate against action-specific baselines and revenue regret.")
    else:
        print("OK: oracle policy has reasonable action diversity.")

    # Predictive leakage hints: these are targets/counterfactuals and must not
    # enter the feature matrix.
    print("\nColumns EXCLUDED from features:")
    for col in sorted(NON_FEATURE_COLS):
        print(f"  - {col}")

    print("\nModel feature candidates:")
    feature_candidates = [c for c in schema_df["column_name"].tolist() if c not in NON_FEATURE_COLS]
    for col in feature_candidates:
        print(f"  + {col}")

    # ------------------------------------------------------------------
    # 4. Dataset-level benchmark before ML
    # ------------------------------------------------------------------
    print_header("4. PRE-MODEL POLICY BENCHMARK")

    benchmark = table(
        con,
        """
        WITH oracle AS (
            SELECT
                SUM(oracle_expected_revenue) AS oracle_value
            FROM recovery
        ),
        baseline AS (
            SELECT
                SUM(observed_expected_revenue) AS observed_policy_value
            FROM recovery
        )
        SELECT
            oracle.oracle_value,
            baseline.observed_policy_value,
            oracle.oracle_value - baseline.observed_policy_value AS oracle_minus_observed,
            CASE
                WHEN oracle.oracle_value = 0 THEN NULL
                ELSE baseline.observed_policy_value / oracle.oracle_value
            END AS observed_policy_efficiency
        FROM oracle, baseline
        """,
    )
    print(benchmark.to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Customer-level deterministic split
    # ------------------------------------------------------------------
    print_header("5. CREATING CUSTOMER-LEVEL TRAIN / VAL / TEST SPLIT")

    print(
        "Split policy: HASH(customer_id) -> 80% train / 10% validation / 10% test."
    )
    print(
        "This prevents transactions from the same customer appearing in multiple splits."
    )

    # DuckDB hash() is deterministic within the query engine and avoids bringing
    # the 20M IDs into Python.
    con.execute(
        """
        CREATE OR REPLACE VIEW recovery_split AS
        SELECT
            *,
            CASE
                WHEN MOD(ABS(HASH(customer_id)), 10) < 8 THEN 'train'
                WHEN MOD(ABS(HASH(customer_id)), 10) = 8 THEN 'validation'
                ELSE 'test'
            END AS dataset_split
        FROM recovery
        """
    )

    split_counts = table(
        con,
        """
        SELECT
            dataset_split,
            COUNT(*) AS rows,
            COUNT(DISTINCT customer_id) AS customers,
            AVG(observed_recovered) AS observed_recovery_rate,
            AVG(oracle_recovery_probability) AS oracle_recovery_probability
        FROM recovery_split
        GROUP BY dataset_split
        ORDER BY CASE dataset_split
            WHEN 'train' THEN 1
            WHEN 'validation' THEN 2
            ELSE 3
        END
        """,
    )
    print(split_counts.to_string(index=False))

    # Verify customer disjointness.
    overlap = q(
        con,
        """
        WITH customers AS (
            SELECT customer_id, COUNT(DISTINCT dataset_split) AS n_splits
            FROM recovery_split
            GROUP BY customer_id
        )
        SELECT COUNT(*) FROM customers WHERE n_splits > 1
        """,
    )[0][0]

    print(f"Customer split overlap : {overlap}")
    if overlap != 0:
        raise RuntimeError("FATAL: customer leakage detected across splits.")

    # ------------------------------------------------------------------
    # 6. Model-ready columns
    # ------------------------------------------------------------------
    print_header("6. WRITING MODEL-READY PARQUET FILES")

    # Derived features are based only on information available BEFORE choosing
    # an intervention. No observed outcome, oracle probability, or optimal action
    # is included.
    model_feature_expr = """
        transaction_id,
        customer_id,
        merchant_id,
        timestamp,
        amount,
        payment_method,
        failure_reason,
        device,
        location,
        previous_success_rate,
        days_since_last_payment,
        subscription_age_days,
        historical_retries,
        time_since_failure_mins,
        customer_value,
        merchant_category,
        failure_hour,
        day_of_week,
        day_of_month,
        is_weekend,
        is_salary_window,
        network_quality,
        customer_fatigue,

        -- Derived time features
        SIN(2 * PI() * failure_hour / 24.0) AS failure_hour_sin,
        COS(2 * PI() * failure_hour / 24.0) AS failure_hour_cos,
        SIN(2 * PI() * day_of_week / 7.0) AS dow_sin,
        COS(2 * PI() * day_of_week / 7.0) AS dow_cos,

        -- Amount transformations
        LN(GREATEST(amount, 1.0)) AS log_amount,
        amount / NULLIF(customer_avg_amount, 0) AS amount_vs_customer_avg,

        -- Context interaction indicators
        CASE WHEN failure_reason = 'insufficient_funds' AND is_salary_window = 1
             THEN 1 ELSE 0 END AS insufficient_funds_salary_window,
        CASE WHEN failure_reason = 'insufficient_funds' AND failure_hour BETWEEN 18 AND 22
             THEN 1 ELSE 0 END AS insufficient_funds_evening,
        CASE WHEN failure_reason = 'expired_card' AND payment_method IN ('Credit Card', 'Debit Card')
             THEN 1 ELSE 0 END AS expired_card_payment_method_match,
        CASE WHEN failure_reason = 'bank_timeout' AND time_since_failure_mins <= 60
             THEN 1 ELSE 0 END AS recent_bank_timeout,
        CASE WHEN historical_retries >= 3 THEN 1 ELSE 0 END AS high_retry_fatigue,

        -- Latent profile feature from the synthetic customer table.
        customer_avg_amount,
        salary_sensitive,
        contact_tolerance,
        preferred_hour,
        value_score AS customer_latent_value_score
    """

    # The above needs customer profile fields (customer_avg_amount, salary_sensitive,
    # etc.). In V2 they are not in the raw transaction file, so reconstruct only the
    # fields that are present indirectly is impossible. To avoid inventing columns,
    # create the model dataset using raw transaction columns plus safe derived fields.
    model_feature_expr = """
        transaction_id,
        customer_id,
        merchant_id,
        timestamp,
        amount,
        payment_method,
        failure_reason,
        device,
        location,
        previous_success_rate,
        days_since_last_payment,
        subscription_age_days,
        historical_retries,
        time_since_failure_mins,
        customer_value,
        merchant_category,
        failure_hour,
        day_of_week,
        day_of_month,
        is_weekend,
        is_salary_window,
        network_quality,
        customer_fatigue,

        SIN(2 * PI() * failure_hour / 24.0) AS failure_hour_sin,
        COS(2 * PI() * failure_hour / 24.0) AS failure_hour_cos,
        SIN(2 * PI() * day_of_week / 7.0) AS dow_sin,
        COS(2 * PI() * day_of_week / 7.0) AS dow_cos,
        LN(GREATEST(amount, 1.0)) AS log_amount,
        CASE WHEN failure_reason = 'insufficient_funds' AND is_salary_window = 1
             THEN 1 ELSE 0 END AS insufficient_funds_salary_window,
        CASE WHEN failure_reason = 'insufficient_funds' AND failure_hour BETWEEN 18 AND 22
             THEN 1 ELSE 0 END AS insufficient_funds_evening,
        CASE WHEN failure_reason = 'expired_card' AND payment_method IN ('Credit Card', 'Debit Card')
             THEN 1 ELSE 0 END AS expired_card_payment_method_match,
        CASE WHEN failure_reason = 'bank_timeout' AND time_since_failure_mins <= 60
             THEN 1 ELSE 0 END AS recent_bank_timeout,
        CASE WHEN historical_retries >= 3 THEN 1 ELSE 0 END AS high_retry_fatigue
    """

    # Files for learning to choose an action from observed history.
    # Keep observed action/outcome as labels/evaluation fields, but never include
    # oracle counterfactuals in the model feature matrix.
    output_columns_sql = f"""
        SELECT
            {model_feature_expr},
            observed_intervention,
            observed_recovered,
            observed_recovery_probability,
            observed_expected_revenue,
            optimal_intervention,
            oracle_recovery_probability,
            oracle_expected_revenue,
            p_retry_30m,
            p_retry_evening,
            p_payment_link,
            p_whatsapp_reminder,
            p_alternate_method,
            p_stop
        FROM recovery_split
        WHERE dataset_split = ?
    """

    # The output is evaluation-ready rather than an encoded numeric matrix.
    # Encoding comes later after model selection.
    for split_name in ("train", "validation", "test"):
        out_file = outdir / f"recovery_{split_name}.parquet"
        if out_file.exists():
            out_file.unlink()

        sql = output_columns_sql.replace("?", f"'{split_name}'")
        print(f"Writing {split_name}: {out_file.name}")
        con.execute(
            f"COPY ({sql}) TO '{out_file.as_posix().replace(chr(39), chr(39)+chr(39))}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )

    # Save audit artifacts.
    audit = {
        "input_file": str(input_path),
        "rows": int(row_count),
        "unique_customers": int(customer_count),
        "unique_merchants": int(merchant_count),
        "dominant_oracle_action": str(top_oracle[0]),
        "dominant_oracle_action_share": float(top_share),
        "customer_split_overlap": int(overlap),
        "non_feature_columns": sorted(NON_FEATURE_COLS),
        "generated_seconds": time.time() - started,
    }

    with open(outdir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    intervention_df.to_csv(outdir / "observed_intervention_summary.csv", index=False)
    oracle_df.to_csv(outdir / "oracle_intervention_summary.csv", index=False)
    failure_df.to_csv(outdir / "failure_reason_summary.csv", index=False)
    split_counts.to_csv(outdir / "split_summary.csv", index=False)

    # Feature manifest.
    manifest = {
        "safe_model_features": [
            c for c in schema_df["column_name"].tolist()
            if c not in NON_FEATURE_COLS
        ] + [
            "failure_hour_sin",
            "failure_hour_cos",
            "dow_sin",
            "dow_cos",
            "log_amount",
            "insufficient_funds_salary_window",
            "insufficient_funds_evening",
            "expired_card_payment_method_match",
            "recent_bank_timeout",
            "high_retry_fatigue",
        ],
        "observed_labels_or_eval_fields": OBSERVED_COLS,
        "oracle_fields_for_benchmark_only": ORACLE_COLS,
    }
    with open(outdir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - started
    print_header("DONE")
    print(f"Prepared data directory : {outdir}")
    print(f"Runtime                  : {elapsed / 60:.2f} minutes")
    print("\nFiles:")
    for p in sorted(outdir.iterdir()):
        size_mb = p.stat().st_size / (1024**2)
        print(f"  {p.name:40s} {size_mb:10.1f} MB")

    print("\nNEXT STEP")
    print("Train a recovery-probability model with observed_intervention as the treatment/action label,")
    print("then build an action-ranking policy and evaluate it against the preserved oracle columns.")


if __name__ == "__main__":
    main()
