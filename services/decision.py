

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import shap
except ImportError:
    shap = None


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("recoveryos.decision")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | RecoveryOS | %(levelname)s | %(message)s",
    )


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_VERSION = "catboost-recovery-v1"

POLICY_VERSION = "recovery-policy-v1"

MODEL_PATH = os.getenv(
    "RECOVERY_MODEL_PATH",
    str(
        Path(__file__).resolve().parent.parent
        / "model"
        / "catboost_recovery_laptop.cbm"
    ),
)

# ============================================================================
# EXACT TRAINING FEATURE CONTRACT
# ============================================================================

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
    feature
    for feature in MODEL_FEATURES
    if feature not in CATEGORICAL_FEATURES
]


# ============================================================================
# RECOVERY ACTIONS
# ============================================================================

ACTIONS = [
    "retry_30m",
    "retry_evening",
    "payment_link",
    "whatsapp_reminder",
    "alternate_method",
    "stop",
]


ACTION_LABELS = {
    "retry_30m": "Retry in 30 min",
    "retry_evening": "Retry Evening",
    "payment_link": "Payment Link",
    "whatsapp_reminder": "WhatsApp Reminder",
    "alternate_method": "Alternate Method",
    "stop": "Stop",
}


# ============================================================================
# ACTION COSTS
# ============================================================================
#
# Currently zero because your benchmark uses zero intervention costs.
#
# Keeping this configurable means you can later demonstrate:
#
# Expected Value =
#     P(recovery) * amount - action_cost
#
# without changing the decision architecture.
# ============================================================================

ACTION_COSTS = {
    "retry_30m": 0.0,
    "retry_evening": 0.0,
    "payment_link": 0.0,
    "whatsapp_reminder": 0.0,
    "alternate_method": 0.0,
    "stop": 0.0,
}


# ============================================================================
# GUARDRAIL THRESHOLDS
# ============================================================================

MAX_RETRIES = 4

MAX_CUSTOMER_FATIGUE = 0.85

MIN_EXPECTED_REVENUE = 0.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        if value is None:
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def clamp_probability(value: float) -> float:
    """Keep model probability inside [0, 1]."""
    return float(np.clip(safe_float(value), 0.0, 1.0))


def format_action(action: str) -> str:
    """Convert internal action name to dashboard-friendly label."""
    return ACTION_LABELS.get(
        action,
        action.replace("_", " ").title(),
    )


def money(value: float) -> float:
    """Round INR monetary values."""
    return round(safe_float(value), 2)


def generate_decision_id(transaction_id: str) -> str:
    """
    Generate deterministic decision ID.

    Same transaction → same decision ID.
    Useful for audit trails and demo reproducibility.
    """
    digest = hashlib.sha256(
        transaction_id.encode("utf-8")
    ).hexdigest()[:16]

    return f"DEC_{digest.upper()}"


# ============================================================================
# DETERMINISTIC OUTCOME SIMULATION
# ============================================================================

def simulate_outcome(
    transaction_id: str,
    final_action: str,
    probability: float,
    amount: float,
) -> Dict[str, Any]:
    """
    Simulate an execution outcome deterministically.

    Why deterministic?

    A random outcome would change every time the dashboard/API
    is refreshed. That makes demos inconsistent.

    Transaction ID acts as the deterministic seed.

    STOP is never treated as a recovery attempt.
    """

    probability = clamp_probability(probability)
    amount = max(0.0, safe_float(amount))

    if final_action == "stop":
        return {
            "execution_status": "NOT_ATTEMPTED",
            "outcome_status": "NOT_ATTEMPTED",
            "recovered_amount": 0.0,
            "simulation_score": None,
            "feedback_status": "NOT_APPLICABLE",
            "outcome_timestamp": utc_now(),
        }

    digest = hashlib.sha256(
        transaction_id.encode("utf-8")
    ).hexdigest()

    seed = int(digest[:8], 16)

    simulation_score = (seed % 10000) / 10000.0

    recovered = simulation_score < probability

    return {
        "execution_status": "EXECUTED",
        "outcome_status": (
            "RECOVERED"
            if recovered
            else "FAILED"
        ),
        "recovered_amount": (
            money(amount)
            if recovered
            else 0.0
        ),
        "simulation_score": round(
            simulation_score,
            4,
        ),
        "feedback_status": "QUEUED_FOR_REVIEW",
        "outcome_timestamp": utc_now(),
    }


# ============================================================================
# DECISION ENGINE
# ============================================================================

class RecoveryDecisionEngine:
    """
    Main RecoveryOS AI decision engine.

    Responsibilities:

        1. Load CatBoost model
        2. Validate incoming payment context
        3. Evaluate all six interventions
        4. Calculate expected recovery value
        5. Select AI proposal
        6. Apply guardrails
        7. Produce final executable action
        8. Generate SHAP explanation
        9. Simulate deterministic outcome
        10. Return audit-ready decision object
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        enable_shap: bool = True,
    ):

        self.model_path = model_path or MODEL_PATH

        self.enable_shap = (
            enable_shap
            and shap is not None
        )

        self.model = None
        self.explainer = None

        self._load_model()

        if self.enable_shap:
            self._load_shap()

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load CatBoost model."""

        if CatBoostClassifier is None:
            raise RuntimeError(
                "CatBoost is not installed. "
                "Install with: pip install catboost"
            )

        model_file = Path(self.model_path)

        if not model_file.exists():
            raise FileNotFoundError(
                f"RecoveryOS model not found:\n"
                f"{model_file}\n\n"
                f"Set RECOVERY_MODEL_PATH if the model is elsewhere."
            )

        logger.info(
            "Loading RecoveryOS model: %s",
            model_file,
        )

        self.model = CatBoostClassifier()

        self.model.load_model(
            str(model_file)
        )

        logger.info(
            "Model loaded successfully | version=%s",
            MODEL_VERSION,
        )

    # ------------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------------

    def _load_shap(self) -> None:
        """Create SHAP TreeExplainer."""

        try:
            self.explainer = shap.TreeExplainer(
                self.model
            )

            logger.info(
                "SHAP TreeExplainer initialized"
            )

        except Exception as exc:
            logger.warning(
                "SHAP initialization failed: %s",
                exc,
            )

            self.explainer = None
            self.enable_shap = False

    # ------------------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------------------

    def _validate_context(
        self,
        payment_context: Dict[str, Any],
    ) -> None:
        """
        Validate required observable context.

        Transaction/customer/merchant IDs are not model features,
        but transaction_id is required for audit/outcome simulation.
        """

        missing = [
            feature
            for feature in MODEL_FEATURES
            if feature != "observed_intervention"
            and feature not in payment_context
        ]

        if missing:
            raise ValueError(
                "Missing required payment context fields: "
                + ", ".join(missing)
            )

        if "transaction_id" not in payment_context:
            raise ValueError(
                "transaction_id is required for "
                "RecoveryOS decision auditing."
            )

    # ------------------------------------------------------------------------
    # DATAFRAME PREPARATION
    # ------------------------------------------------------------------------

    def _prepare_dataframe(
        self,
        payment_context: Dict[str, Any],
        action: str,
    ) -> pd.DataFrame:
        """
        Create one model row for a candidate action.

        observed_intervention is intentionally overwritten with
        the candidate action because the trained model is
        action-conditioned.
        """

        if action not in ACTIONS:
            raise ValueError(
                f"Unsupported action: {action}"
            )

        row = {}

        for feature in MODEL_FEATURES:

            if feature == "observed_intervention":
                row[feature] = action
                continue

            value = payment_context.get(
                feature
            )

            if feature in CATEGORICAL_FEATURES:
                if value is None:
                    value = "__MISSING__"

                row[feature] = str(value)

            else:
                row[feature] = safe_float(value)

        df = pd.DataFrame(
            [row],
            columns=MODEL_FEATURES,
        )

        # CatBoost categorical columns must be strings.
        for feature in CATEGORICAL_FEATURES:
            df[feature] = (
                df[feature]
                .fillna("__MISSING__")
                .astype(str)
            )

        return df

    # ------------------------------------------------------------------------
    # SINGLE ACTION PREDICTION
    # ------------------------------------------------------------------------

    def _predict_action(
        self,
        payment_context: Dict[str, Any],
        action: str,
    ) -> float:
        """Predict P(recovery | context, action)."""

        X = self._prepare_dataframe(
            payment_context,
            action,
        )

        probability = self.model.predict_proba(
            X
        )[0, 1]

        return clamp_probability(
            probability
        )

    # ------------------------------------------------------------------------
    # EXPECTED REVENUE
    # ------------------------------------------------------------------------

    def _expected_revenue(
        self,
        probability: float,
        amount: float,
        action: str,
    ) -> float:
        """
        Expected Recovery Value.

            EV(action)
              = P(recovery | action) * amount
                - action_cost
        """

        probability = clamp_probability(
            probability
        )

        amount = max(
            0.0,
            safe_float(amount),
        )

        cost = ACTION_COSTS.get(
            action,
            0.0,
        )

        return money(
            probability * amount - cost
        )

    # ------------------------------------------------------------------------
    # EVALUATE ALL ACTIONS
    # ------------------------------------------------------------------------

    def evaluate_actions(
        self,
        payment_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate every candidate intervention.

        Returns actions sorted by expected recovery value.
        """

        amount = max(
            0.0,
            safe_float(
                payment_context["amount"]
            ),
        )

        results = []

        for action in ACTIONS:

            probability = self._predict_action(
                payment_context,
                action,
            )

            expected_revenue = (
                self._expected_revenue(
                    probability,
                    amount,
                    action,
                )
            )

            results.append(
                {
                    "action": action,
                    "label": format_action(
                        action
                    ),
                    "recovery_probability": round(
                        probability,
                        4,
                    ),
                    "recovery_probability_pct": round(
                        probability * 100,
                        2,
                    ),
                    "expected_revenue": expected_revenue,
                    "action_cost": ACTION_COSTS.get(
                        action,
                        0.0,
                    ),
                }
            )

        results.sort(
            key=lambda item: (
                item["expected_revenue"],
                item["recovery_probability"],
            ),
            reverse=True,
        )

        # Rank after sorting.
        for rank, item in enumerate(
            results,
            start=1,
        ):
            item["rank"] = rank

        return results

    # ------------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------------

    def _extract_shap_values(
        self,
        shap_values: Any,
    ) -> np.ndarray:
        """
        Normalize SHAP output across SHAP versions.

        CatBoost/SHAP can return:
            list
            2D ndarray
            3D ndarray
        """

        values = shap_values

        if isinstance(values, list):
            values = values[-1]

        values = np.asarray(values)

        if values.ndim == 3:
            values = values[0, :, -1]

        elif values.ndim == 2:
            values = values[0]

        elif values.ndim != 1:
            raise ValueError(
                f"Unsupported SHAP shape: "
                f"{values.shape}"
            )

        return values.astype(float)

    def explain_action(
        self,
        payment_context: Dict[str, Any],
        action: str,
    ) -> Dict[str, Any]:
        """
        Explain the selected action using SHAP.

        SHAP values are model-output contributions
        (typically log-odds for this binary classifier),
        NOT percentage-point changes in probability.
        """

        empty = {
            "positive_drivers": [],
            "negative_drivers": [],
        }

        if not self.enable_shap:
            return empty

        if self.explainer is None:
            return empty

        try:
            X = self._prepare_dataframe(
                payment_context,
                action,
            )

            raw_values = self.explainer.shap_values(
                X
            )

            values = self._extract_shap_values(
                raw_values
            )

            feature_names = list(
                X.columns
            )

            contributions = list(
                zip(
                    feature_names,
                    values,
                )
            )

            # observed_intervention is the selected
            # treatment itself, so excluding it from
            # the explanation makes the dashboard easier
            # to interpret.
            contributions = [
                (feature, float(value))
                for feature, value in contributions
                if feature
                != "observed_intervention"
            ]

            positive = sorted(
                [
                    (feature, value)
                    for feature, value
                    in contributions
                    if value > 0
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            negative = sorted(
                [
                    (feature, value)
                    for feature, value
                    in contributions
                    if value < 0
                ],
                key=lambda x: x[1],
            )

            return {
                "positive_drivers": [
                    {
                        "feature": feature,
                        "value": round(
                            value,
                            4,
                        ),
                        "direction": "positive",
                    }
                    for feature, value
                    in positive[:3]
                ],
                "negative_drivers": [
                    {
                        "feature": feature,
                        "value": round(
                            value,
                            4,
                        ),
                        "direction": "negative",
                    }
                    for feature, value
                    in negative[:2]
                ],
            }

        except Exception as exc:

            logger.warning(
                "SHAP explanation failed: %s",
                exc,
            )

            return empty

    # ------------------------------------------------------------------------
    # GUARDRAILS
    # ------------------------------------------------------------------------

    def apply_guardrails(
        self,
        payment_context: Dict[str, Any],
        ai_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Apply policy constraints sequentially.

        IMPORTANT:
        Each rule operates on the CURRENT final_action.

        This prevents the classic bug where rule #3 still sees
        the original AI action even after rule #1 has overridden it.
        """

        if not ai_results:
            raise ValueError(
                "ai_results cannot be empty."
            )

        original = ai_results[0]

        original_action = original[
            "action"
        ]

        final_action = original_action

        guardrail_flags: List[str] = []

        retries = int(
            safe_float(
                payment_context.get(
                    "historical_retries",
                    0,
                )
            )
        )

        fatigue = safe_float(
            payment_context.get(
                "customer_fatigue",
                0,
            )
        )

        failure_reason = str(
            payment_context.get(
                "failure_reason",
                "",
            )
        )

        # ------------------------------------------------------------
        # RULE 1
        # Too many retries
        # ------------------------------------------------------------

        if (
            retries >= MAX_RETRIES
            and final_action
            in {
                "retry_30m",
                "retry_evening",
            }
        ):

            final_action = "alternate_method"

            guardrail_flags.append(
                "MAX_RETRIES_EXCEEDED"
            )

        # ------------------------------------------------------------
        # RULE 2
        # Customer fatigue
        # ------------------------------------------------------------

        if (
            fatigue >= MAX_CUSTOMER_FATIGUE
            and final_action
            in {
                "whatsapp_reminder",
                "payment_link",
            }
        ):

            final_action = "stop"

            guardrail_flags.append(
                "HIGH_CUSTOMER_FATIGUE"
            )

        # ------------------------------------------------------------
        # RULE 3
        # Invalid retry for expired card / limit
        # ------------------------------------------------------------

        if (
            failure_reason
            in {
                "expired_card",
                "limit_exceeded",
            }
            and final_action
            in {
                "retry_30m",
                "retry_evening",
            }
        ):

            final_action = "alternate_method"

            guardrail_flags.append(
                "RETRY_INELIGIBLE_FOR_FAILURE_REASON"
            )

        # ------------------------------------------------------------
        # RULE 4
        # Economic sanity check
        # ------------------------------------------------------------

        final_data = next(
            (
                item
                for item in ai_results
                if item["action"]
                == final_action
            ),
            None,
        )

        if final_data is None:
            raise RuntimeError(
                f"No AI evaluation found for "
                f"final action: {final_action}"
            )

        if (
            final_data["expected_revenue"]
            <= MIN_EXPECTED_REVENUE
            and final_action != "stop"
        ):

            final_action = "stop"

            guardrail_flags.append(
                "NON_POSITIVE_EXPECTED_VALUE"
            )

        # ------------------------------------------------------------
        # Final action lookup
        # ------------------------------------------------------------

        final_data = next(
            item
            for item in ai_results
            if item["action"]
            == final_action
        )

        overridden = (
            final_action
            != original_action
        )

        return {
            "original_action": original_action,
            "final_action": final_action,
            "action_overridden": overridden,
            "guardrail_flags": guardrail_flags,
            "guardrails_passed": not overridden,
            "final_probability": final_data[
                "recovery_probability"
            ],
            "final_expected_revenue": final_data[
                "expected_revenue"
            ],
        }

    # ------------------------------------------------------------------------
    # DECISION MARGIN
    # ------------------------------------------------------------------------

    def calculate_decision_margin(
        self,
        ai_results: List[Dict[str, Any]],
    ) -> float:
        """
        Difference between best and second-best
        recovery probability.

        This is a decision margin, NOT calibrated confidence.
        """

        if len(ai_results) < 2:
            return 0.0

        best = ai_results[0][
            "recovery_probability"
        ]

        second = ai_results[1][
            "recovery_probability"
        ]

        return round(
            best - second,
            4,
        )

    # ------------------------------------------------------------------------
    # MAIN DECISION
    # ------------------------------------------------------------------------

    def decide(
        self,
        payment_context: Dict[str, Any],
        simulate: bool = True,
    ) -> Dict[str, Any]:
        """
        Full RecoveryOS decision pipeline.
        """

        self._validate_context(
            payment_context
        )

        transaction_id = str(
            payment_context[
                "transaction_id"
            ]
        )

        amount = max(
            0.0,
            safe_float(
                payment_context["amount"]
            ),
        )

        decision_id = generate_decision_id(
            transaction_id
        )

        # ================================================================
        # STEP 1 — AI ACTION EVALUATION
        # ================================================================

        ai_results = self.evaluate_actions(
            payment_context
        )

        ai_proposal = ai_results[0]

        # ================================================================
        # STEP 2 — POLICY / GUARDRAILS
        # ================================================================

        policy_result = (
            self.apply_guardrails(
                payment_context,
                ai_results,
            )
        )

        final_action = policy_result[
            "final_action"
        ]

        final_probability = policy_result[
            "final_probability"
        ]

        final_expected_revenue = (
            policy_result[
                "final_expected_revenue"
            ]
        )

        # ================================================================
        # STEP 3 — MODEL EXPLANATION
        # ================================================================

        explanation = self.explain_action(
            payment_context,
            final_action,
        )

        # ================================================================
        # STEP 4 — DECISION MARGIN
        # ================================================================

        decision_margin = (
            self.calculate_decision_margin(
                ai_results
            )
        )

        # ================================================================
        # STEP 5 — OUTCOME SIMULATION
        # ================================================================

        outcome = {
            "execution_status": "NOT_EXECUTED",
            "outcome_status": "PENDING",
            "recovered_amount": 0.0,
            "simulation_score": None,
            "feedback_status": "PENDING",
            "outcome_timestamp": None,
        }

        if simulate:

            outcome = simulate_outcome(
                transaction_id=transaction_id,
                final_action=final_action,
                probability=final_probability,
                amount=amount,
            )

        # ================================================================
        # STEP 6 — HUMAN-READABLE REASON
        # ================================================================

        if policy_result[
            "action_overridden"
        ]:

            reason = (
                f"AI proposed "
                f"{format_action(ai_proposal['action'])}, "
                f"but policy enforced "
                f"{format_action(final_action)} "
                f"because of "
                f"{', '.join(policy_result['guardrail_flags'])}."
            )

        else:

            reason = (
                f"AI selected "
                f"{format_action(final_action)} "
                f"as the highest expected-recovery action."
            )

        # ================================================================
        # STEP 7 — FINAL AUDIT OBJECT
        # ================================================================

        decision = {
            # ------------------------------------------------------------
            # Identity
            # ------------------------------------------------------------

            "decision_id": decision_id,

            "transaction_id": transaction_id,

            "timestamp": utc_now(),

            # ------------------------------------------------------------
            # Model metadata
            # ------------------------------------------------------------

            "model_version": MODEL_VERSION,

            "policy_version": POLICY_VERSION,

            # ------------------------------------------------------------
            # Transaction
            # ------------------------------------------------------------

            "amount": money(amount),

            # ------------------------------------------------------------
            # AI proposal
            # ------------------------------------------------------------

            "ai_proposal": {
                "action": ai_proposal[
                    "action"
                ],
                "label": ai_proposal[
                    "label"
                ],
                "recovery_probability": ai_proposal[
                    "recovery_probability"
                ],
                "expected_revenue": ai_proposal[
                    "expected_revenue"
                ],
            },

            # ------------------------------------------------------------
            # Policy result
            # ------------------------------------------------------------

            "policy": {
                "original_action": policy_result[
                    "original_action"
                ],
                "final_action": final_action,
                "action_overridden": policy_result[
                    "action_overridden"
                ],
                "guardrails_passed": policy_result[
                    "guardrails_passed"
                ],
                "guardrail_flags": policy_result[
                    "guardrail_flags"
                ],
            },

            # ------------------------------------------------------------
            # Final decision
            # ------------------------------------------------------------

            "selected_action": final_action,

            "selected_action_label": format_action(
                final_action
            ),

            "recovery_probability": round(
                final_probability,
                4,
            ),

            "recovery_probability_pct": round(
                final_probability * 100,
                2,
            ),

            "expected_revenue": final_expected_revenue,

            "decision_margin": decision_margin,

            # ------------------------------------------------------------
            # Explanation
            # ------------------------------------------------------------

            "explanation": explanation,

            "reason": reason,

            # ------------------------------------------------------------
            # All counterfactual action evaluations
            # ------------------------------------------------------------

            "action_evaluations": ai_results,

            # ------------------------------------------------------------
            # Outcome / feedback
            # ------------------------------------------------------------

            "outcome": outcome,

            # ------------------------------------------------------------
            # Convenience fields for Firestore/dashboard
            # ------------------------------------------------------------

            "execution_status": outcome[
                "execution_status"
            ],

            "outcome_status": outcome[
                "outcome_status"
            ],

            "recovered_amount": outcome[
                "recovered_amount"
            ],

            "feedback_status": outcome[
                "feedback_status"
            ],

            "execution_mode": (
                "SIMULATION"
                if simulate
                else "DECISION_ONLY"
            ),
        }

        logger.info(
            "Decision complete | txn=%s | "
            "AI=%s | final=%s | "
            "override=%s | EV=₹%.2f",
            transaction_id,
            ai_proposal["action"],
            final_action,
            policy_result[
                "action_overridden"
            ],
            final_expected_revenue,
        )

        return decision


# ============================================================================
# SIMPLE FUNCTION API
# ============================================================================

_engine: Optional[
    RecoveryDecisionEngine
] = None


def get_engine() -> RecoveryDecisionEngine:
    """
    Singleton access to the decision engine.

    The CatBoost model is loaded once per Flask process
    instead of once per API request.
    """

    global _engine

    if _engine is None:
        _engine = RecoveryDecisionEngine()

    return _engine


def make_decision(
    payment_context: Dict[str, Any],
    simulate: bool = True,
) -> Dict[str, Any]:
    """
    Convenience API.

    Example:

        decision = make_decision({
            "transaction_id": "TXN_001",
            "amount": 3500,
            ...
        })

    """

    return get_engine().decide(
        payment_context,
        simulate=simulate,
    )


# ============================================================================
# LOCAL TEST
# ============================================================================

if __name__ == "__main__":

    test_payment = {
        "transaction_id": "TXN_TEST_5389",

        "amount": 3500.0,

        "payment_method": "UPI",

        "failure_reason": "bank_timeout",

        "device": "Android",

        "location": "Mumbai, MH",

        "previous_success_rate": 0.82,

        "days_since_last_payment": 12,

        "subscription_age_days": 180,

        "historical_retries": 1,

        "time_since_failure_mins": 25,

        "customer_value": "High",

        "merchant_category": "E-commerce",

        "failure_hour": 14,

        "day_of_week": 3,

        "day_of_month": 4,

        "is_weekend": 0,

        "is_salary_window": 1,

        "network_quality": 0.85,

        "customer_fatigue": 0.20,
    }

    engine = RecoveryDecisionEngine()

    result = engine.decide(
        test_payment,
        simulate=True,
    )

    print("\n" + "=" * 80)
    print("RECOVERYOS DECISION")
    print("=" * 80)

    print(
        f"Transaction      : "
        f"{result['transaction_id']}"
    )

    print(
        f"AI Proposal      : "
        f"{result['ai_proposal']['label']}"
    )

    print(
        f"Final Action     : "
        f"{result['selected_action_label']}"
    )

    print(
        f"Recovery Prob.   : "
        f"{result['recovery_probability_pct']:.2f}%"
    )

    print(
        f"Expected Revenue : "
        f"₹{result['expected_revenue']:,.2f}"
    )

    print(
        f"Override          : "
        f"{result['policy']['action_overridden']}"
    )

    print(
        f"Outcome           : "
        f"{result['outcome_status']}"
    )

    print(
        f"Recovered Amount  : "
        f"₹{result['recovered_amount']:,.2f}"
    )

    print(
        f"Feedback          : "
        f"{result['feedback_status']}"
    )

    print(
        f"Reason            : "
        f"{result['reason']}"
    )

    print("\nCOUNTERFACTUAL ACTIONS")
    print("-" * 80)

    for item in result[
        "action_evaluations"
    ]:

        print(
            f"{item['label']:22s} "
            f"{item['recovery_probability_pct']:6.2f}% "
            f"₹{item['expected_revenue']:,.2f}"
        )

    print("\nSHAP POSITIVE DRIVERS")
    print("-" * 80)

    for item in result[
        "explanation"
    ]["positive_drivers"]:

        print(
            f"{item['feature']:30s} "
            f"{item['value']:+.4f}"
        )

    print("\nSHAP NEGATIVE DRIVERS")
    print("-" * 80)

    for item in result[
        "explanation"
    ]["negative_drivers"]:

        print(
            f"{item['feature']:30s} "
            f"{item['value']:+.4f}"
        )

    print("=" * 80)
