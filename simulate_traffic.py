import random
import time
import uuid
from datetime import datetime

import requests


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:5000/api/payment/failed"

DEFAULT_INTERVAL_SECONDS = 4
DEFAULT_GUARDRAIL_CHANCE = 0.25
REQUEST_TIMEOUT_SECONDS = 8

# Amount range for simulated failed payments
MIN_AMOUNT = 500
MAX_AMOUNT = 5000


# =========================================================
# STATIC SIMULATION DATA
# =========================================================

PAYMENT_METHODS = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Netbanking",
]

NORMAL_FAILURE_REASONS = [
    "insufficient_funds",
    "bank_timeout",
    "network_error",
]

DEVICES = [
    "Android",
    "iOS",
    "Desktop",
]

LOCATIONS = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
]

CUSTOMER_VALUES = [
    "High",
    "Medium",
    "Low",
]

MERCHANT_CATEGORIES = [
    "SaaS",
    "E-commerce",
    "Digital Goods",
]


# =========================================================
# GUARDRAIL SCENARIOS
# =========================================================

GUARDRAIL_SCENARIOS = [
    {
        "name": "Circuit Breaker",
        "description": "Maximum retry limit exceeded",
        "overrides": {
            "historical_retries": 5,
            "failure_reason": "bank_timeout",
        },
    },
    {
        "name": "Fatigue Suppression",
        "description": "Customer fatigue threshold exceeded",
        "overrides": {
            "customer_fatigue": 0.95,
            "failure_reason": "authentication_failed",
        },
    },
    {
        "name": "Hard Failure",
        "description": "Expired card blocks retry",
        "overrides": {
            "failure_reason": "expired_card",
            "historical_retries": 1,
        },
    },
    {
        "name": "Payment Limit",
        "description": "Payment limit exceeded",
        "overrides": {
            "failure_reason": "limit_exceeded",
            "historical_retries": 1,
        },
    },
]


# =========================================================
# HTTP SESSION
# =========================================================

session = requests.Session()

session.headers.update(
    {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
)


# =========================================================
# TRANSACTION ID
# =========================================================

def generate_transaction_id():
    """
    Generates a unique transaction ID.

    UUID avoids collisions that can happen with randint()
    when the simulator runs continuously.
    """
    return f"TXN_SIM_{uuid.uuid4().hex[:10].upper()}"


def generate_customer_id():
    """
    Generates a simulated customer ID.
    """
    return f"CUST_SIM_{random.randint(10000, 99999)}"


# =========================================================
# BASE PAYLOAD
# =========================================================

def generate_base_payload():
    """
    Generates realistic normal failed-payment traffic.

    Values intentionally stay within the feature ranges
    used by the RecoveryOS model.
    """

    failure_hour = random.randint(0, 23)
    day_of_week = random.randint(0, 6)

    return {
        "transaction_id": generate_transaction_id(),
        "customer_id": generate_customer_id(),

        "amount": round(
            random.uniform(MIN_AMOUNT, MAX_AMOUNT),
            2,
        ),

        "payment_method": random.choice(PAYMENT_METHODS),

        "failure_reason": random.choice(
            NORMAL_FAILURE_REASONS
        ),

        "device": random.choice(DEVICES),

        "location": random.choice(LOCATIONS),

        "previous_success_rate": round(
            random.uniform(0.50, 1.00),
            2,
        ),

        "days_since_last_payment": random.randint(
            1,
            60,
        ),

        "subscription_age_days": random.randint(
            10,
            365,
        ),

        "historical_retries": random.randint(
            0,
            2,
        ),

        "time_since_failure_mins": random.randint(
            1,
            30,
        ),

        "customer_value": random.choice(
            CUSTOMER_VALUES
        ),

        "merchant_category": random.choice(
            MERCHANT_CATEGORIES
        ),

        "failure_hour": failure_hour,

        "day_of_week": day_of_week,

        "day_of_month": random.randint(
            1,
            28,
        ),

        "is_weekend": int(
            day_of_week >= 5
        ),

        "is_salary_window": random.choice(
            [0, 1]
        ),

        "network_quality": round(
            random.uniform(0.50, 1.00),
            2,
        ),

        "customer_fatigue": round(
            random.uniform(0.00, 0.50),
            2,
        ),
    }


# =========================================================
# SCENARIO GENERATOR
# =========================================================

def generate_payload(guardrail_chance):
    """
    Generates either normal traffic or a deliberate
    guardrail edge case.
    """

    payload = generate_base_payload()

    is_edge_case = (
        random.random() < guardrail_chance
    )

    if not is_edge_case:
        return payload, "Normal Traffic"

    scenario = random.choice(
        GUARDRAIL_SCENARIOS
    )

    payload.update(
        scenario["overrides"]
    )

    scenario_label = (
        f"EDGE CASE: {scenario['name']}"
    )

    return payload, scenario_label


# =========================================================
# RESPONSE EXTRACTION
# =========================================================

def extract_decision(response_data):
    """
    Safely extracts the decision object returned by Flask.
    """

    if not isinstance(response_data, dict):
        return {}

    decision = response_data.get(
        "decision",
        {},
    )

    if not isinstance(decision, dict):
        return {}

    return decision


# =========================================================
# TERMINAL DISPLAY
# =========================================================

def print_request_summary(
    count,
    scenario,
    payload,
):
    timestamp = datetime.now().strftime(
        "%H:%M:%S"
    )

    print(
        f"\n[{timestamp}] "
        f"#{count:04d} | {scenario}"
    )

    print(
        f"   TXN       : {payload['transaction_id']}"
    )

    print(
        f"   Amount    : ₹{payload['amount']:,.2f}"
    )

    print(
        f"   Failure   : {payload['failure_reason']}"
    )

    print(
        f"   Retries   : {payload['historical_retries']}"
    )

    print(
        f"   Fatigue   : {payload['customer_fatigue']:.2f}"
    )


def print_decision_summary(decision):
    """
    Displays the most important RecoveryOS decision
    information for the live demo.
    """

    if not decision:
        print("   ❌ Empty decision returned")
        return

    ai_proposal = (
        decision
        .get("ai_proposal", {})
    )

    policy = (
        decision
        .get("policy", {})
    )

    ai_action = ai_proposal.get(
        "selected_action",
        "UNKNOWN",
    )

    final_action = decision.get(
        "selected_action",
        "UNKNOWN",
    )

    probability = decision.get(
        "recovery_probability",
        0,
    )

    expected_revenue = decision.get(
        "expected_revenue",
        0,
    )

    overridden = policy.get(
        "action_overridden",
        False,
    )

    guardrail_flags = policy.get(
        "guardrail_flags",
        [],
    )

    outcome = decision.get(
        "outcome",
        {},
    )

    outcome_status = (
        decision.get(
            "outcome_status"
        )
        or outcome.get(
            "status",
            "PENDING",
        )
    )

    recovered_amount = (
        decision.get(
            "recovered_amount"
        )
        if decision.get(
            "recovered_amount"
        ) is not None
        else outcome.get(
            "recovered_amount",
            0,
        )
    )

    # -----------------------------------------------------
    # AI PROPOSAL
    # -----------------------------------------------------

    print(
        f"   🤖 AI Proposal : "
        f"{ai_action.replace('_', ' ').title()}"
    )

    print(
        f"      Probability: "
        f"{probability * 100:.1f}%"
    )

    print(
        f"      Expected   : "
        f"₹{expected_revenue:,.2f}"
    )

    # -----------------------------------------------------
    # POLICY
    # -----------------------------------------------------

    if overridden:
        print(
            f"   🛑 POLICY OVERRIDE"
        )

        if guardrail_flags:
            print(
                f"      Rule(s): "
                f"{', '.join(guardrail_flags)}"
            )

        print(
            f"      Final Action: "
            f"{final_action.replace('_', ' ').title()}"
        )

    else:
        print(
            f"   ✅ POLICY PASSED"
        )

        print(
            f"      Final Action: "
            f"{final_action.replace('_', ' ').title()}"
        )

    # -----------------------------------------------------
    # OUTCOME
    # -----------------------------------------------------

    if outcome_status != "PENDING":
        if outcome_status == "RECOVERED":
            print(
                f"   💰 OUTCOME: RECOVERED "
                f"₹{float(recovered_amount):,.2f}"
            )

        elif outcome_status == "FAILED":
            print(
                f"   ❌ OUTCOME: FAILED"
            )

        elif outcome_status == "NOT_ATTEMPTED":
            print(
                f"   ⏹️ OUTCOME: NOT ATTEMPTED"
            )

        else:
            print(
                f"   ℹ️ OUTCOME: "
                f"{outcome_status}"
            )


# =========================================================
# API REQUEST
# =========================================================

def send_payment(payload):
    """
    Sends a failed payment event to RecoveryOS.
    """

    try:
        response = session.post(
            API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        return response

    except requests.exceptions.Timeout:
        print(
            "   ⏱️ Request timed out"
        )
        return None

    except requests.exceptions.ConnectionError:
        print(
            "   ❌ Connection failed — "
            "is Flask running?"
        )
        return None

    except requests.exceptions.RequestException as exc:
        print(
            f"   ❌ Request error: {exc}"
        )
        return None


# =========================================================
# HEALTH CHECK
# =========================================================

def check_server():
    """
    Optional startup check.

    If /health exists in Flask, this provides a cleaner
    error before starting the simulator.
    """

    health_url = API_URL.replace(
        "/api/payment/failed",
        "/health",
    )

    try:
        response = session.get(
            health_url,
            timeout=3,
        )

        if response.ok:
            print(
                "✅ RecoveryOS API is online."
            )
            return True

    except requests.RequestException:
        pass

    print(
        "⚠️ Health endpoint unavailable."
    )

    print(
        "   Continuing anyway — "
        "the payment endpoint will be tested."
    )

    return True


# =========================================================
# MAIN SIMULATION
# =========================================================

def run_simulation(
    interval_seconds=DEFAULT_INTERVAL_SECONDS,
    guardrail_chance=DEFAULT_GUARDRAIL_CHANCE,
):
    """
    Continuously generates failed-payment traffic.

    Parameters
    ----------
    interval_seconds:
        Delay between requests.

    guardrail_chance:
        Probability that a transaction becomes an
        intentional edge case.
    """

    print(
        "\n"
        "===================================================="
    )

    print(
        "        RecoveryOS Traffic Simulator"
    )

    print(
        "===================================================="
    )

    print(
        f"API Endpoint      : {API_URL}"
    )

    print(
        f"Traffic Interval  : {interval_seconds}s"
    )

    print(
        f"Edge Case Chance  : "
        f"{guardrail_chance * 100:.0f}%"
    )

    print(
        "\nPress Ctrl+C to stop the simulator."
    )

    print(
        "===================================================="
    )

    check_server()

    count = 1

    try:

        while True:

            payload, scenario = generate_payload(
                guardrail_chance
            )

            print_request_summary(
                count,
                scenario,
                payload,
            )

            response = send_payment(
                payload
            )

            # -------------------------------------------------
            # API RESPONSE
            # -------------------------------------------------

            if response is None:
                time.sleep(
                    interval_seconds
                )

                count += 1

                continue

            if response.ok:

                try:
                    response_data = (
                        response.json()
                    )

                except ValueError:
                    print(
                        "   ❌ API returned "
                        "invalid JSON"
                    )

                else:
                    decision = (
                        extract_decision(
                            response_data
                        )
                    )

                    print_decision_summary(
                        decision
                    )

            else:

                print(
                    f"   ❌ API Error "
                    f"{response.status_code}"
                )

                print(
                    f"      {response.text[:300]}"
                )

            count += 1

            time.sleep(
                interval_seconds
            )

    except KeyboardInterrupt:

        print(
            "\n\n🛑 Traffic simulation stopped."
        )

        print(
            f"Total events generated: "
            f"{count - 1}"
        )

        print(
            "RecoveryOS simulator exited cleanly."
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run_simulation(
        interval_seconds=4,
        guardrail_chance=0.25,
    )