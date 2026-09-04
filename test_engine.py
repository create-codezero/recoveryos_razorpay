import json
from services.ai_engine import RecoveryEngine

def run_test():
    print("Initializing RecoveryEngine...")
    
    try:
        engine = RecoveryEngine(model_path="model/catboost_recovery_laptop.cbm")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # A mock failed transaction representing what we'd get from a webhook
    mock_payment = {
        "transaction_id": "TXN_999123",
        "customer_id": "CUS_555",
        "amount": 8499.0,
        "payment_method": "upi",
        "failure_reason": "bank_timeout",
        "device": "mobile",
        "location": "tier_1",
        "previous_success_rate": 0.85,
        "days_since_last_payment": 12,
        "subscription_age_days": 150,
        "historical_retries": 1,
        "time_since_failure_mins": 5,
        "customer_value": "high",
        "merchant_category": "saas",
        "failure_hour": 14,
        "day_of_week": 2,
        "day_of_month": 15,
        "is_weekend": 0,
        "is_salary_window": 0,
        "network_quality": 0.9,
        "customer_fatigue": 0.1
    }

    print("Evaluating mock payment...")
    decision = engine.evaluate(mock_payment)

    print("\n=== AI DECISION OUTPUT ===")
    print(json.dumps(decision, indent=2))

if __name__ == "__main__":
    run_test()