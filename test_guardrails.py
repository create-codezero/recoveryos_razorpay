import requests
import json
import random

API_URL = "http://127.0.0.1:5000/api/payment/failed"

def run_test(test_name, overrides, target_overrides):
    print(f"\n{'='*50}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*50}")
    
    # Base payload with all 20 required features for the CatBoost model
    payload = {
        "transaction_id": f"TXN_TEST_{random.randint(1000, 9999)}",
        "customer_id": f"CUST_{random.randint(1000, 9999)}",
        "amount": 1000.00,
        "payment_method": "UPI",
        "failure_reason": "insufficient_funds",
        "device": "Android",
        "location": "Mumbai",
        "previous_success_rate": 0.85,
        "days_since_last_payment": 30,
        "subscription_age_days": 120,
        "historical_retries": 1,
        "time_since_failure_mins": 5,
        "customer_value": "High",
        "merchant_category": "SaaS",
        "failure_hour": 14,
        "day_of_week": 2,
        "day_of_month": 15,
        "is_weekend": 0,
        "is_salary_window": 0,
        "network_quality": 0.9,
        "customer_fatigue": 0.1
    }
    
    # Apply the specific test conditions
    payload.update(overrides)

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        decision = data["decision"]
        
        original_action = decision["policy"]["original_action"]
        final_action = decision["selected_action"]
        action_overridden = decision["policy"]["action_overridden"]
        flags = decision["policy"]["guardrail_flags"]
        
        if action_overridden:
            orig_data = next((a for a in decision.get("action_evaluations", []) if a["action"] == original_action), None)
            orig_prob = orig_data["recovery_probability"] if orig_data else 0.0
            orig_rev = orig_data["expected_revenue"] if orig_data else 0.0
        else:
            orig_prob = decision["recovery_probability"]
            orig_rev = decision["expected_revenue"]
            
        final_prob = decision["recovery_probability"]
        final_rev = decision["expected_revenue"]

        print(f"🤖 AI Proposed   : {original_action.upper()} ({orig_prob*100:.1f}% | ₹{orig_rev:.2f})")
        
        if action_overridden:
            print("🛑 Guardrail     : OVERRIDE TRIGGERED")
            for flag in flags:
                print(f"   -> {flag}")
        else:
            print("✅ Guardrail     : Passed (No override required)")
            if original_action not in target_overrides:
                print(f"   ℹ️ AI naturally avoided restricted actions {target_overrides}. Model is policy-aligned.")
                
        print(f"🟢 Final Action  : {final_action.upper()} ({final_prob*100:.1f}% | ₹{final_rev:.2f})")
        print(f"✅ Outcome Sim   : {decision['outcome']['outcome_status']} (₹{decision['outcome']['recovered_amount']})")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")

if __name__ == "__main__":
    # Test 1: Circuit Breaker
    run_test(
        test_name="Circuit Breaker (Retries >= 4)",
        overrides={"historical_retries": 5, "failure_reason": "bank_timeout", "amount": 2500.00},
        target_overrides=["retry_30m", "retry_evening"]
    )
    
    # Test 2: Fatigue Suppression
    run_test(
        test_name="Customer Fatigue (Fatigue >= 0.85)",
        overrides={"customer_fatigue": 0.95, "failure_reason": "authentication_failed", "amount": 1500.00},
        target_overrides=["whatsapp_reminder", "payment_link"]
    )
    
    # Test 3: Hard Failure
    run_test(
        test_name="Hard Failure (Expired Card)",
        overrides={"failure_reason": "expired_card", "historical_retries": 1, "amount": 3500.00},
        target_overrides=["retry_30m", "retry_evening"]
    )