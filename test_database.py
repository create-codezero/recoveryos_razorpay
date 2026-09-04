from services.database import save_decision

def run_test():
    print("Testing Firestore write...")
    
    mock_payment = {
        "transaction_id": "TXN_TEST_001",
        "customer_id": "CUS_TEST_001",
        "amount": 8499.0,
        "failure_reason": "bank_timeout"
    }
    
    mock_decision = {
        "selected_action": "retry_evening",
        "predicted_probability": 0.9101,
        "expected_revenue": 7735.36,
        "alternatives": [
            {"action": "retry_30m", "probability": 0.7887, "expected_revenue": 6703.36}
        ],
        "reason": "Transient error (bank_timeout). Evening retries historically yield maximum revenue.",
        "confidence": "High"
    }
    
    doc_id = save_decision(mock_payment, mock_decision)
    if doc_id:
        print(f"Success! Document created in Firestore with ID: {doc_id}")
    else:
        print("Failed to write to Firestore. Check your credentials file at config/firebase_service_account.json.")

if __name__ == "__main__":
    run_test()