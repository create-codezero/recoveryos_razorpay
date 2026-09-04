import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase only once (Flask hot-reloads can cause errors if we don't check this)
if not firebase_admin._apps:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(current_dir, "..", "config", "firebase_service_account.json")
    
    if os.path.exists(cert_path):
        cred = credentials.Certificate(cert_path)
        firebase_admin.initialize_app(cred)
        print("[DB] Firebase initialized successfully.")
    else:
        print(f"[WARNING] Firebase credentials not found at {cert_path}. Writes will fail.")

def get_db():
    return firestore.client()

def save_decision(payment_context: dict, decision: dict) -> str:
    """
    Saves the failed payment context and the AI's decision to Firestore.
    Returns the generated document ID.
    """
    try:
        db = get_db()
        
        # 1. Map SHAP explanation 'value' to 'impact' for dashboard.js compatibility
        explanation = decision.get("explanation", {})
        shap_drivers = {
            "positive": [
                {"feature": d["feature"], "impact": d["value"]} 
                for d in explanation.get("positive_drivers", [])
            ],
            "negative": [
                {"feature": d["feature"], "impact": d["value"]} 
                for d in explanation.get("negative_drivers", [])
            ]
        }

        # 2. Map action_evaluations to alternative_actions for dashboard.js
        alt_actions = [
            {
                "action": alt.get("action"),
                "probability": alt.get("recovery_probability"),
                "expected_revenue": alt.get("expected_revenue")
            }
            for alt in decision.get("action_evaluations", [])
        ]

        # 3. Structure the flat data for our audit trail
        doc_data = {
            "transaction_id": decision.get("transaction_id", "UNKNOWN"),
            "customer_id": payment_context.get("customer_id", "UNKNOWN"),
            "amount": decision.get("amount", 0.0),
            "failure_reason": payment_context.get("failure_reason", "unknown"),
            
            "recommended_action": decision.get("selected_action"),
            "predicted_recovery_probability": decision.get("recovery_probability"),
            "expected_revenue": decision.get("expected_revenue"),
            
            "original_action": decision.get("policy", {}).get("original_action"),
            "action_overridden": decision.get("policy", {}).get("action_overridden", False),
            
            "alternative_actions": alt_actions,
            "decision_reason": decision.get("reason"),
            "shap_drivers": shap_drivers,
            
            "guardrail_flags": decision.get("policy", {}).get("guardrail_flags", []),
            "guardrails_passed": decision.get("policy", {}).get("guardrails_passed", True),
            
            # --- OUTCOME MONITOR & FEEDBACK FIELDS ---
            "outcome_status": decision.get("outcome_status", "PENDING"),
            "recovered_amount": decision.get("recovered_amount", 0.0),
            "feedback_queue": decision.get("feedback_status", "PENDING"),
            
            "status": "recommended",
            "timestamp": decision.get("timestamp") or (datetime.datetime.utcnow().isoformat() + "Z")
        }
        
        # Write to the 'recovery_decisions' collection
        # We use the deterministic decision_id if available to prevent duplicates on retries
        doc_id = decision.get("decision_id")
        if doc_id:
            db.collection("recovery_decisions").document(doc_id).set(doc_data)
            return doc_id
        else:
            _, doc_ref = db.collection("recovery_decisions").add(doc_data)
            return doc_ref.id
        
    except Exception as e:
        print(f"[DB ERROR] Failed to save decision to Firestore: {e}")
        return None