import os
from flask import Flask, request, jsonify, render_template
from services.decision import make_decision, get_engine
from services.database import save_decision, get_db

app = Flask(__name__)

# Initialize the RecoveryOS Decision Engine singleton once at startup
print("[SYSTEM] Starting RecoveryOS server...")
engine = get_engine()
print("[SYSTEM] RecoveryOS Decision Engine & CatBoost model ready.")

@app.route("/")
def index():
    """Serves the main RecoveryOS dashboard."""
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple ping endpoint to verify the server is running."""
    return jsonify({"status": "healthy", "service": "RecoveryOS Decision API"}), 200

@app.route("/api/payment/failed", methods=["POST"])
def handle_failed_payment():
    """
    Main webhook endpoint:
    1. Ingests failed payment event
    2. Runs action-conditioned CatBoost inference across 6 interventions
    3. Evaluates deterministic guardrails / policies
    4. Computes SHAP attributions
    5. Simulates execution outcome and feedback loop
    6. Writes structured audit entry to Firestore
    """
    payment_context = request.get_json()
    if not payment_context:
        return jsonify({"error": "Missing payload"}), 400

    try:
        # Run the complete RecoveryOS decision pipeline
        decision = make_decision(payment_context, simulate=True)

        # Persist the audit-ready decision record to Firestore
        doc_id = save_decision(payment_context, decision)

        response = {
            "status": "processed",
            "decision_id": decision.get("decision_id", doc_id),
            "transaction_id": payment_context.get("transaction_id"),
            "customer_id": payment_context.get("customer_id"),
            "decision": decision
        }
        return jsonify(response), 200

    except Exception as e:
        print(f"[ERROR] Failed to process payment context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/decisions", methods=["GET"])
def get_recent_decisions():
    """Returns the latest 30 decisions from Firestore for dashboard display."""
    try:
        db = get_db()
        docs = (
            db.collection("recovery_decisions")
            .order_by("timestamp", direction="DESCENDING")
            .limit(30)
            .stream()
        )
        decisions = [{"id": d.id, **d.to_dict()} for d in docs]
        return jsonify(decisions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/decisions/<decision_id>/execute", methods=["POST"])
def execute_decision(decision_id):
    """Marks a decision as executed in the audit log."""
    try:
        db = get_db()
        doc_ref = db.collection("recovery_decisions").document(decision_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Decision not found"}), 404

        doc_ref.update({"execution_status": "EXECUTED", "status": "executed"})
        return jsonify({"status": "executed", "decision_id": decision_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/decisions/<decision_id>/stop", methods=["POST"])
def stop_decision(decision_id):
    """Manually stops an automated recovery flow."""
    try:
        db = get_db()
        doc_ref = db.collection("recovery_decisions").document(decision_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Decision not found"}), 404

        doc_ref.update({"execution_status": "STOPPED", "status": "stopped", "stopped_by": "operator"})
        return jsonify({"status": "stopped", "decision_id": decision_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)