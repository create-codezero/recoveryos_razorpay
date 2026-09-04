import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import shap

class RecoveryEngine:
    def __init__(self, model_path: str = "model/catboost_recovery_laptop.cbm"):
        """
        Initializes the AI Engine, loads the CatBoost model, and preps the SHAP explainer.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        
        # Initialize SHAP TreeExplainer for real-time feature attribution
        self.explainer = shap.TreeExplainer(self.model)
        
        self.features = [
            "amount", "payment_method", "failure_reason", "device", 
            "location", "previous_success_rate", "days_since_last_payment", 
            "subscription_age_days", "historical_retries", "time_since_failure_mins", 
            "customer_value", "merchant_category", "failure_hour", "day_of_week", 
            "day_of_month", "is_weekend", "is_salary_window", "network_quality", 
            "customer_fatigue", "observed_intervention"
        ]

        self.categorical_features = [
            "payment_method", "failure_reason", "device", "location", 
            "customer_value", "merchant_category", "observed_intervention"
        ]

        self.actions = [
            "retry_30m", "retry_evening", "payment_link", 
            "whatsapp_reminder", "alternate_method", "stop"
        ]
        
        self.action_costs = {action: 0.0 for action in self.actions}

    def _prepare_dataframe(self, payment_context: dict) -> pd.DataFrame:
        rows = [payment_context.copy() for _ in self.actions]
        
        for i, action in enumerate(self.actions):
            rows[i]["observed_intervention"] = action
            
        df = pd.DataFrame(rows)
        
        for col in self.features:
            if col not in df.columns:
                df[col] = "__MISSING__" if col in self.categorical_features else 0.0
                    
        df = df[self.features]
        
        for col in self.categorical_features:
            df[col] = df[col].astype(str)
            
        numeric_features = [c for c in self.features if c not in self.categorical_features]
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            
        return df

    def evaluate(self, payment_context: dict) -> dict:
        amount = float(payment_context.get("amount", 0.0))
        
        # 1. Prepare data
        df = self._prepare_dataframe(payment_context)
        
        # 2. Predict probabilities for all 6 actions
        probabilities = self.model.predict_proba(df)[:, 1]
        
        # 3. Calculate expected revenue
        results = []
        for i, action in enumerate(self.actions):
            prob = float(probabilities[i])
            expected_revenue = (prob * amount) - self.action_costs.get(action, 0.0)
            results.append({
                "action": action,
                "probability": prob,
                "expected_revenue": expected_revenue,
                "original_index": i
            })
            
        # 4. Find the best action safely
        best_index = int(np.argmax(probabilities))
        
        # Sort results for the alternatives display, but use the safe index for the winner
        results_sorted = sorted(results, key=lambda x: x["expected_revenue"], reverse=True)
        best_action = next(item for item in results_sorted if item["original_index"] == best_index)
        
        # 5. SHAP Explainability (Real-time dynamic reasoning)
        winning_row = df.iloc[[best_index]]
        
        # Safely extract SHAP values (handles different CatBoost/SHAP version return shapes)
        raw_shap = self.explainer.shap_values(winning_row)
        if isinstance(raw_shap, list): 
            shap_values = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif len(raw_shap.shape) == 3: 
            shap_values = raw_shap[0, :, 1] if raw_shap.shape[2] > 1 else raw_shap[0, :, 0]
        else:
            shap_values = raw_shap[0]

        # Categorize positive and negative drivers
        pos_drivers = []
        neg_drivers = []
        
        for feat, shap_val in zip(self.features, shap_values):
            if feat == "observed_intervention":
                continue
            if shap_val > 0:
                pos_drivers.append({"feature": feat, "impact": float(shap_val)})
            elif shap_val < 0:
                neg_drivers.append({"feature": feat, "impact": float(shap_val)})

        # Sort by magnitude
        pos_drivers.sort(key=lambda x: x["impact"], reverse=True)
        neg_drivers.sort(key=lambda x: x["impact"]) # Most negative first
        
        top_pos = pos_drivers[:3]
        top_neg = neg_drivers[:2]
        
        dynamic_reason = f"{best_action['action'].replace('_', ' ').title()} was selected as the optimal intervention to maximize expected revenue."

        # 6. Format Output
        decision = {
            "selected_action": best_action["action"],
            "predicted_probability": round(best_action["probability"], 4),
            "expected_revenue": round(best_action["expected_revenue"], 2),
            "alternatives": [
                {
                    "action": alt["action"],
                    "probability": round(alt["probability"], 4),
                    "expected_revenue": round(alt["expected_revenue"], 2)
                } for alt in results_sorted if alt["action"] != best_action["action"]
            ],
            "reason": dynamic_reason,
            "shap_drivers": {
                "positive": top_pos,
                "negative": top_neg
            },
            "confidence": "High" if best_action["probability"] > 0.75 else "Medium" if best_action["probability"] > 0.5 else "Low"
        }
        
        return decision