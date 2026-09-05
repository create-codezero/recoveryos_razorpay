def apply_guardrails(payment_context: dict, ai_decision: dict) -> dict:

    final_decision = ai_decision.copy()
    flags = []

    retries = float(payment_context.get("historical_retries", 0))
    fatigue = float(payment_context.get("customer_fatigue", 0.0))
    failure_reason = str(
        payment_context.get("failure_reason", "")
    ).lower()

    original_action = ai_decision.get("selected_action")
    final_action = original_action

    # ---------------------------------------------------------
    # Build lookup of all model-evaluated actions
    # ---------------------------------------------------------
    action_data = {
        original_action: {
            "probability": float(
                ai_decision.get("predicted_probability", 0.0)
            ),
            "expected_revenue": float(
                ai_decision.get("expected_revenue", 0.0)
            ),
        }
    }

    for alt in ai_decision.get("alternatives", []):
        action_data[alt["action"]] = {
            "probability": float(alt["probability"]),
            "expected_revenue": float(alt["expected_revenue"]),
        }

    # ---------------------------------------------------------
    # Rule 1: Retry circuit breaker
    # ---------------------------------------------------------
    if retries >= 4 and final_action in [
        "retry_30m",
        "retry_evening"
    ]:
        final_action = "alternate_method"

        flags.append(
            "GUARDRAIL: Retries exceeded threshold (>= 4). "
            "Switched to alternate_method."
        )

    # ---------------------------------------------------------
    # Rule 2: Customer fatigue suppression
    # ---------------------------------------------------------
    if fatigue >= 0.85 and final_action in [
        "whatsapp_reminder",
        "payment_link"
    ]:
        final_action = "stop"

        flags.append(
            "GUARDRAIL: High customer fatigue (>= 0.85). "
            "Action halted to protect customer UX."
        )

    # ---------------------------------------------------------
    # Rule 3: Hard failure incompatibility
    # ---------------------------------------------------------
    if (
        failure_reason in ["expired_card", "limit_exceeded"]
        and final_action in ["retry_30m", "retry_evening"]
    ):
        final_action = "alternate_method"

        flags.append(
            f"GUARDRAIL: Hard failure ({failure_reason}) "
            "cannot be resolved by automatic retries. "
            "Switched to alternate_method."
        )

    # ---------------------------------------------------------
    # Rule 4: Economic viability
    # ---------------------------------------------------------
    current_action_data = action_data.get(final_action)

    if current_action_data:
        final_expected_revenue = current_action_data["expected_revenue"]

        if final_expected_revenue <= 0.0 and final_action != "stop":
            final_action = "stop"

            flags.append(
                "GUARDRAIL: Expected recovered revenue is non-positive. "
                "Workflow terminated."
            )

    # ---------------------------------------------------------
    # Apply final action's model outputs
    # ---------------------------------------------------------
    if final_action in action_data:

        final_decision["selected_action"] = final_action

        final_decision["predicted_probability"] = round(
            action_data[final_action]["probability"],
            4
        )

        final_decision["expected_revenue"] = round(
            action_data[final_action]["expected_revenue"],
            2
        )

    # ---------------------------------------------------------
    # Rebuild alternatives
    # ---------------------------------------------------------
    alternatives = []

    for action, data in action_data.items():

        if action == final_action:
            continue

        alternatives.append({
            "action": action,
            "probability": round(data["probability"], 4),
            "expected_revenue": round(
                data["expected_revenue"], 2
            )
        })

    alternatives.sort(
        key=lambda x: x["expected_revenue"],
        reverse=True
    )

    final_decision["alternatives"] = alternatives

    # ---------------------------------------------------------
    # Audit metadata
    # ---------------------------------------------------------
    final_decision["original_action"] = original_action
    final_decision["action_overridden"] = (
        final_action != original_action
    )

    final_decision["guardrail_flags"] = flags

    final_decision["guardrails_passed"] = len(flags) == 0

    return final_decision
