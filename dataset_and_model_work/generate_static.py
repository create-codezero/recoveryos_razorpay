import time
import numpy as np
import pandas as pd

def generate_recovery_dataset(num_records: int = 1_000_000, output_file: str = "recovery_dataset.parquet"):
    print(f"⚡ Generating {num_records:,} payment failure records with causal ground-truth...")
    start_time = time.time()

    # Reproducibility
    np.random.seed(42)

    # 1. Primary Entity Identifiers
    customer_ids = np.random.randint(100000, 999999, size=num_records)
    merchant_ids = np.random.randint(1000, 9999, size=num_records)
    
    customer_id_strs = [f"CUST_{cid}" for cid in customer_ids]
    merchant_id_strs = [f"MERCH_{mid}" for mid in merchant_ids]

    # 2. Categorical Features
    merchant_categories = np.random.choice(
        ["SaaS", "E-commerce", "EdTech", "Gaming", "Financial Services", "OTT / Media"], 
        size=num_records, 
        p=[0.20, 0.30, 0.15, 0.15, 0.10, 0.10]
    )
    
    payment_methods = np.random.choice(
        ["UPI", "Credit Card", "Debit Card", "Netbanking", "BNPL", "Wallet"], 
        size=num_records, 
        p=[0.55, 0.20, 0.12, 0.08, 0.03, 0.02]
    )

    failure_reasons = np.random.choice(
        ["insufficient_funds", "bank_timeout", "authentication_failed", "expired_card", "limit_exceeded", "user_cancelled"], 
        size=num_records, 
        p=[0.40, 0.25, 0.15, 0.08, 0.07, 0.05]
    )

    customer_values = np.random.choice(
        ["Low", "Medium", "High", "VIP"], 
        size=num_records, 
        p=[0.50, 0.35, 0.12, 0.03]
    )

    devices = np.random.choice(
        ["Android", "iOS", "Web-Desktop", "Web-Mobile"], 
        size=num_records, 
        p=[0.65, 0.15, 0.12, 0.08]
    )

    locations = np.random.choice(
        ["Bengaluru, KA", "Mumbai, MH", "Delhi, DL", "Hyderabad, TS", "Pune, MH", "Chennai, TN"], 
        size=num_records
    )

    # 3. Continuous & Numerical Features
    # Log-normal distribution for amounts (realistic payment ranges in INR)
    amounts = np.round(np.random.lognormal(mean=7.2, sigma=1.1, size=num_records), 2)
    amounts = np.clip(amounts, 99.0, 150000.0)

    previous_success_rates = np.round(np.random.beta(a=7, b=2, size=num_records), 3)
    days_since_last_payment = np.random.randint(0, 90, size=num_records)
    subscription_age_days = np.random.randint(1, 730, size=num_records)
    historical_retries = np.random.poisson(lam=0.8, size=num_records)
    historical_retries = np.clip(historical_retries, 0, 5)
    time_since_failure_mins = np.random.exponential(scale=120, size=num_records).astype(int) + 1

    # Hour of failure (0 to 23)
    failure_hour = np.random.randint(0, 24, size=num_records)

    # Day of month (1 to 30) - used for salary cycle logic
    day_of_month = np.random.randint(1, 31, size=num_records)

    # 4. Interventions Assigned (Simulated Historical Policy)
    interventions = np.random.choice(
        ["retry_30m", "retry_evening", "payment_link", "whatsapp_reminder", "alternate_method", "stop"],
        size=num_records,
        p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
    )

    # =========================================================================
    # 5. CAUSAL GROUND TRUTH MATHEMATICAL ENGINE
    # $P(\text{recovery}) = \sigma(\text{base\_logit} + \text{treatment\_effect} + \text{context\_interaction})$
    # =========================================================================
    
    # Base Log-Odds centered at ~0 (50% probability)
    logits = -0.5 + (previous_success_rates * 2.2) - (historical_retries * 0.4)

    # Salary cycle boost (1st-5th of month)
    is_salary_days = (day_of_month <= 5) | (day_of_month >= 28)
    logits += np.where(is_salary_days, 0.45, -0.1)

    # Rule 1: Insufficient funds responds best to evening retries or salary days
    mask_inf = (failure_reasons == "insufficient_funds")
    logits[mask_inf & (interventions == "retry_evening")] += 1.8
    logits[mask_inf & is_salary_days & (interventions == "retry_30m")] += 1.2
    logits[mask_inf & (~is_salary_days) & (interventions == "retry_30m")] -= 1.5  # Immediate retry fails if no money

    # Rule 2: Bank timeout responds best to quick retries
    mask_timeout = (failure_reasons == "bank_timeout")
    logits[mask_timeout & (interventions == "retry_30m")] += 2.1
    logits[mask_timeout & (interventions == "stop")] -= 2.0

    # Rule 3: Expired card MUST use alternate payment method
    mask_expired = (failure_reasons == "expired_card")
    logits[mask_expired & (interventions == "alternate_method")] += 2.4
    logits[mask_expired & (interventions == "retry_30m")] -= 3.5  # Retrying same card is useless
    logits[mask_expired & (interventions == "retry_evening")] -= 3.5

    # Rule 4: Authentication / User cancellation responds best to WhatsApp / Payment Link
    mask_auth = (failure_reasons == "authentication_failed") | (failure_reasons == "user_cancelled")
    logits[mask_auth & (interventions == "whatsapp_reminder")] += 1.6
    logits[mask_auth & (interventions == "payment_link")] += 1.4

    # Rule 5: 'stop' action never recovers money directly
    logits[interventions == "stop"] = -10.0

    # Calculate Probability using Sigmoid Function: 1 / (1 + exp(-logit))
    true_probabilities = 1.0 / (1.0 + np.exp(-logits))
    true_probabilities = np.clip(true_probabilities, 0.01, 0.98)

    # Bernoulli trial to generate actual binary outcome (1 = Recovered, 0 = Lost)
    recovered_outcomes = (np.random.uniform(0, 1, size=num_records) < true_probabilities).astype(int)

    # 6. Build Pandas DataFrame
    df = pd.DataFrame({
        "customer_id": customer_id_strs,
        "merchant_id": merchant_id_strs,
        "amount": amounts,
        "payment_method": payment_methods,
        "failure_reason": failure_reasons,
        "device": devices,
        "location": locations,
        "previous_success_rate": previous_success_rates,
        "days_since_last_payment": days_since_last_payment,
        "subscription_age_days": subscription_age_days,
        "historical_retries": historical_retries,
        "time_since_failure_mins": time_since_failure_mins,
        "failure_hour": failure_hour,
        "day_of_month": day_of_month,
        "customer_value": customer_values,
        "merchant_category": merchant_categories,
        "applied_intervention": interventions,
        "ground_truth_recovery_prob": np.round(true_probabilities, 4),
        "recovered": recovered_outcomes
    })

    # Save to Parquet (Fast & Compressed) or CSV
    if output_file.endswith(".parquet"):
        df.to_parquet(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)

    elapsed = time.time() - start_time
    print(f" Success! Generated {len(df):,} records in {elapsed:.2f} seconds.")
    print(f" File saved to: {output_file}")
    print(f" Overall Recovery Rate in Synthetic Data: {df['recovered'].mean() * 100:.2f}%\n")
    
    return df

if __name__ == "__main__":
    # Generate 1,000,000 records for model training
    df = generate_recovery_dataset(num_records=20_000_000, output_file="razorpay_causal_recovery_20m.parquet")
    print(df.head())