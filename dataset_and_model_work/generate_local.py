import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
import ollama

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_NAME = "qwen2.5:7b"
OUTPUT_FILE = "razorpay_failed_payments.csv"
BATCH_SIZE_PER_LLM_CALL = 20  # Number of records generated per LLM request
SAVE_FLUSH_INTERVAL = 1000    # Log progress & flush to disk every N records

# CSV Header Definition
FIELDNAMES = [
    "customer_id",
    "merchant_id",
    "timestamp",
    "amount",
    "payment_method",
    "failure_reason",
    "device",
    "location",
    "previous_success_rate",
    "days_since_last_payment",
    "subscription_age",
    "historical_retries",
    "time_since_failure",
    "customer_value",
    "merchant_category",
    "optimal_intervention",
    "recovery_probability",
    "recovered"
]

# Random seed options to pass into prompts to avoid LLM repetition
MERCHANT_CATEGORIES = ["SaaS", "E-commerce", "EdTech", "Gaming", "Financial Services", "Health & Fitness", "Ott / Media"]
PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Netbanking", "BNPL", "Wallet"]
FAILURE_REASONS = [
    "insufficient_funds", 
    "bank_timeout", 
    "authentication_failed", 
    "expired_card", 
    "network_error", 
    "limit_exceeded", 
    "user_cancelled"
]
INDIAN_CITIES = [
    "Mumbai, MH", "Bengaluru, KA", "Delhi, DL", "Hyderabad, TS", 
    "Pune, MH", "Chennai, TN", "Kolkata, WB", "Ahmedabad, GJ", "Jaipur, RJ"
]

# =====================================================================
# SYSTEM & USER PROMPT BUILDER
# =====================================================================
SYSTEM_PROMPT = """
You are a senior data scientist specializing in Indian payment gateways (like Razorpay) and autonomous AI payment recovery systems.
Your job is to generate highly accurate, realistic, high-entropy synthetic data of failed payment transactions and recovery outcomes.

STRICT JSON OUTPUT RULES:
1. Output MUST be a valid JSON array containing exactly {batch_size} payment failure objects.
2. Do NOT include markdown code blocks, explanations, or text outside the raw JSON array.
3. Every field must strictly conform to valid data types (numbers, floats, strings).

Field expectations:
- customer_id: String (e.g., CUST_10842)
- merchant_id: String (e.g., MERCH_4091)
- timestamp: ISO 8601 string within the last 30 days
- amount: Float in INR (e.g., 199.00 to 85000.00)
- payment_method: One of ['UPI', 'Credit Card', 'Debit Card', 'Netbanking', 'BNPL', 'Wallet']
- failure_reason: Realistic payment failure reason
- device: One of ['Android', 'iOS', 'Web-Desktop', 'Web-Mobile']
- location: Indian City/State string
- previous_success_rate: Float between 0.05 and 0.99
- days_since_last_payment: Integer (0 to 180)
- subscription_age: Integer in days (0 to 1000)
- historical_retries: Integer (0 to 5)
- time_since_failure: Integer minutes (1 to 1440)
- customer_value: One of ['Low', 'Medium', 'High', 'VIP']
- merchant_category: Merchant vertical
- optimal_intervention: One of ['retry_30m', 'retry_evening', 'payment_link', 'whatsapp_reminder', 'alternate_method', 'stop']
- recovery_probability: Float between 0.01 and 0.98 reflecting realistic recovery odds based on failure reason and customer history
- recovered: Integer (1 if recovery succeeded under optimal intervention, 0 if failed)
"""

def build_user_prompt(batch_size: int, start_index: int) -> str:
    # Inject dynamic parameters so the model never generates repetitive patterns
    focus_category = random.choice(MERCHANT_CATEGORIES)
    focus_method = random.choice(PAYMENT_METHODS)
    focus_failure = random.choice(FAILURE_REASONS)
    focus_city = random.choice(INDIAN_CITIES)
    
    return f"""
Generate a batch of {batch_size} failed payment events starting with sequence index offset around {start_index}.

Context Seed for Variety:
- Focus heavily on merchant category: '{focus_category}'
- Primary payment method focus for this batch: '{focus_method}'
- Introduce realistic correlation around failure reason: '{focus_failure}' (e.g. insufficient_funds often recovers better in evening or salary hours; expired_card retries usually fail unless alternate payment method is used).
- Include diverse customer segments from cities like '{focus_city}'.

Return ONLY the JSON array of {batch_size} objects with no additional text or formatting.
"""

# =====================================================================
# UTILITIES
# =====================================================================
def count_existing_rows(filepath: str) -> int:
    """Counts existing rows in CSV to allow resuming seamless iteration."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)
        return max(0, row_count - 1)  # Subtract 1 for header

def initialize_csv(filepath: str):
    """Creates the CSV file with headers if it doesn't already exist."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        print(f"Initialized new dataset file: {filepath}")
    else:
        print(f"Found existing dataset file: {filepath}")

def clean_and_parse_json(raw_response: str) -> list:
    """Extracts and parses JSON array safely from Ollama response."""
    text = raw_response.strip()
    # Strip markdown code blocks if the LLM wraps it despite instructions
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data:
            return data["data"]
        else:
            return []
    except json.JSONDecodeError:
        # Fallback manual extraction between '[' and ']'
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        return []

# =====================================================================
# MAIN GENERATION LOOP
# =====================================================================
def main():
    initialize_csv(OUTPUT_FILE)
    total_generated = count_existing_rows(OUTPUT_FILE)
    print(f"Resuming generation. Current record count: {total_generated:,} rows.")
    print("Press Ctrl + C at any time to safely stop. Progress is continuously saved.\n")

    batch_buffer = []
    start_time = time.time()
    last_flush_count = total_generated

    try:
        while True:
            # 1. Call Ollama qwen2.5:7b
            user_prompt = build_user_prompt(BATCH_SIZE_PER_LLM_CALL, total_generated + 1)
            
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT.format(batch_size=BATCH_SIZE_PER_LLM_CALL)},
                        {"role": "user", "content": user_prompt},
                    ],
                    format="json",  # Enables native JSON mode in Ollama
                    options={
                        "temperature": 0.75,  # Higher temperature for rich entropy/variety
                        "top_p": 0.9,
                    },
                )
                
                raw_text = response["message"]["content"]
                records = clean_and_parse_json(raw_text)
                
                if not records:
                    print("⚠️ Warning: Received empty or unparseable JSON from model. Retrying batch...")
                    continue

                # 2. Append records to file immediately
                with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    valid_batch_count = 0
                    
                    for row in records:
                        if not isinstance(row, dict):
                            continue
                        
                        # Fallback defaults for missing fields
                        cleaned_row = {
                            key: row.get(key, "N/A") for key in FIELDNAMES
                        }
                        
                        writer.writerow(cleaned_row)
                        valid_batch_count += 1
                        total_generated += 1

                # 3. Status reporting & Periodic flushing feedback
                if total_generated - last_flush_count >= SAVE_FLUSH_INTERVAL:
                    elapsed = time.time() - start_time
                    rate = (total_generated - count_existing_rows(OUTPUT_FILE)) / max(elapsed, 1)
                    print(
                        f"✅ Milestone Reached: {total_generated:,} total rows saved to CSV "
                        f"({rate:.1f} rows/sec | Total Runtime: {elapsed/60:.1f} mins)"
                    )
                    last_flush_count = total_generated

            except Exception as e:
                print(f"❌ API or Processing Error: {e}. Retrying in 3 seconds...")
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n🛑 Process stopped by user.")
        print(f"💾 Total dataset saved successfully: {total_generated:,} records in '{OUTPUT_FILE}'.")
        sys.exit(0)

if __name__ == "__main__":
    main()