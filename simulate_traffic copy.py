import time
import requests
import pandas as pd
import random
import json

# Point this to wherever your parquet file is located
DATA_PATH = "../dataset_gen/recovery_prepared/recovery_test.parquet"
API_URL = "http://127.0.0.1:5000/api/payment/failed"

def run_simulator():
    print(f"Loading data from {DATA_PATH}...")
    try:
        # Load just a chunk to save memory
        df = pd.read_parquet(DATA_PATH).sample(n=500, random_state=42)
    except FileNotFoundError:
        print(f"File not found at {DATA_PATH}. Please update the DATA_PATH variable.")
        return

    print(f"Loaded {len(df)} transactions. Starting simulator...\n")

    for index, row in df.iterrows():
        # Use Pandas built-in to_json to safely handle Timestamps, then load back to dict
        payload = json.loads(row.to_json(date_format='iso'))
        
        # Add mock IDs if they don't exist in the parquet
        if "transaction_id" not in payload:
            payload["transaction_id"] = f"TXN_{random.randint(100000, 999999)}"
        if "customer_id" not in payload:
            payload["customer_id"] = f"CUS_{random.randint(1000, 9999)}"

        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                data = response.json()
                decision = data.get("decision", {}).get("selected_action")
                revenue = data.get("decision", {}).get("expected_revenue")
                print(f"[SENT] {payload['transaction_id']} - ₹{payload['amount']:.2f} | AI chose: {decision} (₹{revenue})")
            else:
                print(f"[ERROR] API returned {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[CONNECTION ERROR] Ensure 'python app.py' is running. ({e})")
            break

        # Pause between 1 and 3 seconds to simulate organic traffic
        time.sleep(random.uniform(1.0, 3.0))

if __name__ == "__main__":
    run_simulator()