python generate_v2.py --rows 20000000 --customers 2000000 --merchants 20000 --chunk-size 250000 --output razorpay_recovery_v2_20m.parquet


python prepare_recovery_data.py --input razorpay_recovery_v2_20m.parquet --outdir recovery_prepared


python train_parallel_gpu.py --data-dir recovery_prepared


python evaluate_recovery_policy.py --input razorpay_recovery_v2_20m.parquet --output-dir policy_evaluation


python evaluate_recovery_policy_ai_fixed.py --input razorpay_recovery_v2_20m.parquet --model recovery_models\catboost_recovery_laptop.cbm --output-dir policy_evaluation_ai --prediction-chunk-size 100000 --no-segments


python evaluate_recovery_policy_ai_heldout.py --input recovery_prepared\recovery_test.parquet --model recovery_models\catboost_recovery_laptop.cbm --output-dir policy_evaluation_ai_heldout --prediction-chunk-size 100000