export CUDA_VISIBLE_DEVICES=1
python inference/sft_llm_inference.py \
    --model_path "huggingface/deepseek-ai/deepseek-coder-6.7b-instruct" \
    --final_ckpts_path "ckpts/track-sql/cosql/deepseek-coder-6.7b/checkpoint-600" \
    --dev_file "preprocessed_data/cosql_test/sft_dev.json" \
    --original_dev_path "raw_data/cosql_dataset/sql_state_tracking/cosql_dev.json" \
    --db_path "raw_data/cosql_dataset/database/" \
    --results_path "inference/results/cosql" \
    --dataset_name "cosql" \
    --table_path "raw_data/cosql_dataset/tables.json"

