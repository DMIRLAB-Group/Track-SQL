export CUDA_VISIBLE_DEVICES=1
python inference/sft_llm_inference.py \
    --model_path "/amax/storage/nfs/vpcctrl/d7/huggingface/deepseek-ai/deepseek-coder-6.7b-instruct" \
    --final_ckpts_path "ckpts/llm_ckpts/cosql/deepseek-coder-6.7b/checkpoint-600" \
    --dev_file "data/preprocessed_data/cosql_test/sft_dev.json" \
    --original_dev_path "data/original_data/cosql/sql_state_tracking/cosql_dev.json" \
    --db_path "data/original_data/cosql/database/" \
    --results_path "inference/results/cosql" \
    --dataset_name "cosql" \
    --table_path "data/original_data/cosql/tables.json"

