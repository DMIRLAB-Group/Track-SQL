export CUDA_VISIBLE_DEVICES=2
python inference/sft_llm_inference.py \
    --model_path "huggingface/deepseek-ai/deepseek-coder-6.7b-instruct" \
    --final_ckpts_path "ckpts/track-sql/sparc/deepseek-coder-6.7b/checkpoint-900" \
    --dev_file "data/preprocessed_data/sparc_test/sft_dev.json" \
    --original_dev_path "raw_data/sparc/dev.json" \
    --db_path "raw_data/sparc/database/" \
    --results_path "inference/results/sparc" \
    --dataset_name "sparc" \
    --table_path "raw_data/sparc/tables.json"

