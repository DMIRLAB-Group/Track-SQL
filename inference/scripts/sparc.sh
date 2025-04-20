export CUDA_VISIBLE_DEVICES=2
python inference/sft_llm_inference.py \
    --model_path "/amax/storage/nfs/vpcctrl/d7/huggingface/deepseek-ai/deepseek-coder-6.7b-instruct" \
    --final_ckpts_path "ckpts/llm_ckpts/sparc/deepseek-coder-6.7b/checkpoint-900" \
    --dev_file "data/preprocessed_data/sparc_test/sft_dev.json" \
    --original_dev_path "data/original_data/sparc/dev.json" \
    --db_path "data/original_data/sparc/database/" \
    --results_path "inference/results/sparc" \
    --dataset_name "sparc" \
    --table_path "data/original_data/sparc/tables.json"

