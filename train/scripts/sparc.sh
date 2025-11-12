export CUDA_VISIBLE_DEVICES=0,1
python train/sft_llm_training.py \
    --model_path "huggingface/mistralai/Mistral-7B-Instruct-v0.2" \
    --train_file "data/preprocessed_data/sparc/sft_train.json" \
    --dev_file "data/preprocessed_data/sparc/sft_dev.json" \
    --output_dir "ckpts/llm_ckpts/sparc/instruct_v3/Mistral-7B-Instruct-v0.2" \
    --dataset_name "sparc" 

python train/sft_llm_training.py \
    --model_path "huggingface/mistralai/Mistral-7B-Instruct-v0.2" \
    --train_file "data/preprocessed_data/cosql/sft_train.json" \
    --dev_file "data/preprocessed_data/cosql/sft_dev.json" \
    --output_dir "ckpts/llm_ckpts/cosql/instruct_v3/Mistral-7B-Instruct-v0.2" \
    --dataset_name "cosql" 


