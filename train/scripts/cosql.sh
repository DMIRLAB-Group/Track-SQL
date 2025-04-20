export CUDA_VISIBLE_DEVICES=0,1
python train/sft_llm_training.py \
    --model_path "/amax/storage/nfs/vpcctrl/d7/huggingface/mistralai/Mistral-7B-Instruct-v0.2" \
    --train_file "data/preprocessed_data/sparc/symlink_star/sft_train.json" \
    --dev_file "data/preprocessed_data/sparc/symlink_star/sft_dev.json" \
    --output_dir "ckpts/llm_ckpts/sparc/symlink_star/instruct_v3/Mistral-7B-Instruct-v0.2" \
    --dataset_name "sparc" \
    --instruct_version 3

python train/sft_llm_training.py \
    --model_path "/amax/storage/nfs/vpcctrl/d7/huggingface/mistralai/Mistral-7B-Instruct-v0.2" \
    --train_file "data/preprocessed_data/cosql/symlink_star/sft_train.json" \
    --dev_file "data/preprocessed_data/cosql/symlink_star/sft_dev.json" \
    --output_dir "ckpts/llm_ckpts/cosql/symlink_star/instruct_v3/Mistral-7B-Instruct-v0.2" \
    --dataset_name "cosql" \
    --instruct_version 3


