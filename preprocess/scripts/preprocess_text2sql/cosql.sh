export CUDA_VISIBLE_DEVICES=2
python preprocess/cosql/inference_sic.py \
        --batch_size 1 \
        --device "2" \
        --seed 42 \
        --model_name_or_path "ckpts/sic/cosql" \
        --input_train_dataset_path "data/preprocessed_data/cosql/preprocessed_train.json" \
        --output_train_dataset_path "data/preprocessed_data/cosql_test/sft_train.json" \
        --input_dev_dataset_path "data/preprocessed_data/cosql/preprocessed_dev.json" \
        --output_dev_dataset_path "data/preprocessed_data/cosql_test/sft_dev.json" \
        --use_contents true \
        --add_fk_info true \
        --topk_table_num 4 \
        --topk_column_num 5 \
        --noise_rate 0.08 \
        --table_threshold 0.1 \
        --column_threshold 0.1 \
        --plm_name "roberta-large"  \
        --max_input_len 512 \
        --plm_hidden_state_dim 1024 \
        --truncation true  \
        --add_comment \
        --pooling_function "attention" \
        --use_comment_enhanced \
        --use_column_enhanced

