python preprocess/cosql/preprocess_src.py \
    --model "deepseek-chat" \
    --api_key "" \
    --base_url "https://api.deepseek.com" \
    --train_file "data/original_data/cosql/sql_state_tracking/cosql_train.json" \
    --dev_file "data/original_data/cosql/sql_state_tracking/cosql_dev.json" \
    --preprocessed_train_file "data/preprocessed_data/cosql/preprocessed_train.json" \
    --preprocessed_dev_file "data/preprocessed_data/cosql/preprocessed_dev.json" \
    --comment_cache_train_file "data/preprocessed_data/cosql/comment_cache_train.json" \
    --comment_cache_dev_file "data/preprocessed_data/cosql/comment_cache_dev.json" \
    --table_path "data/original_data/cosql/tables.json" \
    --db_path "data/original_data/cosql/database" \
    --with_star True \
    --max_retries 10 \
    --random_content_num 10

