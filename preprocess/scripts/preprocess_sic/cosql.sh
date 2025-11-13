python preprocess/cosql/preprocess_src.py \
    --model "" \
    --api_key "" \
    --base_url "" \
    --train_file "raw_data/cosql_dataset/sql_state_tracking/cosql_train.json" \
    --dev_file "raw_data/cosql_dataset/sql_state_tracking/cosql_dev.json" \
    --preprocessed_train_file "preprocessed_data/cosql/preprocessed_train.json" \
    --preprocessed_dev_file "preprocessed_data/cosql/preprocessed_dev.json" \
    --comment_cache_train_file "preprocessed_data/cosql/comment_cache_train.json" \
    --comment_cache_dev_file "preprocessed_data/cosql/comment_cache_dev.json" \
    --table_path "raw_data/cosql_dataset/tables.json" \
    --db_path "raw_data/cosql_dataset/database" \
    --with_star True \
    --max_retries 10 \
    --random_content_num 10

