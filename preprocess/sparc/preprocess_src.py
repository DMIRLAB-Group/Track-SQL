import os, sys
import re
import json
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocess.sparc.preprocess_utils import get_database_matches
from sql_metadata import Parser
from tqdm import tqdm
from utils.common_utils import load_json_file, save_json_file
from openai import OpenAI
import tiktoken
import time, sqlite3, re
from func_timeout import func_set_timeout, FunctionTimedOut

sql_keywords = ['select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', \
                'except', 'join', 'on', 'as', 'not', 'between', 'in', 'like', 'is', 'exists', 'max', 'min', \
                'count', 'sum', 'avg', 'and', 'or', 'desc', 'asc']

def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor

@func_set_timeout(120)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()

def parse_args():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--api_key', type=str, default="")
    parser.add_argument('--base_url', type=str, default="")
    parser.add_argument('--train_file', type=str, default="")
    parser.add_argument('--dev_file', type=str, default="")
    parser.add_argument('--preprocessed_train_file', type=str, default="")
    parser.add_argument('--preprocessed_dev_file', type=str, default="")
    parser.add_argument('--comment_cache_train_file', type=str, default="")
    parser.add_argument('--comment_cache_dev_file', type=str, default="")
    parser.add_argument('--table_path', type=str, default="")
    parser.add_argument('--db_path', type=str, default="")
    parser.add_argument('--max_retries', type=int, default=10)
    parser.add_argument('--random_content_num', type=int, default=10)
    parser.add_argument('--with_star', type=bool, default=True)
    parser.add_argument('--use_comment_cache', action='store_true')
    

    args = parser.parse_args()

    return args

def get_db_contents(question, table_name_original, column_names_original, db_id, db_path):
    matched_contents = []
    # extract matched contents for each column
    for column_name_original in column_names_original:
        matches = get_database_matches(
            question,
            table_name_original,
            column_name_original,
            os.path.join(db_path , "{}/{}.sqlite".format(db_id, db_id))
        )
        matches = sorted(matches)
        matched_contents.append(matches)

    return matched_contents


def get_db_schemas(args, all_db_infos):
    db_schemas = {}

    for db in all_db_infos:
        table_names_original = db["table_names_original"]
        table_names = db["table_names"]
        column_names_original = db["column_names_original"]
        column_names = db["column_names"]
        column_types = db["column_types"]

        db_schemas[db["db_id"]] = {}

        primary_keys, foreign_keys = [], []
        # record primary keys
        for pk_column_idx in db["primary_keys"]:
            pk_table_name_original = table_names_original[column_names_original[pk_column_idx][0]]
            pk_column_name_original = column_names_original[pk_column_idx][1]

            primary_keys.append(
                {
                    "table_name_original": pk_table_name_original.lower(),
                    "column_name_original": pk_column_name_original.lower()
                }
            )

        db_schemas[db["db_id"]]["pk"] = primary_keys

        # record foreign keys
        for source_column_idx, target_column_idx in db["foreign_keys"]:
            fk_source_table_name_original = table_names_original[column_names_original[source_column_idx][0]]
            fk_source_column_name_original = column_names_original[source_column_idx][1]

            fk_target_table_name_original = table_names_original[column_names_original[target_column_idx][0]]
            fk_target_column_name_original = column_names_original[target_column_idx][1]

            foreign_keys.append(
                {
                    "source_table_name_original": fk_source_table_name_original.lower(),
                    "source_column_name_original": fk_source_column_name_original.lower(),
                    "target_table_name_original": fk_target_table_name_original.lower(),
                    "target_column_name_original": fk_target_column_name_original.lower(),
                }
            )
        db_schemas[db["db_id"]]["fk"] = foreign_keys

        db_schemas[db["db_id"]]["schema_items"] = []
        for idx, table_name_original in enumerate(table_names_original):
            if args.with_star:
                column_names_original_list, column_names_list, column_types_list = ['*'], ['*'], ['text']
            else:
                column_names_original_list, column_names_list, column_types_list = [], [], []
            for column_idx, (table_idx, column_name_original) in enumerate(column_names_original):
                if idx == table_idx:
                    column_names_original_list.append(column_name_original.lower())
                    column_names_list.append(column_names[column_idx][1].lower())
                    column_types_list.append(column_types[column_idx])

            db_schemas[db["db_id"]]["schema_items"].append({
                "table_name_original": table_name_original.lower(),
                "table_name": table_names[idx].lower(),
                "column_names": column_names_list,
                "column_names_original": column_names_original_list,
                "column_types": column_types_list
            })

    return db_schemas


def normalization(sql):
    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace("\"", "'")

    def add_asc(s):
        pattern = re.compile(
            r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

        tables_aliases = new_tables_aliases
        for k, v in tables_aliases.items():
            s = s.replace("as " + k + " ", "")
            s = s.replace(k, v)

        return s

    processing_func = lambda x: remove_table_alias(add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))

    return processing_func(sql)


# extract the skeleton of sql and natsql
def extract_skeleton(sql, db_schema):
    table_names_original, table_dot_column_names_original, column_names_original = [], [], []
    for table in db_schema["schema_items"]:
        table_name_original = table["table_name_original"]
        table_names_original.append(table_name_original)

        for column_name_original in ["*"] + table["column_names_original"]:
            table_dot_column_names_original.append(table_name_original + "." + column_name_original)
            column_names_original.append(column_name_original)

    parsed_sql = Parser(sql)
    new_sql_tokens = []
    for token in parsed_sql.tokens:
        # mask table names
        if token.value in table_names_original:
            new_sql_tokens.append("_")
        # mask column names
        elif token.value in column_names_original \
                or token.value in table_dot_column_names_original:
            new_sql_tokens.append("_")
        # mask string values
        elif token.value.startswith("'") and token.value.endswith("'"):
            new_sql_tokens.append("_")
        # mask positive int number
        elif token.value.isdigit():
            new_sql_tokens.append("_")
        # mask negative int number
        elif isNegativeInt(token.value):
            new_sql_tokens.append("_")
        # mask float number
        elif isFloat(token.value):
            new_sql_tokens.append("_")
        else:
            new_sql_tokens.append(token.value.strip())

    sql_skeleton = " ".join(new_sql_tokens)

    # remove JOIN ON keywords
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    pattern3 = re.compile("_ (?:join _ ?)+")
    sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

    # "_ , _ , ..., _" -> "_"
    while ("_ , _" in sql_skeleton):
        sql_skeleton = sql_skeleton.replace("_ , _", "_")

    # remove clauses in WHERE keywords
    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        if "_ {} _".format(op) in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("_ {} _".format(op), "_")
    while ("where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton):
        if "where _ and _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        if "where _ or _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

    # remove additional spaces in the skeleton
    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")

    return sql_skeleton


def isNegativeInt(string):
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False


def isFloat(string):
    if string.startswith("-"):
        string = string[1:]

    s = string.split(".")
    if len(s) > 2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True
def fix_tables(tables):
    for table in tables:
        if table["db_id"] == "flight_2":
            table["foreign_keys"].append(
                [
                    1,
                    10
                ]
            )
    return tables

def get_db_contents_for_comment(db_path, db_id, table_name, random_content_num, original_column_names):
    sample_contents, column_types = [], []
    sqlite_db_path  = os.path.join(db_path, f"{db_id}/{db_id}.sqlite")
    cursor = get_cursor_from_path(sqlite_db_path)
    GET_RANDOM_ROWW_SQL = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {random_content_num};"
    GET_COLUMN_NAME_SQL = f"PRAGMA table_info({table_name})"
    column_info = execute_sql(cursor, GET_COLUMN_NAME_SQL)
    database_column_names = [row[1] for row in column_info]
    for db_col, ori_col in zip(database_column_names, original_column_names):
        db_col = db_col.lower()
        if  db_col != ori_col:
            print(f"database column name: {db_col}")
            print(f"tables column name: {ori_col}")
            raise "database column name not equal tables column names"
        
    column_types = [row[2] for row in column_info]
    sample_contents = execute_sql(cursor, GET_RANDOM_ROWW_SQL)   

    return sample_contents, column_types

def add_comment(args, dataset):
    preprocessed_schemas = {}
    total_tokens = 0
    for idx, data in enumerate(tqdm(dataset)):
        if idx > 1 :
            break
        db_id = data["db_id"]
        db_schemas = data["db_schema"]

        if db_id not in list(preprocessed_schemas.keys()):
            preprocessed_schemas[db_id] = []
            for schema in db_schemas:
                column_comments = ["All columns"]
                table_comment = ""
                table_name = schema["table_name_original"]
                column_names = schema["column_names_original"]
                # remove *
                column_names = column_names[1:]
                contents, column_types = get_db_contents_for_comment(db_path=args.db_path, 
                                                            db_id=db_id, 
                                                            table_name=table_name, 
                                                            random_content_num=args.random_content_num,
                                                            original_column_names=column_names
                                                            )
                sample_contents = {}
                for name in column_names:
                    sample_contents[name] = []
                for row in contents:
                    for index, cell in enumerate(row):
                        sample_contents[column_names[index]].append(cell)
                print(f"table name: {table_name}")
                table_comment, table_tokens = generate_table_description_function(args=args,
                                                                    client=client,
                                                                    table_name=table_name,
                                                                    column_names=column_names,
                                                                    column_types=column_types,
                                                                    sample_contents=sample_contents
                                                                    )
                total_tokens += table_tokens
                print(f"table comment: {table_comment}")
                for name, type in zip(column_names, column_types):
                    print(f"column name: {name}")
                    column_coment, column_tokens = generate_column_description_function(args=args,
                                                                                client=client,
                                                                                table_name=table_name,
                                                                                column_name=name, 
                                                                                column_type=type,
                                                                                sample_contents=sample_contents
                                                                                )
                    total_tokens += column_tokens
                    
                    column_comments.append(column_coment)
                    print(f"column comment: {column_coment}")
                schema["column_comments"] = column_comments
                schema["table_comment"] = table_comment
            preprocessed_schemas[db_id] = db_schemas
        else:
            db_schemas = preprocessed_schemas[db_id]

        data["db_schema"] = db_schemas
    print(f"token num = {total_tokens}")
        
    return dataset, preprocessed_schemas

def generate_column_description_function(args, client, table_name, column_name, column_type, sample_contents):
    values = sample_contents[column_name]

    input = f"""
            "Table Name: {table_name}
            Column: {column_name}
            Type: {column_type}
            Sample Values: {values}"
            """
    # print(f"Column Input: {input} -------------------------------------")
    prompt = f"""
                You are a database schema designer who specializes in generating concise descriptions for table columns based on the provided column names, types, and sample values. The descriptions should be brief, adjective-noun phrases that reflect the nature of the data in the column.
            """
    retry_count = 0
    while retry_count < args.max_retries:
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input},
                ],
                stream=False
            )
            break  
        except Exception as e:
            print(f"Request failed with error: {e}. Retrying...")
            retry_count += 1
            time.sleep(2)  
    column_descript = response.choices[0].message.content
    column_descript = re.sub(r'\s+', ' ', column_descript).strip()
    column_descript = column_descript.replace("Short Description:", "").replace("\"", "").replace("Column Description:", "")

    encoding = tiktoken.encoding_for_model(args.model_name)
    prompt_tokens = len(encoding.encode(prompt))
    input_tokens = len(encoding.encode(input))
    output_tokens = len(encoding.encode(column_descript))
    total_tokens = input_tokens + output_tokens + prompt_tokens

    return column_descript, total_tokens

def generate_table_description_function(args, client, table_name, column_names, column_types, sample_contents):
    column_strs = ""
    for index , (col, type) in enumerate(zip(column_names, column_types)):
        values = sample_contents[col]
        column_strs += f"- Column {index}: name = {col}; type = {type}; values = {values}" + "\n"

    input = f"""
            The table name is '{table_name}'. It has the following columns:
            {column_strs}
            Based on this information, please generate a concise description for the table.
            """
    # print(f"Table input: {input} -------------------------------------------------------------")
    prompt = """
                You are a database schema expert who is skilled at generating concise descriptions for database tables based on their names and column details. Your job is to create a brief, adjective-noun form description that captures the essence of the table. The description should be short and to the point, not exceeding a few words. You will be given the table name, column names, types, and sample values. Generate a descriptive phrase using this information.
             """
    retry_count = 0
    while retry_count < args.max_retries:
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input},
                ],
                stream=False
            )
            break  
        except Exception as e:
            print(f"Request failed with error: {e}. Retrying...")
            retry_count += 1
            time.sleep(2)  
    table_descript = response.choices[0].message.content
    table_descript = re.sub(r'\s+', ' ', table_descript).strip()
    table_descript = table_descript.replace("Short Description:", "").replace("\"", "").replace("Column Description:", "")

    encoding = tiktoken.encoding_for_model(args.model_name)
    prompt_tokens = len(encoding.encode(prompt))
    input_tokens = len(encoding.encode(input))
    output_tokens = len(encoding.encode(table_descript))
    total_tokens = input_tokens + output_tokens + prompt_tokens

    return table_descript, total_tokens


def preprocess_data(args, dataset, output_dataset_path, comment_cache_file_path):
    all_db_infos = json.load(open(args.table_path))
    db_schemas = get_db_schemas(args, all_db_infos)

    preprocessed_dataset = []

    for data in tqdm(dataset):
        question = data["question"].replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "'").replace(
            "\u201d", "'").strip()
        db_id = data["db_id"]

        sql = data["query"].strip()
        norm_sql = normalization(sql).strip()
        sql_tokens = norm_sql.split()

        preprocessed_data = {}
        preprocessed_data["question"] = question
        preprocessed_data["question_relations"] = data["question_relations"]
        preprocessed_data["db_id"] = db_id
        preprocessed_data["norm_sql"] = norm_sql
        preprocessed_data["db_schema"] = []
        preprocessed_data["pk"] = db_schemas[db_id]["pk"]
        preprocessed_data["fk"] = db_schemas[db_id]["fk"]
        preprocessed_data["table_labels"] = []
        preprocessed_data["column_labels"] = []
        preprocessed_data["turn_idx"] = data["turn_idx"]

        # add database information (including table name, column name, ..., table_labels, and column labels)
        for table in db_schemas[db_id]["schema_items"]:
            db_contents = get_db_contents(
                question,
                table["table_name_original"],
                table["column_names_original"],
                db_id,
                args.db_path
            )

            preprocessed_data["db_schema"].append({
                "table_name_original": table["table_name_original"],
                "table_name": table["table_name"],
                "column_names": table["column_names"],
                "column_names_original": table["column_names_original"],
                "column_types": table["column_types"],
                "db_contents": db_contents
            })

            # extract table and column classification labels
            if table["table_name_original"] in sql_tokens:  # for used tables
                preprocessed_data["table_labels"].append(1)
                column_labels = []
                for column_name_original in table["column_names_original"]:
                    if column_name_original in sql_tokens or \
                            table[
                                "table_name_original"] + "." + column_name_original in sql_tokens:  # for used columns
                        column_labels.append(1)
                    else:
                        column_labels.append(0)
                preprocessed_data["column_labels"].append(column_labels)
            else:  # for unused tables and their columns
                preprocessed_data["table_labels"].append(0)
                preprocessed_data["column_labels"].append([0 for _ in range(len(table["column_names_original"]))])

        preprocessed_dataset.append(preprocessed_data)
    print("start LLM")
    if args.use_comment_cache:
        comment_cache = load_json_file(args.comment_cache_file)
        for data in preprocessed_dataset:
            schemas = data["db_schema"]
            db_id = data["db_id"]
            comment_schema = comment_cache[db_id]
            for schema in schemas:
                schema["column_comments"] = comment_schema["column_comments"]
                schema["table_comment"] = comment_schema["table_comment"]
            data["db_schema"] = schemas
    else:
        preprocessed_dataset, comment_cache = add_comment(args, preprocessed_dataset)
        save_json_file(comment_cache_file_path, comment_cache)
    with open(output_dataset_path, "w") as f:
        preprocessed_dataset_str = json.dumps(preprocessed_dataset, indent=2, ensure_ascii=False)
        f.write(preprocessed_dataset_str)

if __name__ == "__main__":
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    # Sparc Train
    dataset = load_json_file(args.train_file)
    new_dataset = []
    print("preprocess train dataset , preprocess multi-turn data to single-turn data")
    for idx, data in enumerate(tqdm(dataset)):
        # if idx > 2:
        #     break
        pre_question = ""
        for index, entry in enumerate(data["interaction"]):
            question = ""
            question_relations = []
            if index == 0:
                pre_question = entry["utterance"]
                question = entry["utterance"]
            else:
                entry["utterance"] = pre_question + " & " + entry["utterance"]
                pre_question = entry["utterance"]
                question = entry["utterance"]
            new_dataset.append(
                {
                    "db_id": data["database_id"],
                    "question": question,
                    "question_relations": question_relations,
                    "query": entry["query"],
                    "turn_idx": index
                }
            )
    output_dataset_path = args.preprocessed_train_file
    comment_cache_file_path = args.comment_cache_train_file
    
    preprocess_data(args, new_dataset, output_dataset_path, comment_cache_file_path)

    # Sparc Dev
    dataset = load_json_file(args.dev_file)
    new_dataset = []
    print("preprocess dev dataset , preprocess multi-turn data to single-turn data")
    for idx, data in enumerate(tqdm(dataset)):
        # if idx > 2:
        #     break
        pre_question = ""
        for index, entry in enumerate(data["interaction"]):
            question = ""
            question_relations = []
            if index == 0:
                pre_question = entry["utterance"]
                question = entry["utterance"]
            else:
                entry["utterance"] = pre_question + " & " + entry["utterance"]
                pre_question = entry["utterance"]
                question = entry["utterance"]
                
            new_dataset.append(
                {
                    "db_id": data["database_id"],
                    "question": question,
                    "question_relations": question_relations,
                    "query": entry["query"],
                    "turn_idx": index
                }
            )
    output_dataset_path = args.preprocessed_dev_file
    comment_cache_file_path = args.comment_cache_dev_file
    preprocess_data(args, new_dataset, output_dataset_path, comment_cache_file_path)



