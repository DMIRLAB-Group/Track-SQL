import torch
import re, os, sys
import pandas as pd
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel
from torch import cuda
from transformers import StoppingCriteria
from tqdm import tqdm
import argparse, json, os
from utils.common_utils import load_json_file, save_json_file
from utils.exec_filter import exec_filter
from inference.sbert_request import request_one
import numpy as np
from scipy.stats import entropy

def compute_js_divergence(a, b, epsilon=1e-10):
    # 归一化并添加epsilon防止数值问题
    def normalize(lst):
        total = sum(lst)
        normalized = [x / total for x in lst]
        # 添加epsilon并重新归一化
        normalized = [x + epsilon for x in normalized]
        return [x / sum(normalized) for x in normalized]
    
    a_norm = normalize(a)
    b_norm = normalize(b)
    
    # 计算平均分布M
    m = [(x + y) / 2 for x, y in zip(a_norm, b_norm)]
    
    # 计算KL散度
    kl_am = entropy(a_norm, m)
    kl_bm = entropy(b_norm, m)
    
    # 计算JS散度并归一化到[0,1]
    js_divergence = 0.5 * kl_am + 0.5 * kl_bm
    js_normalized = js_divergence / np.log(2) 
    
    return js_normalized

def flattening(tab_probs, col_probs):
    schema_probs = [item for sublist in col_probs for item in sublist]
    schema_probs = tab_probs + schema_probs
    return schema_probs


def filter_history_query(history_query, 
                         history_questions, 
                         current_question,
                         history_table_probs, 
                         history_column_probs, 
                         current_table_probs,
                         current_column_probs,
                         db_file_path):
    # return history_query[-1]
    if len(history_query) == 1:
        return history_query[0]
    
    # SQL exec filter
    exec_scores = []
    exec_queries = []
    for query in history_query:
        exec = exec_filter(query, db_file_path)
        exec_scores.append(int(exec))
        if exec:
            exec_queries.append(query)

    if len(exec_queries) in [0,1]:
        return history_query[-1]

    # question semantic
    scores = [0]*len(history_questions)
    for index, pre_question in enumerate(history_questions):
        _,_,r = request_one(current_question, pre_question)
        scores[index] += r

    # entity
    cur_schema_probs = flattening(current_table_probs, current_column_probs)
    for index, (pre_tab_probs, pre_col_probs) in enumerate(zip(history_table_probs, history_column_probs)):
        pre_schema_probs = flattening(pre_tab_probs, pre_col_probs)
    
        js_value = compute_js_divergence(pre_schema_probs, cur_schema_probs)
        scores[index] += 1 - js_value
    
    final_scores = [ exec * se_entity_value for exec, se_entity_value in zip(exec_scores, scores)]

    max_score_index = final_scores.index(max(final_scores))

    return history_query[max_score_index]

INSTRUCT = """
You are a SQL query generator that converts multi-round questions along with associated database schema information into corresponding SQL statements. The multi-round questions will be concatenated using the '&' symbol, and you should generate the SQL statement that answers the current round of the question based on the provided database schema.

Each database schema is provided in the following format:

Table name : Column name1, Column name2 Different tables are separated by the '|' symbol, and the order of table names and column names is relevant to the current question; those appearing earlier are more closely related.
"""

def detect_special_char(name):
    for special_char in ['(', '-', ')', ' ', '/']:
        if special_char in name:
            return True

    return False
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

def parse_args():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")

    parser.add_argument('--model_path', type=str, default="huggingface/deepseek-ai/deepseek-coder-6.7b-instruct/")
    parser.add_argument('--final_ckpts_path', type=str, default="ckpts/llm_ckpts/sparc/deepseek-coder-6.7b/")
    parser.add_argument('--dev_file', type=str, default="data/preprocessed_data/sparc/sft_dev.json")
    parser.add_argument('--original_dev_path', type=str, default="./data/original_data/sparc/dev.json")
    parser.add_argument('--db_path', type=str, default="data/original_data/sparc/database/")
    parser.add_argument('--results_path', type=str, default="inference/results/sparc/")
    parser.add_argument('--dataset_name', type=str, default="sparc")
    parser.add_argument('--table_path', type=str, default="")
    args = parser.parse_args()

    return args



def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2", # use with amper architecture
        torch_dtype=torch.bfloat16,
        device_map = "auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, args.final_ckpts_path, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.encode(' ;')
    special_end_token = ' ;'
    # special_end_token_id = tokenizer.encode(special_end_token, add_special_tokens=False)[0]
    # print(special_end_token_id)
    special_end_token_id = 6203

    def call_mistral(inputs, special_end_token_id):
        output_tokens = model.generate(inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id,
                                       stopping_criteria=[EosListStoppingCriteria(eos_sequence=[special_end_token_id])])
        return tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True)

    dataset = load_json_file(args.dev_file)

    results = []
    result = []
    pre_result = ""
    user_messages = []
    history = {
        "queries":[],
        "question":[],
        "tab_probs": [],
        "col_probs": []
    }
    for index, data in enumerate(tqdm(dataset)):
        input_sequence = data['input_sequence']
        question = input_sequence.split(" | ")[0]
        database_schema = " | ".join(input_sequence.split(" | ")[1:])
        if data["turn_idx"] == 0 and pre_result != "":
            # results.append(pre_result)
            results.append(result)
            result = []
            history = {
                "queries":[],
                "question":[],
                "tab_probs": [],
                "col_probs": []
            }
        db_file_path = os.path.join(args.db_path, data["db_id"], data["db_id"] + ".sqlite")
        if data["turn_idx"] > 0:
            previous_query = filter_history_query(history["queries"], 
                                                  history["question"], 
                                                  question, 
                                                  history["tab_probs"], 
                                                  history["col_probs"], 
                                                  data["table_pred_probs"], 
                                                  data["column_pred_probs"],
                                                  db_file_path
                                                  )
            previous_query = "SQL corresponding to the previous round of questions: " + previous_query
        else:
            previous_query = ""
        instruct = INSTRUCT + "\n" + previous_query + "\n" + "The task is to take the multi-round question and the database schema as input and output an SQL statement that answers the question correctly."
        user_message = instruct + "\n" + "database schema: " + database_schema + "\n" + "question: " + question

        user_messages.append(user_message)
        messages = [
            {"role": "user", "content": user_message.strip()}
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True,
                                               tokenize=True).to(model.device)
        response = call_mistral(inputs, special_end_token_id)
        
        if ";" in response:
            response = response.split(";")[0]
            if "```sql" in response:
                response = response.split("```sql")[1]
        response = re.sub(r'\s+', ' ', response).strip()
        pre_result = response
        # results.append(response)
        result.append([response, data["db_id"]])
        history["queries"].append(response)
        history["question"].append(question)
        history["tab_probs"].append(data["table_pred_probs"])
        history["col_probs"].append(data["column_pred_probs"])
        print(response)
    results.append(result)
    # evaluate
    if args.dataset_name in ["sparc", "cosql"]:

        from metrics.multiturn.evaluator import Evaluator
        evaluator = Evaluator(args.table_path, args.db_path)
        evaluate_result = evaluator.evaluate_with_official_interface(results, args.original_dev_path)
        simple_evaluate_result = {}
        simple_evaluate_result["QM"] = evaluate_result["exact"]
        simple_evaluate_result["QX"] = evaluate_result["exec"]
        simple_evaluate_result["IM"] = evaluate_result['in_exact']
        simple_evaluate_result["IX"] = evaluate_result['in_exec']
        detail_results = evaluate_result["results"]
        evaluator.change_database("metrics/ts_database/database")
        evaluate_result = evaluator.evaluate_with_official_interface(results, args.original_dev_path)
        simple_evaluate_result["TS-QX"] = evaluate_result["exec"]
        simple_evaluate_result["TS-IX"] = evaluate_result['in_exec']

        print('QM score: {}'.format(simple_evaluate_result["QM"]))
        print('QX score: {}'.format(simple_evaluate_result["QX"]))
        print('IM score: {}'.format(simple_evaluate_result["IM"]))
        print('IX score: {}'.format(simple_evaluate_result["IX"]))
        print('TS QX score: {}'.format(simple_evaluate_result["TS-QX"]))
        print('TS IX score: {}'.format(simple_evaluate_result["TS-IX"]))

        # save evaluate result
        save_json_file(os.path.join(args.results_path, "evaluate_result.json"), simple_evaluate_result)
        # save prediction result
        new_dir = "/".join(args.results_path.split("/")[:-1]).strip()
        if new_dir != "":
            os.makedirs(new_dir, exist_ok=True)
        assert len(detail_results) == len(user_messages)
        for i in range(len(user_messages)):
            detail_results[i]["user_message"] = user_messages[i]
        save_json_file(os.path.join(args.results_path, "predicted_sqls.json"), detail_results)
    else:
        raise " dataset name error"

if __name__ == '__main__':
    args = parse_args()
    main(args)
