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