import os, sys
import json
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import argparse
from tqdm import tqdm
from copy import deepcopy
from preprocess.sparc.sic_utils import MyClassifier, ColumnAndTableClassifierDataset, auc_metric, roc_auc_score, cls_metric
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, T5TokenizerFast, AutoTokenizer, BigBirdTokenizerFast
from transformers.trainer_utils import set_seed
from utils.common_utils import load_json_file, save_json_file
import random, copy
from tokenizers import AddedToken
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning schema item classifier.")

    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size.')
    parser.add_argument('--device', type=str, default="2",
                        help='the id of used GPU device.')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--model_name_or_path', type=str, default="ckpts/sic/sparc",
                        help='save path of best fine-tuned model on validation set.')
    parser.add_argument('--input_train_dataset_path', type=str, default="data/preprocessed_data/sparc/preprocessed_train.json",
                        help='path of pre-processed development dataset.')
    parser.add_argument('--input_dev_dataset_path', type=str, default="data/preprocessed_data/sparc/preprocessed_dev.json",
                        help='path of pre-processed development dataset.')
    parser.add_argument('--output_train_dataset_path', type=str, default="data/preprocessed_data/sparc/sft_train.json",
                        help='path of the output dataset (used in eval mode).')
    parser.add_argument('--output_dev_dataset_path', type=str, default="data/preprocessed_data/sparc/sft_dev.json",
                        help='path of the output dataset (used in eval mode).')
    parser.add_argument('--use_contents', default=True,
                        help='whether to integrate db contents into input sequence')
    parser.add_argument('--add_fk_info', default=True,
                        help='whether to add [FK] tokens into input sequence')
    parser.add_argument('--mode', type=str, default="train",
                        help='trian, eval or test.')
    parser.add_argument('--topk_table_num', type = int, default = 4,
                        help = 'we only remain topk_table_num tables in the ranked dataset (k_1 in the paper).')
    parser.add_argument('--topk_column_num', type = int, default = 5,
                        help = 'we only remain topk_column_num columns for each table in the ranked dataset (k_2 in the paper).')
    parser.add_argument('--noise_rate', type = float, default = 0.08,
                        help = 'the noise rate in the ranked training dataset (needed when the mode = "train")')
    parser.add_argument('--table_threshold', type=float, default=0.3,
                        help='the noise rate in the ranked training dataset (needed when the mode = "train")')
    parser.add_argument('--column_threshold', type=float, default=0.2,
                        help='the noise rate in the ranked training dataset (needed when the mode = "train")')
    parser.add_argument('--plm_name', type=str, default="roberta-large")
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--plm_hidden_state_dim', type=int, default=1024)
    parser.add_argument('--truncation', type=bool, default=False)
    parser.add_argument('--pooling_function', type=str, default="attention")
    parser.add_argument('--add_comment', action='store_true')
    parser.add_argument('--use_comment_enhanced', action='store_true')
    parser.add_argument('--use_column_enhanced', action='store_true')
    args = parser.parse_args()
    return args

def tokenizer_schema(opt, 
                     batch_size, 
                     tokenizer, 
                     batch_questions, 
                     batch_table_names, 
                     batch_table_labels, 
                     batch_column_infos, 
                     batch_column_labels, 
                     pre_turn_logits,
                     batch_turn_idx):

    batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []
    assert batch_size == 1
    if batch_turn_idx[0] > 0:
        table_pred_probs = pre_turn_logits["table_pred_probs"] 
        column_pred_probs = pre_turn_logits["column_pred_probs"]
        # print("")
    for batch_id in range(batch_size):
        input_tokens = [batch_questions[batch_id]]
        table_names_in_one_db = batch_table_names[batch_id]
        column_infos_in_one_db = batch_column_infos[batch_id]

        batch_column_number_in_each_table.append(
            [len(column_infos_in_one_table) for column_infos_in_one_table in column_infos_in_one_db])

        column_info_ids, table_name_ids = [], []

        for table_id, table_name in enumerate(table_names_in_one_db):
            input_tokens.append("|")
            input_tokens.append(table_name)
            table_name_ids.append(len(input_tokens) - 1)
            input_tokens.append(":")

            for col_id,column_info in enumerate(column_infos_in_one_db[table_id]):

                input_tokens.append(column_info)
                column_info_ids.append(len(input_tokens) - 1)
                input_tokens.append(",")

            input_tokens = input_tokens[:-1]

        batch_input_tokens.append(input_tokens)
        batch_column_info_ids.append(column_info_ids)
        batch_table_name_ids.append(table_name_ids)

    # notice: the trunction operation will discard some tables and columns that exceed the max length
    # print(input_tokens)
    tokenized_inputs = tokenizer(
        batch_input_tokens,
        return_tensors="pt",
        is_split_into_words=True,
        padding="max_length",
        max_length=opt.max_input_len,
        truncation=opt.truncation
    )

    batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []
    batch_aligned_table_labels, batch_aligned_column_labels = [], []

    # align batch_question_ids, batch_column_info_ids, and batch_table_name_ids after tokenizing
    for batch_id in range(batch_size):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_id)

        aligned_question_ids, aligned_table_name_ids, aligned_column_info_ids = [], [], []
        aligned_table_labels, aligned_column_labels = [], []

        # align question tokens
        for token_id, word_id in enumerate(word_ids):
            if word_id == 0:
                aligned_question_ids.append(token_id)

        # align table names
        for t_id, table_name_id in enumerate(batch_table_name_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if table_name_id == word_id:
                    temp_list.append(token_id)
            # if the tokenizer doesn't discard current table name
            if len(temp_list) != 0:
                aligned_table_name_ids.append(temp_list)
                aligned_table_labels.append(batch_table_labels[batch_id][t_id])

        # align column names
        for c_id, column_id in enumerate(batch_column_info_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if column_id == word_id:
                    temp_list.append(token_id)
            # if the tokenizer doesn't discard current column name
            if len(temp_list) != 0:
                aligned_column_info_ids.append(temp_list)
                aligned_column_labels.append(batch_column_labels[batch_id][c_id])

        batch_aligned_question_ids.append(aligned_question_ids)
        batch_aligned_table_name_ids.append(aligned_table_name_ids)
        batch_aligned_column_info_ids.append(aligned_column_info_ids)
        batch_aligned_table_labels.append(aligned_table_labels)
        batch_aligned_column_labels.append(aligned_column_labels)

    # update column number in each table (because some tables and columns are discarded)
    for batch_id in range(batch_size):
        if len(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_table_labels[batch_id]):
            batch_column_number_in_each_table[batch_id] = batch_column_number_in_each_table[batch_id][
                                                          : len(batch_aligned_table_labels[batch_id])]

        if sum(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_column_labels[batch_id]):
            truncated_column_number = sum(batch_column_number_in_each_table[batch_id]) - len(
                batch_aligned_column_labels[batch_id])
            batch_column_number_in_each_table[batch_id][-1] -= truncated_column_number

    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    batch_aligned_column_labels = [torch.LongTensor(column_labels) for column_labels in batch_aligned_column_labels]
    batch_aligned_table_labels = [torch.LongTensor(table_labels) for table_labels in batch_aligned_table_labels]

    # print("\n".join(tokenizer.batch_decode(encoder_input_ids, skip_special_tokens = True)))

    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
        batch_aligned_column_labels = [column_labels.cuda() for column_labels in batch_aligned_column_labels]
        batch_aligned_table_labels = [table_labels.cuda() for table_labels in batch_aligned_table_labels]

    return encoder_input_ids, encoder_input_attention_mask, \
        batch_aligned_column_labels, batch_aligned_table_labels, \
        batch_aligned_question_ids, batch_aligned_column_info_ids, \
        batch_aligned_table_name_ids, batch_column_number_in_each_table

def split_ids(ids, word_ids):
    # 使用字典来存储每个key对应的value列表
    result_dict = {}
    for key, value in zip(word_ids, ids):
        if key not in result_dict:
            result_dict[key] = []
        result_dict[key].append(value)

    # 将结果字典转换为二维列表
    result_list = [result_dict[key] for key in sorted(result_dict.keys())]
    return result_list

def evaluate_lists(lista, listb):
    # 检查listb的所有元素是否都在lista中
    if not all(item in lista for item in listb):
        return 1
    
    # 计算lista中有多少元素不在listb中
    redundant_elements = [item for item in lista if item not in listb]
    
    # 如果没有冗余元素，则得分为1
    if not redundant_elements:
        return 0
    
    # 否则，根据冗余元素与总元素的比例来调整得分
    # 例如，这里简单地使用非冗余元素占总数的比例作为得分
    non_redundant_count = len(lista) - len(redundant_elements)
    total_elements = len(lista)
    score = 1- (non_redundant_count / total_elements)
    return score

def prepare_input_and_output(args, ranked_data):

    question = ranked_data["question"]

    schema_sequence = ""
    for table_id in range(len(ranked_data["db_schema"])):
        table_name_original = ranked_data["db_schema"][table_id]["table_name_original"]
        # add table name
        schema_sequence += " | " + table_name_original + " : "
        column_info_list = []
        for column_id in range(len(ranked_data["db_schema"][table_id]["column_names_original"])):
            # extract column name
            column_name_original = ranked_data["db_schema"][table_id]["column_names_original"][column_id]
            db_contents = ranked_data["db_schema"][table_id]["db_contents"][column_id]
            # use database contents if opt.use_contents = True
            if args.use_contents and len(db_contents) != 0:
                column_contents = " , ".join(db_contents)
                column_info = table_name_original + "." + column_name_original + " ( " + column_contents + " ) "
            else:
                column_info = table_name_original + "." + column_name_original

            column_info_list.append(column_info)

        # add column names
        schema_sequence += " , ".join(column_info_list)

    if args.add_fk_info:
        for fk in ranked_data["fk"]:
            schema_sequence += " | " + fk["source_table_name_original"] + "." + fk["source_column_name_original"] + \
                               " = " + fk["target_table_name_original"] + "." + fk["target_column_name_original"]

    # remove additional spaces in the schema sequence
    while "  " in schema_sequence:
        schema_sequence = schema_sequence.replace("  ", " ")

    # input_sequence = question + schema sequence

    input_sequence = question + schema_sequence
    output_sequence = ranked_data["norm_sql"]

    return input_sequence, output_sequence

def prepare_batch_inputs_and_labels(opt, batch, tokenizer, pre_turn_logits):
    batch_size = len(batch)
    batch_questions = [data[0] for data in batch]
    batch_table_names = [data[1] for data in batch]
    batch_table_labels = [data[2] for data in batch]

    batch_column_infos = [data[3] for data in batch]
    batch_column_labels = [data[4] for data in batch]

    batch_table_comments = [data[5] for data in batch]
    batch_column_comment_infos = [data[6] for data in batch]

    batch_turn_idx = [data[7] for data in batch]

    encoder_input_ids, encoder_input_attention_mask, \
        batch_aligned_column_labels, batch_aligned_table_labels, \
        batch_aligned_question_ids, batch_aligned_column_info_ids, \
        batch_aligned_table_name_ids, batch_column_number_in_each_table = tokenizer_schema(opt,
                                                                                            batch_size, 
                                                                                           tokenizer, 
                                                                                           batch_questions, 
                                                                                           batch_table_names, 
                                                                                           batch_table_labels, 
                                                                                           batch_column_infos, 
                                                                                           batch_column_labels,
                                                                                           pre_turn_logits,
                                                                                           batch_turn_idx
                                                                                           )
    truncted_batch_table_comments = []
    truncted_batch_column_comment_infos = []
    for i, column_number_in_each_table in enumerate(batch_column_number_in_each_table):
        table_number = len(column_number_in_each_table)
        truncted_batch_table_comments.append(
            batch_table_comments[i][:table_number]
        )
        tmp = []
        for j, column_number in enumerate(column_number_in_each_table):
            if len(batch_column_comment_infos[i][j][:column_number]) > 0:
                tmp.append(
                    batch_column_comment_infos[i][j][:column_number]
                )
        if len(tmp) > 0:
            truncted_batch_column_comment_infos.append(
                tmp
            )
    batch_table_comment_ids, batch_table_comment_word_ids = [], []
    batch_column_comment_info_ids,batch_column_comment_info_word_ids  = [], []
    for table_comment in truncted_batch_table_comments:
        tokenized_inputs = tokenizer(
            table_comment,
            is_split_into_words=True,
            truncation=False
        )
        assert len(tokenized_inputs.word_ids(0)) == len(tokenized_inputs["input_ids"])
        batch_table_comment_ids.append(tokenized_inputs["input_ids"][1:-1])
        batch_table_comment_word_ids.append(tokenized_inputs.word_ids(0)[1:-1])
    
    for column_comment_in_one_db in truncted_batch_column_comment_infos:
        tmp1 = []
        tmp2 = []
        for column_comment_in_one_table in column_comment_in_one_db:
            if len(column_comment_in_one_table) == 0:
                print(column_comment_in_one_db)
                raise
            tokenized_inputs = tokenizer(
                                column_comment_in_one_table,
                                is_split_into_words=True,
                                truncation=False
                                )
            assert len(tokenized_inputs.word_ids(0)) == len(tokenized_inputs["input_ids"])
            tmp1.append(tokenized_inputs["input_ids"][1:-1])
            tmp2.append(tokenized_inputs.word_ids(0)[1:-1])
        if len(tmp1) > 0:
            batch_column_comment_info_ids.append(tmp1)
            batch_column_comment_info_word_ids.append(tmp2)
    input_dict = {}
    input_dict["encoder_input_ids"] = encoder_input_ids
    input_dict["encoder_input_attention_mask"] = encoder_input_attention_mask
    input_dict["batch_column_labels"] = batch_aligned_column_labels
    input_dict["batch_table_labels"] = batch_aligned_table_labels
    input_dict["batch_aligned_question_ids"] = batch_aligned_question_ids
    input_dict["batch_aligned_column_info_ids"] = batch_aligned_column_info_ids
    input_dict["batch_aligned_table_name_ids"] = batch_aligned_table_name_ids
    input_dict["batch_column_number_in_each_table"] = batch_column_number_in_each_table
    input_dict["batch_table_comment_ids"] = batch_table_comment_ids
    input_dict["batch_column_comment_info_ids"] = batch_column_comment_info_ids
    input_dict["batch_table_comment_word_ids"] = batch_table_comment_word_ids
    input_dict["batch_column_comment_info_word_ids"] = batch_column_comment_info_word_ids

    return input_dict

def _test(args):
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if "roberta-large" == args.plm_name:
        tokenizer_class = RobertaTokenizerFast
    elif "bigbird" in args.plm_name:
        tokenizer_class = BigBirdTokenizerFast
    elif "t5" in args.plm_name:
        tokenizer_class = T5TokenizerFast
    elif "gte-qwen" in args.plm_name:
        tokenizer_class = AutoTokenizer
    else:
        raise "plm type error"

    # load tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        add_prefix_space=True
    )

    dataset = ColumnAndTableClassifierDataset(
        args=args,
        dir_=args.input_dev_dataset_path,
        mode="eval"
    )

    dataloder = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x
    )

    # initialize model
    model = MyClassifier(
        args,
        vocab_size=len(tokenizer)
    )

    # load fine-tuned params
    model.load_state_dict(torch.load(args.model_name_or_path + "/dense_classifier.pt", map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    table_labels_for_auc, column_labels_for_auc = [], []
    table_pred_probs_for_auc, column_pred_probs_for_auc = [], []

    returned_table_pred_probs, returned_column_pred_probs = [], []

    pre_turn_logits = {}

    for batch in tqdm(dataloder):
        input_dict= prepare_batch_inputs_and_labels(args, batch, tokenizer, pre_turn_logits)
        with torch.no_grad():
            model_outputs = model(input_dict)

        for batch_id, table_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
            table_pred_probs = torch.nn.functional.softmax(table_logits, dim=1)
            table_pred_probs = table_pred_probs[:, 1].cpu().tolist()
            returned_table_pred_probs.append(table_pred_probs)
            pre_turn_logits["table_pred_probs"] = table_pred_probs
            table_pred_probs_for_auc.extend(table_pred_probs)
            table_labels_for_auc.extend(input_dict["batch_table_labels"][batch_id].cpu().tolist())

        for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
            column_number_in_each_table = input_dict["batch_column_number_in_each_table"][batch_id]
            column_pred_probs = torch.nn.functional.softmax(column_logits, dim=1)
            returned_column_pred_probs.append([column_pred_probs[:, 1].cpu().tolist()[
                                               sum(column_number_in_each_table[:table_id]):sum(
                                                   column_number_in_each_table[:table_id + 1])] \
                                               for table_id in range(len(column_number_in_each_table))])
            pre_turn_logits["column_pred_probs"] = [column_pred_probs[:, 1].cpu().tolist()[
                                               sum(column_number_in_each_table[:table_id]):sum(
                                                   column_number_in_each_table[:table_id + 1])] \
                                               for table_id in range(len(column_number_in_each_table))]
            column_pred_probs_for_auc.extend(column_pred_probs[:, 1].cpu().tolist())
            column_labels_for_auc.extend(input_dict["batch_column_labels"][batch_id].cpu().tolist())

    # calculate AUC score for table classification
    table_auc = auc_metric(table_labels_for_auc, table_pred_probs_for_auc)
    # calculate AUC score for column classification
    column_auc = auc_metric(column_labels_for_auc, column_pred_probs_for_auc)
    print("table auc:", table_auc)
    print("column auc:", column_auc)
    print("total auc:", table_auc + column_auc)

    table_predict_labels = [1 if prob > 0.5 else 0 for prob in table_pred_probs_for_auc]
    column_predict_labels = [1 if prob > 0.5 else 0 for prob in column_pred_probs_for_auc]
    cls_report = cls_metric(table_labels_for_auc, table_predict_labels)
    print(f"table cls report: {cls_report}")
    cls_report = cls_metric(column_labels_for_auc, column_predict_labels)
    print(f"column cls report: {cls_report}")

    return returned_table_pred_probs, returned_column_pred_probs

def filter_post_process(total_table_pred_probs, total_column_pred_probs, dataset):
    # record predicted probability
    truncated_data_info = []
    for data_id, data in enumerate(dataset):
        table_num = len(data["table_labels"])
        if table_num == len(total_table_pred_probs[data_id]):
            table_pred_probs = total_table_pred_probs[data_id]
        else:
            table_pred_probs = total_table_pred_probs[data_id] + [-1 for _ in range(
                table_num - len(total_table_pred_probs[data_id]))]

        truncated_table_ids = []
        column_pred_probs = []
        for table_id in range(table_num):
            if table_id >= len(total_column_pred_probs[data_id]):
                truncated_table_ids.append(table_id)
                column_pred_probs.append([-1 for _ in range(len(data["column_labels"][table_id]))])
                continue
            if len(total_column_pred_probs[data_id][table_id]) == len(data["column_labels"][table_id]):
                column_pred_probs.append(total_column_pred_probs[data_id][table_id])
            else:
                truncated_table_ids.append(table_id)
                truncated_column_num = len(data["column_labels"][table_id]) - len(
                    total_column_pred_probs[data_id][table_id])
                column_pred_probs.append(
                    total_column_pred_probs[data_id][table_id] + [-1 for _ in range(truncated_column_num)])

        data["column_pred_probs"] = column_pred_probs
        data["table_pred_probs"] = table_pred_probs

        if len(truncated_table_ids) > 0:
            truncated_data_info.append([data_id, truncated_table_ids])

    # additionally, we need to consider and predict discarded tables and columns
    while len(truncated_data_info) != 0:
        truncated_dataset = []
        for truncated_data_id, truncated_table_ids in truncated_data_info:
            print(dataset[truncated_data_id]["question"])
            truncated_data = deepcopy(dataset[truncated_data_id])
            truncated_data["db_schema"] = [truncated_data["db_schema"][table_id] for table_id in
                                           truncated_table_ids]
            truncated_data["table_labels"] = [truncated_data["table_labels"][table_id] for table_id in
                                              truncated_table_ids]
            truncated_data["column_labels"] = [truncated_data["column_labels"][table_id] for table_id in
                                               truncated_table_ids]
            truncated_data["table_pred_probs"] = [truncated_data["table_pred_probs"][table_id] for table_id in
                                                  truncated_table_ids]
            truncated_data["column_pred_probs"] = [truncated_data["column_pred_probs"][table_id] for table_id in
                                                   truncated_table_ids]

            truncated_dataset.append(truncated_data)

        with open("./processed_data/pre-processing/truncated_dataset.json", "w") as f:
            f.write(json.dumps(truncated_dataset, indent=2, ensure_ascii=False))

        args.dev_filepath = "./data/pre-processing/truncated_dataset.json"
        total_table_pred_probs, total_column_pred_probs = _test(args)

        for data_id, data in enumerate(truncated_dataset):
            table_num = len(data["table_labels"])
            if table_num == len(total_table_pred_probs[data_id]):
                table_pred_probs = total_table_pred_probs[data_id]
            else:
                table_pred_probs = total_table_pred_probs[data_id] + [-1 for _ in range(
                    table_num - len(total_table_pred_probs[data_id]))]

            column_pred_probs = []
            for table_id in range(table_num):
                if table_id >= len(total_column_pred_probs[data_id]):
                    column_pred_probs.append([-1 for _ in range(len(data["column_labels"][table_id]))])
                    continue
                if len(total_column_pred_probs[data_id][table_id]) == len(data["column_labels"][table_id]):
                    column_pred_probs.append(total_column_pred_probs[data_id][table_id])
                else:
                    truncated_column_num = len(data["column_labels"][table_id]) - len(
                        total_column_pred_probs[data_id][table_id])
                    column_pred_probs.append(
                        total_column_pred_probs[data_id][table_id] + [-1 for _ in range(truncated_column_num)])

            # fill the predicted probability into the dataset
            truncated_data_id = truncated_data_info[data_id][0]
            truncated_table_ids = truncated_data_info[data_id][1]
            for idx, truncated_table_id in enumerate(truncated_table_ids):
                dataset[truncated_data_id]["table_pred_probs"][truncated_table_id] = table_pred_probs[idx]
                dataset[truncated_data_id]["column_pred_probs"][truncated_table_id] = column_pred_probs[idx]

        # check if there are tables and columns in the new dataset that have not yet been predicted
        truncated_data_info = []
        for data_id, data in enumerate(dataset):
            table_num = len(data["table_labels"])

            truncated_table_ids = []
            for table_id in range(table_num):
                # the current table is not predicted
                if data["table_pred_probs"][table_id] == -1:
                    truncated_table_ids.append(table_id)
                # some columns in the current table are not predicted
                if data["table_pred_probs"][table_id] != -1 and -1 in data["column_pred_probs"][table_id]:
                    truncated_table_ids.append(table_id)

            if len(truncated_table_ids) > 0:
                truncated_data_info.append([data_id, truncated_table_ids])

        os.remove("./processed_data/data/pre-processing/truncated_dataset.json")
    return dataset


def generate_train_ranked_dataset(args):
    with open(args.input_train_dataset_path) as f:
        dataset = json.load(f)

    output_dataset = []
    for data_id, data in enumerate(tqdm(dataset)):
        ranked_data = dict()

        ranked_data["question"] = data["question"]
        ranked_data["norm_sql"] = data["norm_sql"]
        ranked_data["db_id"] = data["db_id"]
        ranked_data["db_schema"] = []

        # record ids of used tables
        used_table_ids = [idx for idx, label in enumerate(data["table_labels"]) if label == 1]
        topk_table_ids = copy.deepcopy(used_table_ids)

        if len(topk_table_ids) < args.topk_table_num:
            remaining_table_ids = [idx for idx in range(len(data["table_labels"])) if idx not in topk_table_ids]
            # if topk_table_num is large than the total table number, all tables will be selected
            if args.topk_table_num >= len(data["table_labels"]):
                topk_table_ids += remaining_table_ids
            # otherwise, we randomly select some unused tables
            else:
                randomly_sampled_table_ids = random.sample(remaining_table_ids,
                                                           args.topk_table_num - len(topk_table_ids))
                topk_table_ids += randomly_sampled_table_ids

        # add noise to the training set
        if random.random() < args.noise_rate:
            random.shuffle(topk_table_ids)

        for table_id in topk_table_ids:
            new_table_info = dict()
            new_table_info["table_name_original"] = data["db_schema"][table_id]["table_name_original"]
            # record ids of used columns
            used_column_ids = [idx for idx, column_label in enumerate(data["column_labels"][table_id]) if
                               column_label == 1]
            topk_column_ids = copy.deepcopy(used_column_ids)

            if len(topk_column_ids) < args.topk_column_num:
                remaining_column_ids = [idx for idx in range(len(data["column_labels"][table_id])) if
                                        idx not in topk_column_ids]
                # same as the selection of top-k tables
                if args.topk_column_num >= len(data["column_labels"][table_id]):
                    random.shuffle(remaining_column_ids)
                    topk_column_ids += remaining_column_ids
                else:
                    randomly_sampled_column_ids = random.sample(remaining_column_ids,
                                                                args.topk_column_num - len(topk_column_ids))
                    topk_column_ids += randomly_sampled_column_ids

            # add noise to the training set
            if random.random() < args.noise_rate and table_id in used_table_ids:
                random.shuffle(topk_column_ids)

            new_table_info["column_names_original"] = [data["db_schema"][table_id]["column_names_original"][column_id]
                                                       for column_id in topk_column_ids]
            new_table_info["db_contents"] = [data["db_schema"][table_id]["db_contents"][column_id] for column_id in
                                             topk_column_ids]

            ranked_data["db_schema"].append(new_table_info)

        # record foreign keys
        table_names_original = [table["table_name_original"] for table in data["db_schema"]]
        needed_fks = []
        for fk in data["fk"]:
            source_table_id = table_names_original.index(fk["source_table_name_original"])
            target_table_id = table_names_original.index(fk["target_table_name_original"])
            if source_table_id in topk_table_ids and target_table_id in topk_table_ids:
                needed_fks.append(fk)
        ranked_data["fk"] = needed_fks

        input_sequence, output_sequence = prepare_input_and_output(args, ranked_data)

        # record table_name_original.column_name_original for subsequent correction function during inference
        tc_original = []
        for table in ranked_data["db_schema"]:
            for column_name_original in ["*"] + table["column_names_original"]:
                tc_original.append(table["table_name_original"] + "." + column_name_original)

        output_dataset.append(
            {
                "db_id": data["db_id"],
                "input_sequence": input_sequence,
                "output_sequence": output_sequence,
                "tc_original": tc_original,
                "turn_idx": data["turn_idx"]
            }
        )
    return output_dataset

def generate_eval_ranked_dataset(args, dataset):

    table_coverage_state_list, column_coverage_state_list = [], []
    output_dataset = []
    for data_id, data in enumerate(tqdm(dataset)):
        ranked_data = dict()
        ranked_data["question"] = data["question"]
        ranked_data["norm_sql"] = data["norm_sql"]
        ranked_data["db_id"] = data["db_id"]
        ranked_data["db_schema"] = []

        table_pred_probs = list(map(lambda x: round(x, 4), data["table_pred_probs"]))
        # find ids of tables that have top-k probability
        # TODO: d7
        # ---- shcema filter
        topk_table_num = len([prob for prob in data["table_pred_probs"] if prob > args.table_threshold])
        topk_table_num = min(topk_table_num, args.topk_table_num)
        # print(f"topk_table_num: {topk_table_num}")
        # ---- shcema filter
        topk_table_ids = np.argsort(-np.array(table_pred_probs), kind="stable")[:topk_table_num].tolist()

        # if the mode == eval, we record some information for calculating the coverage
        if args.mode == "eval":
            used_table_ids = [idx for idx, label in enumerate(data["table_labels"]) if label == 1]
            table_coverage_state_list.append(evaluate_lists(topk_table_ids, used_table_ids))

            for idx in range(len(data["db_schema"])):
                used_column_ids = [idx for idx, label in enumerate(data["column_labels"][idx]) if label == 1]
                if len(used_column_ids) == 0:
                    continue
                column_pred_probs = list(map(lambda x: round(x, 2), data["column_pred_probs"][idx]))
                # TODO: d7
                # schema filter ---
                topk_column_num = len([prob for prob in data["column_pred_probs"][idx] if prob > args.column_threshold])
                topk_column_num = min(topk_column_num, args.topk_column_num)
                # schema filter --
                topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable")[:topk_column_num].tolist()
                column_coverage_state_list.append(evaluate_lists(topk_column_ids, used_column_ids))

        # record top-k1 tables and top-k2 columns for each table
        for table_id in topk_table_ids:
            new_table_info = dict()
            new_table_info["table_name_original"] = data["db_schema"][table_id]["table_name_original"]
            column_pred_probs = list(map(lambda x: round(x, 2), data["column_pred_probs"][table_id]))
            # schema filter ---
            topk_column_num = len([prob for prob in column_pred_probs if prob > args.column_threshold])
            topk_column_num = min(topk_column_num, args.topk_column_num)
            # schema filter --
            topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable")[:topk_column_num].tolist()

            new_table_info["column_names_original"] = [data["db_schema"][table_id]["column_names_original"][column_id]
                                                       for column_id in topk_column_ids]
            new_table_info["db_contents"] = [data["db_schema"][table_id]["db_contents"][column_id] for column_id in
                                             topk_column_ids]

            ranked_data["db_schema"].append(new_table_info)

        # record foreign keys among selected tables
        table_names_original = [table["table_name_original"] for table in data["db_schema"]]
        needed_fks = []
        for fk in data["fk"]:
            source_table_id = table_names_original.index(fk["source_table_name_original"])
            target_table_id = table_names_original.index(fk["target_table_name_original"])
            if source_table_id in topk_table_ids and target_table_id in topk_table_ids:
                needed_fks.append(fk)
        ranked_data["fk"] = needed_fks

        input_sequence, output_sequence = prepare_input_and_output(args, ranked_data)

        # record table_name_original.column_name_original for subsequent correction function during inference
        tc_original = []
        for table in ranked_data["db_schema"]:
            for column_name_original in table["column_names_original"] + ["*"]:
                tc_original.append(table["table_name_original"] + "." + column_name_original)

        output_dataset.append(
            {
                "db_id": data["db_id"],
                "input_sequence": input_sequence,
                "output_sequence": output_sequence,
                "tc_original": tc_original,
                "turn_idx": data["turn_idx"],
                "table_pred_probs": data["table_pred_probs"],
                "column_pred_probs": data["column_pred_probs"]
            }
        )

    print("Table top-{} coverage: {}".format(args.topk_table_num,
                                             sum(table_coverage_state_list) / len(table_coverage_state_list)))
    print("Column top-{} coverage: {}".format(args.topk_column_num,
                                              sum(column_coverage_state_list) / len(column_coverage_state_list)))
    return output_dataset

if __name__ == "__main__":
    args = parse_option()
    # Sparc train
    args.mode = "train"
    dataset = generate_train_ranked_dataset(args)
    save_json_file(args.output_train_dataset_path, dataset)
    # Sparc Dev
    args.mode = "eval"
    total_table_pred_probs, total_column_pred_probs = _test(args)
    dataset = load_json_file(args.input_dev_dataset_path)
    dataset = filter_post_process(total_table_pred_probs, total_column_pred_probs, dataset)
    dataset = generate_eval_ranked_dataset(args, dataset)
    save_json_file(args.output_dev_dataset_path, dataset)
