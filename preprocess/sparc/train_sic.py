import os, sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import transformers
import argparse
import torch.optim as optim
from tokenizers import AddedToken
from preprocess.sparc.sic_utils import cls_metric, auc_metric, MyClassifier, ClassifierLoss, ColumnAndTableClassifierDataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, T5TokenizerFast, AutoTokenizer, BigBirdTokenizerFast
from transformers.trainer_utils import set_seed

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning schema item classifier.")

    parser.add_argument('--train_bs', type=int, default=2,
                        help='input batch size.')
    parser.add_argument('--dev_bs', type=int, default=1,
                        help='input batch size.')
    parser.add_argument('--gradient_descent_step', type=int, default=2,
                        help='perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type=str, default="2",
                        help='the id of used GPU device.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='learning rate.')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='gamma parameter in the focal loss. Recommended: [0.0-2.0].')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='alpha parameter in the focal loss. Must between [0.0-1.0].')
    parser.add_argument('--epochs', type=int, default=128,
                        help='training epochs.')
    parser.add_argument('--patience', type=int, default=16,
                        help='patience step in early stopping. -1 means no early stopping.')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--save_path', type=str, default="",
                        help='save path of best fine-tuned model on validation set.')
    parser.add_argument('--train_filepath', type=str, default="",
                        help='path of pre-processed training dataset.')
    parser.add_argument('--dev_filepath', type=str, default="",
                        help='path of pre-processed development dataset.')
    parser.add_argument('--model_name_or_path', type=str, default="",
                        help='''pre-trained model name.''')
    parser.add_argument('--use_contents', default=True,
                        help='whether to integrate db contents into input sequence')
    parser.add_argument('--add_fk_info', default=True,
                        help='whether to add [FK] tokens into input sequence')
    parser.add_argument('--mode', type=str, default="train",
                        help='trian, eval or test.')
    parser.add_argument('--plm_name', type=str, default="roberta-large")
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--plm_hidden_state_dim', type=int, default=1024)
    parser.add_argument('--truncation', type=bool, default=False)
    parser.add_argument('--pooling_function', type=str, default="attention")
    parser.add_argument('--add_comment', action='store_true')
    parser.add_argument('--use_comment_enhanced', action='store_true')
    parser.add_argument('--use_column_enhanced', action='store_true')

    opt = parser.parse_args()

    return opt

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


def tokenizer_schema(opt, batch_size, tokenizer, batch_questions, batch_table_names, batch_table_labels, batch_column_infos, batch_column_labels):

    batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []
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

            for column_info in column_infos_in_one_db[table_id]:
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


def prepare_batch_inputs_and_labels(opt, batch, tokenizer):
    batch_size = len(batch)
    batch_questions = [data[0] for data in batch]
    batch_table_names = [data[1] for data in batch]
    batch_table_labels = [data[2] for data in batch]

    batch_column_infos = [data[3] for data in batch]
    batch_column_labels = [data[4] for data in batch]

    batch_table_comments = [data[5] for data in batch]
    batch_column_comment_infos = [data[6] for data in batch]

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
                                                                                           batch_column_labels)
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


def _train(opt):
    print(opt)
    set_seed(opt.seed)

    patience = opt.patience if opt.patience > 0 else float('inf')

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    if "roberta-large" == opt.plm_name:
        tokenizer_class = RobertaTokenizerFast
    elif "bigbird" in opt.plm_name:
        tokenizer_class = BigBirdTokenizerFast
    elif "t5" in opt.plm_name:
        tokenizer_class = T5TokenizerFast
    elif "gte-qwen" in opt.plm_name:
        tokenizer_class = AutoTokenizer
    else:
        raise "plm type error"

    tokenizer = tokenizer_class.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space=True
    )
    tokenizer.add_tokens(AddedToken("[FK]"))

    dir_ =opt.train_filepath
    train_dataset = ColumnAndTableClassifierDataset(
        args=opt,
        dir_=dir_,
        mode="train"
    )

    train_dataloder = DataLoader(
        train_dataset,
        batch_size=opt.train_bs,
        shuffle=True,
        collate_fn=lambda x: x
    )

    dir_ =opt.dev_filepath
    dev_dataset = ColumnAndTableClassifierDataset(
        args=opt,
        dir_=opt.dev_filepath,
        mode="train"
    )

    dev_dataloder = DataLoader(
        dev_dataset,
        batch_size=opt.dev_bs,
        shuffle=False,
        collate_fn=lambda x: x
    )

    # initialize model
    model = MyClassifier(
        args=opt,
        vocab_size=len(tokenizer)
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1 * opt.epochs * len(train_dataset) / opt.train_bs)
    # total training steps
    num_training_steps = int(opt.epochs * len(train_dataset) / opt.train_bs)
    # evaluate model for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider)
    num_checkpoint_steps = int(1.42857 * len(train_dataset) / opt.train_bs)

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=opt.learning_rate
    )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_score, early_stop_step, train_step = 0, 0, 0
    encoder_loss_func = ClassifierLoss(alpha=opt.alpha, gamma=opt.gamma)

    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch + 1}.")
        for batch in train_dataloder:
            model.train()
            train_step += 1
            input_dict= prepare_batch_inputs_and_labels(opt, batch, tokenizer)
            model_outputs = model(input_dict)

            loss = encoder_loss_func.compute_loss(
                model_outputs["batch_table_name_cls_logits"],
                input_dict["batch_table_labels"],
                model_outputs["batch_column_info_cls_logits"],
                input_dict["batch_column_labels"]
            )

            loss.backward()

            # update lr
            if scheduler is not None:
                scheduler.step()

            if train_step % opt.gradient_descent_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                print(f"loss={loss}")

            if train_step % num_checkpoint_steps == 0:
            # if True:
                print(f"At {train_step} training step, start an evaluation.")
                model.eval()

                table_labels_for_auc, column_labels_for_auc = [], []
                table_pred_probs_for_auc, column_pred_probs_for_auc = [], []

                for batch in dev_dataloder:
                    input_dict= prepare_batch_inputs_and_labels(opt, batch, tokenizer)
                    with torch.no_grad():
                        model_outputs = model(input_dict)

                    for batch_id, table_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
                        table_pred_probs = torch.nn.functional.softmax(table_logits, dim=1)

                        table_pred_probs_for_auc.extend(table_pred_probs[:, 1].cpu().tolist())
                        table_labels_for_auc.extend(input_dict["batch_table_labels"][batch_id].cpu().tolist())

                    for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
                        column_pred_probs = torch.nn.functional.softmax(column_logits, dim=1)

                        column_pred_probs_for_auc.extend(column_pred_probs[:, 1].cpu().tolist())
                        column_labels_for_auc.extend(input_dict["batch_column_labels"][batch_id].cpu().tolist())

                # calculate AUC score for table classification
                table_auc = auc_metric(table_labels_for_auc, table_pred_probs_for_auc)
                # calculate AUC score for column classification
                column_auc = auc_metric(column_labels_for_auc, column_pred_probs_for_auc)
                print("table AUC:", table_auc)
                print("column AUC:", column_auc)
                toral_auc_score = table_auc + column_auc
                print("total auc:", toral_auc_score)

                table_predict_labels = [1 if prob > 0.5 else 0 for prob in table_pred_probs_for_auc]
                column_predict_labels = [1 if prob > 0.5 else 0 for prob in column_pred_probs_for_auc]
                cls_report = cls_metric(table_labels_for_auc, table_predict_labels)
                print(f"table cls report: {cls_report}")
                cls_report = cls_metric(column_labels_for_auc, column_predict_labels)
                print(f"column cls report: {cls_report}")
                # save the best ckpt
                if toral_auc_score >= best_score:
                    best_score = toral_auc_score
                    os.makedirs(opt.save_path, exist_ok=True)
                    torch.save(model.state_dict(), opt.save_path + "/dense_classifier.pt")
                    model.plm_encoder.config.save_pretrained(save_directory=opt.save_path)
                    tokenizer.save_pretrained(save_directory=opt.save_path)
                    early_stop_step = 0
                else:
                    early_stop_step += 1

                print("early_stop_step:", early_stop_step)

            if early_stop_step >= patience:
                break

        if early_stop_step >= patience:
            print("Classifier training process triggers early stopping.")
            break

    print("best auc score:", best_score)

if __name__ == "__main__":
    opt = parse_option()
    _train(opt)