from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.data import Dataset

from transformers import AutoConfig, RobertaModel, T5EncoderModel, AutoModel, BigBirdModel


def cls_metric(ground_truth_labels, predict_labels):
    cls_report = classification_report(
        y_true=ground_truth_labels,
        y_pred=predict_labels,
        target_names=["negatives", "positives"],
        digits=4,
        output_dict=True
    )

    return cls_report


def auc_metric(ground_truth_labels, predict_probs):
    auc_score = roc_auc_score(ground_truth_labels, predict_probs)

    return auc_score


class MyClassifier(nn.Module):
    def __init__(
            self,
            args,
            vocab_size
    ):
        super(MyClassifier, self).__init__()
        if "roberta-large" == args.plm_name:
            model_class = RobertaModel
        elif "bigbird" in args.plm_name:
            model_class = BigBirdModel
        elif "t5" in args.plm_name:
            model_class = T5EncoderModel
        elif "gte-qwen" in args.plm_name:
            model_class = AutoModel
        else:
            "plm type error"
        if args.mode in ["eval", "test"]:
            # load config
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            # randomly initialize model's parameters according to the config
            if "roberta-large" == args.plm_name:
                self.plm_encoder = model_class(config)
            else:
                self.plm_encoder = model_class.from_config(config)
                
            # self.plm_encoder = model_class.from_pretrained(args.original_model_name_or_path)
        elif args.mode == "train":
            self.plm_encoder = model_class.from_pretrained(args.model_name_or_path)
            self.plm_encoder.resize_token_embeddings(vocab_size)
        else:
            raise ValueError()
        
        # for param in self.plm_encoder.parameters(): 
        #     param.requires_grad = False
        # 假设layers是一个ModuleList
        # layers = self.plm_encoder.layers
        
        # # 遍历前28层并冻结参数
        # for layer in layers[:28]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
            
        self.args = args
        # column cls head
        self.column_info_cls_head_linear1 = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim // 4)
        self.column_info_cls_head_linear2 = nn.Linear(args.plm_hidden_state_dim // 4, 2)

        # column bi-lstm layer
        self.column_info_bilstm = nn.LSTM(
            input_size=args.plm_hidden_state_dim,
            hidden_size=args.plm_hidden_state_dim // 2,
            num_layers=2,
            dropout=0,
            bidirectional=True
        )

        # linear layer after column bi-lstm layer
        self.column_info_linear_after_pooling = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim)

        # table cls head
        self.table_name_cls_head_linear1 = nn.Linear(args.plm_hidden_state_dim, 256)
        self.table_name_cls_head_linear2 = nn.Linear(256, 2)

        # table bi-lstm pooling layer
        self.table_name_bilstm = nn.LSTM(
            input_size=args.plm_hidden_state_dim,
            hidden_size=args.plm_hidden_state_dim // 2,
            num_layers=2,
            dropout=0,
            bidirectional=True
        )
        self.table_comment_bilstm = nn.LSTM(
            input_size=args.plm_hidden_state_dim,
            hidden_size=args.plm_hidden_state_dim // 2,
            num_layers=2,
            dropout=0,
            bidirectional=True
        )
        self.column_comment_bilstm = nn.LSTM(
            input_size=args.plm_hidden_state_dim,
            hidden_size=args.plm_hidden_state_dim // 2,
            num_layers=2,
            dropout=0,
            bidirectional=True
        )
        # linear layer after table bi-lstm layer
        self.table_name_linear_after_pooling = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim)
        self.table_comment_linear_after_pooling = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim)
        self.column_comment_linear_after_pooling = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim)

        # activation function
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # name and comment pooling linear
        self.table_name_linear = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim)
        self.table_comment_linear = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim)
        self.column_name_linear = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim)
        self.column_comment_linear = nn.Linear(args.plm_hidden_state_dim, args.plm_hidden_state_dim)

        self.table_linear_energy = nn.Linear(args.plm_hidden_state_dim * 2, args.plm_hidden_state_dim)
        self.column_linear_energy = nn.Linear(args.plm_hidden_state_dim * 2, args.plm_hidden_state_dim)

        # table-column cross-attention layer
        self.table_column_cross_attention_layer = nn.MultiheadAttention(embed_dim=args.plm_hidden_state_dim, num_heads=8)

        # dropout function, p=0.2 means randomly set 20% neurons to 0
        self.dropout = nn.Dropout(p=0.2)

    def table_column_cross_attention(
            self,
            table_name_embeddings_in_one_db,
            column_info_embeddings_in_one_db,
            column_number_in_each_table
    ):
        table_num = table_name_embeddings_in_one_db.shape[0]
        table_name_embedding_attn_list = []
        for table_id in range(table_num):
            table_name_embedding = table_name_embeddings_in_one_db[[table_id], :]
            column_info_embeddings_in_one_table = column_info_embeddings_in_one_db[
                                                  sum(column_number_in_each_table[:table_id]): sum(
                                                      column_number_in_each_table[:table_id + 1]), :]

            table_name_embedding_attn, _ = self.table_column_cross_attention_layer(
                table_name_embedding,
                column_info_embeddings_in_one_table,
                column_info_embeddings_in_one_table
            )

            table_name_embedding_attn_list.append(table_name_embedding_attn)

        # residual connection
        table_name_embeddings_in_one_db = table_name_embeddings_in_one_db + torch.cat(table_name_embedding_attn_list,
                                                                                      dim=0)
        # row-wise L2 norm
        table_name_embeddings_in_one_db = torch.nn.functional.normalize(table_name_embeddings_in_one_db, p=2.0, dim=1)

        return table_name_embeddings_in_one_db
    
    def name_comment_pooling_function(
            self,
            table_name_embeddings,
            table_comment_embeddings,
            column_name_embeddings,
            column_comment_embeddings
    ):
        if self.args.pooling_function == 'max':
            table_embedding_attn = torch.max(torch.stack([table_name_embeddings, table_comment_embeddings]), dim=0)[0]
            column_embedding_attn = torch.max(torch.stack([column_name_embeddings, column_comment_embeddings]), dim=0)[0]
        elif self.args.pooling_function == 'avg':
            table_embedding_attn = torch.mean(torch.stack([table_name_embeddings, table_comment_embeddings]), dim=0)
            column_embedding_attn = torch.mean(torch.stack([column_name_embeddings, column_comment_embeddings]), dim=0)
        elif self.args.pooling_function == 'attention':
            # process table
            table_name_embedding_attn = self.leakyrelu(self.table_name_linear(table_name_embeddings))
            table_comment_embedding_attn = self.leakyrelu(self.table_comment_linear(table_comment_embeddings))
            concatenated = torch.cat((table_name_embedding_attn, table_comment_embedding_attn), dim=1)
            energy = self.table_linear_energy(concatenated).squeeze()
            table_attn_weights = F.sigmoid(energy)
            table_attn_weights_complement = 1 - table_attn_weights
            # Apply attention weights to original embeddings
            table_attn_applied_name = table_attn_weights * table_name_embeddings
            table_attn_applied_comment = table_attn_weights_complement * table_comment_embeddings
            # Sum the weighted embeddings
            table_embedding_attn = torch.sum(torch.stack([table_attn_applied_name, table_attn_applied_comment]), dim=0)

            # process column
            column_name_embedding_attn = self.leakyrelu(self.column_name_linear(column_name_embeddings))
            column_comment_embedding_attn = self.leakyrelu(self.column_comment_linear(column_comment_embeddings))
            concatenated = torch.cat((column_name_embedding_attn, column_comment_embedding_attn), dim=1)
            energy = self.column_linear_energy(concatenated).squeeze()
            column_attn_weights = F.sigmoid(energy)
            column_attn_weights_complement = 1 - column_attn_weights
            # Apply attention weights to original embeddings
            column_attn_applied_name = column_attn_weights * column_name_embeddings
            column_attn_applied_comment = column_attn_weights_complement * column_comment_embeddings
            # Sum the weighted embeddings
            column_embedding_attn = torch.sum(torch.stack([column_attn_applied_name, column_attn_applied_comment]), dim=0)
        elif self.args.pooling_function == 'none':

            return table_name_embeddings, column_name_embeddings

        else:
            raise ValueError("Invalid pooling function. Choose from 'max', 'avg', or 'attention'.")

        # residual connection
        table_embeddings = table_name_embeddings + table_embedding_attn
        # row-wise L2 norm
        table_embeddings = torch.nn.functional.normalize(table_embeddings, p=2.0, dim=1)

        # residual connection
        column_embeddings = column_name_embeddings + column_embedding_attn
        # row-wise L2 norm
        column_embeddings = torch.nn.functional.normalize(column_embeddings, p=2.0, dim=1)

        return table_embeddings, column_embeddings
    
    def get_schema_word_embedding(self,
                                  aligned_table_name_ids, 
                                  aligned_column_info_ids, 
                                  sequence_embeddings):
        table_name_embedding_list, column_info_embedding_list = [], []
        # obtain table embedding via bi-lstm pooling + a non-linear layer
        for table_name_ids in aligned_table_name_ids:
            table_name_embeddings = sequence_embeddings[table_name_ids, :]

            # BiLSTM pooling
            output_t, (hidden_state_t, cell_state_t) = self.table_name_bilstm(table_name_embeddings)
            table_name_embedding = hidden_state_t[-2:, :].view(1, self.args.plm_hidden_state_dim)
            table_name_embedding_list.append(table_name_embedding)
        table_name_embeddings_in_one_db = torch.cat(table_name_embedding_list, dim=0)
        # non-linear mlp layer
        table_name_embeddings_in_one_db = self.leakyrelu(
            self.table_name_linear_after_pooling(table_name_embeddings_in_one_db))
        
        # obtain column embedding via bi-lstm pooling + a non-linear layer
        for column_info_ids in aligned_column_info_ids:
            column_info_embeddings = sequence_embeddings[column_info_ids, :]

            # BiLSTM pooling
            output_c, (hidden_state_c, cell_state_c) = self.column_info_bilstm(column_info_embeddings)
            column_info_embedding = hidden_state_c[-2:, :].view(1, self.args.plm_hidden_state_dim)
            column_info_embedding_list.append(column_info_embedding)
        column_info_embeddings_in_one_db = torch.cat(column_info_embedding_list, dim=0)
        # non-linear mlp layer
        column_info_embeddings_in_one_db = self.leakyrelu(
            self.column_info_linear_after_pooling(column_info_embeddings_in_one_db))
        
        return table_name_embeddings_in_one_db, column_info_embeddings_in_one_db

    def table_column_cls(
            self,
            input_dict
    ):
        encoder_input_ids = input_dict["encoder_input_ids"]
        encoder_input_attention_mask = input_dict["encoder_input_attention_mask"]
        batch_aligned_question_ids = input_dict["batch_aligned_question_ids"]
        batch_aligned_column_info_ids = input_dict["batch_aligned_column_info_ids"]
        batch_aligned_table_name_ids = input_dict["batch_aligned_table_name_ids"]
        batch_column_number_in_each_table = input_dict["batch_column_number_in_each_table"]

        batch_table_comment_ids = input_dict["batch_table_comment_ids"]
        batch_column_comment_info_ids = input_dict["batch_column_comment_info_ids"]
        batch_table_comment_word_ids = input_dict["batch_table_comment_word_ids"]
        batch_column_comment_info_word_ids = input_dict["batch_column_comment_info_word_ids"]


        batch_size = encoder_input_ids.shape[0]
        # encoder question table column
        encoder_output = self.plm_encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
            return_dict=True
        )  # encoder_output["last_hidden_state"].shape = (batch_size x seq_length x hidden_size)

        batch_table_name_cls_logits, batch_column_info_cls_logits = [], []

        # handle each data in current batch
        for batch_id in range(batch_size):
            column_number_in_each_table = batch_column_number_in_each_table[batch_id]
            sequence_embeddings = encoder_output["last_hidden_state"][batch_id, :, :]  # (seq_length x hidden_size)

            # obtain the embeddings of tokens in the question
            question_token_embeddings = sequence_embeddings[batch_aligned_question_ids[batch_id], :]

            # obtain table ids for each table
            aligned_table_name_ids = batch_aligned_table_name_ids[batch_id]
            # obtain column ids for each column
            aligned_column_info_ids = batch_aligned_column_info_ids[batch_id]


            table_name_embeddings_in_one_db, column_info_embeddings_in_one_db = self.get_schema_word_embedding(aligned_table_name_ids, 
                                                                                                               aligned_column_info_ids, 
                                                                                                               sequence_embeddings)
            if self.args.use_comment_enhanced:
                # comment
                table_comment_embeddings = self.plm_encoder(torch.tensor([batch_table_comment_ids[batch_id]]).to(encoder_input_ids.device))["last_hidden_state"][0]
                table_comment_embedding_list = []
                table_comment_word_ids=batch_table_comment_word_ids[batch_id]
                num_groups = max(table_comment_word_ids) + 1
                grouped_tensors = [[] for _ in range(num_groups)]
                for word_id, embedding in zip(table_comment_word_ids, table_comment_embeddings):
                    grouped_tensors[word_id].append(embedding)
                grouped_tensors = [torch.stack(group) if group else None for group in grouped_tensors]
                # obtain table embedding via bi-lstm pooling + a non-linear layer
                for embeddings in grouped_tensors:
                    # BiLSTM pooling
                    output_t, (hidden_state_t, cell_state_t) = self.table_comment_bilstm(embeddings)
                    table_comment_embedding = hidden_state_t[-2:, :].view(1, self.args.plm_hidden_state_dim)
                    table_comment_embedding_list.append(table_comment_embedding)
                table_comment_embeddings_in_one_db = torch.cat(table_comment_embedding_list, dim=0)
                # non-linear mlp layer
                table_comment_embeddings_in_one_db = self.leakyrelu(
                    self.table_comment_linear_after_pooling(table_comment_embeddings_in_one_db))
                
                assert table_comment_embeddings_in_one_db.shape[0] == table_name_embeddings_in_one_db.shape[0]


                column_comment_info_ids_in_one_db = batch_column_comment_info_ids[batch_id]
                column_comment_embedding_list = []
                for i, column_comment_info_ids_in_one_table in enumerate(column_comment_info_ids_in_one_db):
                    column_comment_embeddings = self.plm_encoder(torch.tensor([column_comment_info_ids_in_one_table]).to(encoder_input_ids.device))["last_hidden_state"][0]
                    column_comment_word_ids=batch_column_comment_info_word_ids[batch_id][i]
                    num_groups = max(column_comment_word_ids) + 1
                    grouped_tensors = [[] for _ in range(num_groups)]
                    for word_id, embedding in zip(column_comment_word_ids, column_comment_embeddings):
                        grouped_tensors[word_id].append(embedding)
                    grouped_tensors = [torch.stack(group) if group else None for group in grouped_tensors]
                    # obtain table embedding via bi-lstm pooling + a non-linear layer
                    for embeddings in grouped_tensors:
                        # BiLSTM pooling
                        output_t, (hidden_state_t, cell_state_t) = self.column_comment_bilstm(embeddings)
                        column_comment_embedding = hidden_state_t[-2:, :].view(1, self.args.plm_hidden_state_dim)
                        column_comment_embedding_list.append(column_comment_embedding)
                column_comment_embeddings_in_one_db = torch.cat(column_comment_embedding_list, dim=0)
                # non-linear mlp layer
                column_comment_embeddings_in_one_db = self.leakyrelu(
                    self.column_comment_linear_after_pooling(column_comment_embeddings_in_one_db))
                
                assert column_comment_embeddings_in_one_db.shape[0] == column_info_embeddings_in_one_db.shape[0]

                
                table_name_embeddings_in_one_db, column_info_embeddings_in_one_db = self.name_comment_pooling_function(table_name_embeddings_in_one_db,
                                                                                                                        table_comment_embeddings_in_one_db,
                                                                                                                        column_info_embeddings_in_one_db,
                                                                                                                        column_comment_embeddings_in_one_db
                                                                                                                        )
            if self.args.use_column_enhanced:
                # table-column (tc) cross-attention
                table_name_embeddings_in_one_db = self.table_column_cross_attention(
                    table_name_embeddings_in_one_db,
                    column_info_embeddings_in_one_db,
                    column_number_in_each_table
                )
        
            # calculate table 0-1 logits
            table_name_embeddings_in_one_db = self.table_name_cls_head_linear1(table_name_embeddings_in_one_db)
            table_name_embeddings_in_one_db = self.dropout(self.leakyrelu(table_name_embeddings_in_one_db))
            table_name_cls_logits = self.table_name_cls_head_linear2(table_name_embeddings_in_one_db)

            # calculate column 0-1 logits
            column_info_embeddings_in_one_db = self.column_info_cls_head_linear1(column_info_embeddings_in_one_db)
            column_info_embeddings_in_one_db = self.dropout(self.leakyrelu(column_info_embeddings_in_one_db))
            column_info_cls_logits = self.column_info_cls_head_linear2(column_info_embeddings_in_one_db)

            batch_table_name_cls_logits.append(table_name_cls_logits)
            batch_column_info_cls_logits.append(column_info_cls_logits)

        return batch_table_name_cls_logits, batch_column_info_cls_logits

    def forward(
            self,
            input_dict
    ):
        batch_table_name_cls_logits, batch_column_info_cls_logits \
            = self.table_column_cls(
            input_dict
        )

        return {
            "batch_table_name_cls_logits": batch_table_name_cls_logits,
            "batch_column_info_cls_logits": batch_column_info_cls_logits
        }


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.5, reduction=None):
        super(FocalLoss, self).__init__()

        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        assert input_tensor.shape[0] == target_tensor.shape[0]

        prob = F.softmax(input_tensor, dim=-1)
        log_prob = torch.log(prob + 1e-8)

        loss = F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

        return loss


class ClassifierLoss():
    def __init__(self, alpha, gamma):
        weight = torch.FloatTensor([1 - alpha, alpha])
        if torch.cuda.is_available():
            weight = weight.cuda()

        self.focal_loss = FocalLoss(
            weight=weight,
            gamma=gamma,
            reduction='mean'
        )

        # self.ce_loss = nn.CrossEntropyLoss(weight = weight, reduction = "mean")

    def compute_batch_loss(self, batch_logits, batch_labels, batch_size):
        loss = 0
        for logits, labels in zip(batch_logits, batch_labels):
            loss += self.focal_loss(logits, labels)

        return loss / batch_size

    def compute_loss(
            self,
            batch_table_name_cls_logits,
            batch_table_labels,
            batch_column_info_cls_logits,
            batch_column_labels
    ):
        batch_size = len(batch_table_labels)

        table_loss = self.compute_batch_loss(batch_table_name_cls_logits, batch_table_labels, batch_size)
        column_loss = self.compute_batch_loss(batch_column_info_cls_logits, batch_column_labels, batch_size)

        return table_loss + column_loss


class ColumnAndTableClassifierDataset(Dataset):
    def __init__(
            self,
            args,
            dir_: str = None,
            mode="train"
    ):
        super(ColumnAndTableClassifierDataset, self).__init__()

        self.questions: list[str] = []

        self.all_column_infos: list[list[list[str]]] = []
        self.all_column_labels: list[list[list[int]]] = []

        self.all_column_comment_infos: list[list[list[str]]] = []

        self.all_table_names: list[list[str]] = []
        self.all_table_comments: list[list[str]] = []
        self.all_table_labels: list[list[int]] = []
        self.turn_idx: list[int] = []
        self.args = args

        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        pre_tabcol_labels_in_one_db = []
        pre_tab_labels_in_one_db = []
        for data in dataset:
            column_names_in_one_db = []
            column_names_original_in_one_db = []
            column_comment_in_one_db = []
            extra_column_info_in_one_db = []
            column_labels_in_one_db = []


            table_names_in_one_db = []
            table_comment_in_one_db = []
            table_names_original_in_one_db = []
            table_labels_in_one_db = []
            tabcol_labels_in_one_db = []

            for table_id in range(len(data["db_schema"])):
                column_names_original_in_one_db.append(data["db_schema"][table_id]["column_names_original"])
                table_names_original_in_one_db.append(data["db_schema"][table_id]["table_name_original"])

                column_comment_in_one_db.append(data["db_schema"][table_id]["column_comments"])

                table_names_in_one_db.append(data["db_schema"][table_id]["table_name"])
                table_labels_in_one_db.append(data["table_labels"][table_id])

                table_comment_in_one_db.append(data["db_schema"][table_id]["table_comment"])

                column_names_in_one_db.append(data["db_schema"][table_id]["column_names"])
                column_labels_in_one_db += data["column_labels"][table_id]
                tabcol_labels_in_one_db.append(data["column_labels"][table_id])
                extra_column_info = ["" for _ in range(len(data["db_schema"][table_id]["column_names"]))]
                if args.use_contents:
                    contents = data["db_schema"][table_id]["db_contents"]
                    for column_id, content in enumerate(contents):
                        if len(content) != 0:
                            extra_column_info[column_id] += " , ".join(content)
                extra_column_info_in_one_db.append(extra_column_info)

            if args.add_fk_info:
                table_column_id_list = []
                # add a [FK] identifier to foreign keys
                for fk in data["fk"]:
                    source_table_name_original = fk["source_table_name_original"]
                    source_column_name_original = fk["source_column_name_original"]
                    target_table_name_original = fk["target_table_name_original"]
                    target_column_name_original = fk["target_column_name_original"]

                    if source_table_name_original in table_names_original_in_one_db:
                        source_table_id = table_names_original_in_one_db.index(source_table_name_original)
                        source_column_id = column_names_original_in_one_db[source_table_id].index(
                            source_column_name_original)
                        if [source_table_id, source_column_id] not in table_column_id_list:
                            table_column_id_list.append([source_table_id, source_column_id])

                    if target_table_name_original in table_names_original_in_one_db:
                        target_table_id = table_names_original_in_one_db.index(target_table_name_original)
                        target_column_id = column_names_original_in_one_db[target_table_id].index(
                            target_column_name_original)
                        if [target_table_id, target_column_id] not in table_column_id_list:
                            table_column_id_list.append([target_table_id, target_column_id])

                for table_id, column_id in table_column_id_list:
                    if extra_column_info_in_one_db[table_id][column_id] != "":
                        extra_column_info_in_one_db[table_id][column_id] += " , [FK]"
                    else:
                        extra_column_info_in_one_db[table_id][column_id] += "[FK]"
            pre_tab_labels_in_one_db = table_labels_in_one_db
            pre_tabcol_labels_in_one_db = tabcol_labels_in_one_db

            # column_info = column name + extra column info
            column_infos_in_one_db = []
            for table_id in range(len(table_names_in_one_db)):
                column_infos_in_one_table = []
                for column_name, extra_column_info in zip(column_names_in_one_db[table_id],
                                                          extra_column_info_in_one_db[table_id]):
                    if len(extra_column_info) != 0:
                        column_infos_in_one_table.append(column_name + " ( " + extra_column_info + " ) ")
                    else:
                        column_infos_in_one_table.append(column_name)
                column_infos_in_one_db.append(column_infos_in_one_table)


            self.questions.append(data["question"])

            self.turn_idx.append(data["turn_idx"])

            self.all_table_names.append(table_names_in_one_db)
            self.all_table_labels.append(table_labels_in_one_db)

            self.all_column_infos.append(column_infos_in_one_db)
            self.all_column_labels.append(column_labels_in_one_db)

            self.all_column_comment_infos.append(column_comment_in_one_db)
            self.all_table_comments.append(table_comment_in_one_db)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]

        table_names_in_one_db = self.all_table_names[index]
        table_labels_in_one_db = self.all_table_labels[index]

        table_comments_in_one_db = self.all_table_comments[index]

        column_infos_in_one_db = self.all_column_infos[index]
        column_labels_in_one_db = self.all_column_labels[index]

        column_comment_infos_in_one_db = self.all_column_comment_infos[index]

        turn_idx = self.turn_idx[index]


        return question, table_names_in_one_db, table_labels_in_one_db, column_infos_in_one_db, column_labels_in_one_db, table_comments_in_one_db, column_comment_infos_in_one_db, turn_idx


class Text2SQLDataset(Dataset):
    def __init__(
            self,
            dir_: str,
            mode: str
    ):
        super(Text2SQLDataset).__init__()

        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []
        self.db_ids: list[str] = []
        self.all_tc_original: list[list[str]] = []

        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            self.input_sequences.append(data["input_sequence"])
            self.db_ids.append(data["db_id"])
            self.all_tc_original.append(data["tc_original"])

            if self.mode == "train":
                self.output_sequences.append(data["output_sequence"])
            elif self.mode in ["eval", "test"]:
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.input_sequences[index], self.output_sequences[index], self.db_ids[index], self.all_tc_original[
                index]
        elif self.mode in ['eval', "test"]:
            return self.input_sequences[index], self.db_ids[index], self.all_tc_original[index]