import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from datasets import load_dataset
from sql_metadata import Parser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import argparse

def parse_config():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")

    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--num_train_epochs', type=int, default=6)
    parser.add_argument('--bf16', type=bool, default=True)
    parser.add_argument('--overwrite_output_dir', type=bool, default=True)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--gradient_checkpointing', type=bool, default=True)
    parser.add_argument('--evaluation_strategy', type=str, default="steps")
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--group_by_length', type=bool, default=True)
    parser.add_argument('--auto_find_batch_size', type=bool, default=False)
    parser.add_argument('--save_steps', type=int, default=300)
    parser.add_argument('--logging_steps', type=int, default=300)
    parser.add_argument('--load_best_model_at_end', type=bool, default=False)
    parser.add_argument('--packing', type=bool, default=False)
    parser.add_argument('--save_total_limit', type=int, default=6)
    parser.add_argument('--neftune_noise_alpha', type=int, default=5)
    parser.add_argument('--report_to', type=list, default=[])
    parser.add_argument('--max_seq_length', type=int, default=3000)
    parser.add_argument('--output_dir', type=str, default="ckpts/llm_ckpts/spider/left")
    parser.add_argument('--use_dora', action='store_true')
    parser.add_argument('--model_path', type=str, default="huggingface/deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument('--train_file', type=str, default="data/preprocessed_data/spider/sft_spider_train_left.json")
    parser.add_argument('--dev_file', type=str, default="data/preprocessed_data/spider/sft_spider_train_right.json")
    parser.add_argument('--llm_name', type=str, default="deepseek")
    parser.add_argument('--dataset_name', type=str, default="spider")
    parser.add_argument('--instruct_version', type=int, default=1)

    config = parser.parse_args()

    return config

INSTRUCT = """
You are a SQL query generator that converts multi-round questions along with associated database schema information into corresponding SQL statements. The multi-round questions will be concatenated using the '&' symbol, and you should generate the SQL statement that answers the current round of the question based on the provided database schema.

Each database schema is provided in the following format:

Table name : Column name1, Column name2 Different tables are separated by the '|' symbol, and the order of table names and column names is relevant to the current question; those appearing earlier are more closely related.
"""

def main(config):

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype = torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, 
        trust_remote_code=True,
        )
    model_name = config.model_path.split('/')[-1].lower()
    print(f"use model: {model_name}")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def formatting_prompts_func(training_dataset):
        output_texts = []
        for i in range(len(training_dataset['input_sequence'])):
            if config.dataset_name in ["spider", "sparc", "cosql"]:
                input_sequence = training_dataset['input_sequence'][i]
                question = input_sequence.split(" | ")[0]
                database_schema = " | ".join(input_sequence.split(" | ")[1:])

                if training_dataset["turn_idx"][i] > 0:
                    previous_query = training_dataset['output_sequence'][i-1]
                    previous_query = "SQL corresponding to the previous round of questions: " + previous_query
                else:
                    previous_query = ""
                instruct = INSTRUCT + "\n" + previous_query + "\n" + "The task is to take the multi-round question and the database schema as input and output an SQL statement that answers the question correctly."
                user_message = instruct + "\n" + "database schema: " + database_schema + "\n" + "question: " + question

                output_sequence = training_dataset['output_sequence'][i]
                if model_name ==  "deepseek-coder-v2-lite-instruct":
                    assitant_message = f"""
                    ```sql
                    {output_sequence} ;
                    ```
                    <｜end▁of▁sentence｜>
                    """
                else:
                    assitant_message = f"""
                    ```sql
                    {output_sequence} ;
                    ```
                    <|EOT|>
                    """
                # print(f"input form: {user_message}")
                # print(f"output form: {assitant_message}")
            else: 
                raise "dataset name error"
            
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assitant_message},
            ]
            text = tokenizer.apply_chat_template(messages,tokenize=False)
            output_texts.append(text)
            # print(text)
        return output_texts

    data_files = {"train": config.train_file, "validation": config.dev_file}
    dataset = load_dataset('json', data_files=data_files)
    if "deepseek-coder-6.7b-instruct" == model_name:
        response_template = "### Response:"  # deepseek
        print("use template:### Response: ")
    elif "mistral" in model_name:
        response_template = "[/INST]"
        print("use template:[/INST]")
    elif "deepseek-coder-v2-lite-instruct" == model_name:
        response_template = "<｜begin▁of▁sentence｜>"
    elif "qwen2.5-coder-7b-instruct" == model_name:
         response_template = "<|im_start|>"
    else:
        raise "llm name error"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    peft_config = LoraConfig(
        use_dora=config.use_dora,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head"
        ],
        task_type=TaskType.CAUSAL_LM,
    )

    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=config.overwrite_output_dir,
        num_train_epochs=config.num_train_epochs,
        load_best_model_at_end=config.load_best_model_at_end,
        per_device_train_batch_size=config.per_device_train_batch_size,
        evaluation_strategy=config.evaluation_strategy,
        max_grad_norm=config.max_grad_norm,
        auto_find_batch_size=config.auto_find_batch_size,
        save_total_limit=config.save_total_limit,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to=config.report_to,
        neftune_noise_alpha=config.neftune_noise_alpha
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=config.max_seq_length,
        packing=config.packing
    )
    trainer.train()
    print(f"training successful, saving model to {config.output_dir} now.")
    os.makedirs(config.output_dir, exist_ok=True)
    trainer.model.save_pretrained(config.output_dir)

if __name__ == '__main__':
    config = parse_config()
    main(config)