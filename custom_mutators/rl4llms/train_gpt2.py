from pathlib import Path
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
import os.path as path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig,LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def train_tokenizer():
    # Train the tokeniser

    path_to_seeds = path.abspath(path.join(__file__, "../../../FirmWire/out_lte_rcc_G970F/default/queue"))
    seeds = []

    for file in Path(path_to_seeds).glob("*"):
        if not path.isfile(file):
            continue
        with open(file, 'rb') as f:
            seeds.append(str(f.read())[2:-1])
        with open('byte_dataset.txt', 'a') as f:
            f.write(seeds[-1])
            f.write('\n')
    print(seeds[0])
    print(seeds[-1])
    print(len(seeds))


    tokenizer =  AutoTokenizer.from_pretrained("gpt2")
    tokenizer = tokenizer.train_new_from_iterator(seeds, 52000)

    return tokenizer

#tokenizer = AutoTokenizer.from_pretrained('byte_gpt2')

def tokenize(element):
    context_length = 128
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


#tokenized_datasets = raw_datasets.map(
#    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
#)

if __name__ == "__main__":
    save_dir = 'byte_gpt2'
    #tokenizer = train_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained('byte_gpt2')#tokenizer = train_tokenizer()
    context_length = 128
    #tokenized_datasets = raw_datasets.map(
    #    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    #)
    tokenized_datasets = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path='byte_dataset.txt',
        block_size=128,
    )
    print(f"Dataset size: {len(tokenized_datasets)}")
    train_set, val_set = torch.utils.data.random_split(tokenized_datasets, [int(len(tokenized_datasets)*0.9), len(tokenized_datasets) - int(len(tokenized_datasets)*.9)])
    
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=val_set,
    )
    trainer.train()
    trainer.save_model(save_dir)
    #tokenizer = train_tokenizer()
    #tokenizer.save_pretrained(save_dir)
    #tokenizer.save_model(save_dir)

