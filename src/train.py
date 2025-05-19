import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

with open("medical_dataset.json") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)

model_name = "microsoft/BioGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def preprocess(example):
    prompt = example["input"] + "\nAnswer: " + example["output"]
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="./biogpt_medical_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="no"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()
