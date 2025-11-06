from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Step 1: Load base model + tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Wrap with LoRA adapter
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # typical for LLaMA-style models
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)

# Step 3: Load your dataset (replace with your JSONL or dataset path)
dataset = load_dataset("json", data_files="dataset.jsonl")

# Step 4: Training setup
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=True,
)

# Step 5a: Preprocess dataset into model inputs
def preprocess_function(examples):
    # Concatenate prompt + completion into one string
    text = [p + " " + c for p, c in zip(examples["prompt"], examples["completion"])]
    # Tokenize
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    # Labels are the same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply preprocessing - 5b
tokenized_dataset = dataset.map(preprocess_function, batched=True)




# Step 5c: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Step 6: Train + save adapter
trainer.train()
model.save_pretrained("allie_finetuned")
