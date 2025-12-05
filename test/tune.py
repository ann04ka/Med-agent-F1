import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel)

from datasets import Dataset
import warnings

warnings.filterwarnings('ignore')

OUTPUT_PATH = ''
MODEL_OUTPUT_DIR = './saiga_medical_final'
LORA_OUTPUT_DIR = './saiga_medical_lora'

MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B"

TRAIN_CONFIG = {
    'num_epochs': 3,
    'batch_size': 2,
    'gradient_accumulation_steps': 8,
    'learning_rate': 2e-4,
    'max_length': 512,
    'warmup_steps': 100,
    'logging_steps': 10,
    'save_steps': 100,
    'eval_steps': 100,
}

LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

try:
    train_df = pd.read_csv(f'{OUTPUT_PATH}medical_training_data.csv')
    print(f"Загружено {len(train_df)} примеров")
    print(f"Колонки: {train_df.columns.tolist()}")

    if 'text' not in train_df.columns:
        raise ValueError("В датасете отсутствует колонка 'text'!")

    train_df = train_df.dropna(subset=['text'])
    train_df = train_df[train_df['text'].str.strip().str.len() > 20]
    print(f"После очистки: {len(train_df)} примеров")

except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    print("Убедитесь что файл medical_training_data.csv находится в текущей директории")
    raise

print("\n📝 Первые 2 примера:")
for i in range(min(2, len(train_df))):
    print(f"\n{i + 1}. {train_df.iloc[i]['text'][:300]}...")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_CONFIG['r'],
    lora_alpha=LORA_CONFIG['lora_alpha'],
    target_modules=LORA_CONFIG['target_modules'],
    lora_dropout=LORA_CONFIG['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=TRAIN_CONFIG['max_length'],
        padding="max_length",
        return_tensors=None
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

dataset = Dataset.from_pandas(train_df[['text']])

train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"Train: {len(train_dataset)} примеров")
print(f"Validation: {len(eval_dataset)} примеров")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Токенизация train"
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Токенизация validation"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=TRAIN_CONFIG['num_epochs'],
    per_device_train_batch_size=TRAIN_CONFIG['batch_size'],
    per_device_eval_batch_size=TRAIN_CONFIG['batch_size'],
    gradient_accumulation_steps=TRAIN_CONFIG['gradient_accumulation_steps'],
    learning_rate=TRAIN_CONFIG['learning_rate'],

    optim="paged_adamw_32bit",
    fp16=False,
    bf16=True,

    logging_dir=f"{MODEL_OUTPUT_DIR}/logs",
    logging_steps=TRAIN_CONFIG['logging_steps'],
    logging_strategy="steps",

    save_strategy="steps",
    save_steps=TRAIN_CONFIG['save_steps'],
    save_total_limit=3,

    eval_strategy="steps",
    eval_steps=TRAIN_CONFIG['eval_steps'],

    warmup_steps=TRAIN_CONFIG['warmup_steps'],
    max_grad_norm=0.3,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

try:
    train_result = trainer.train()

    print(f"train loss: {train_result.training_loss:.4f}")
    print(f"Время обучения: {train_result.metrics['train_runtime']:.2f} сек")

except Exception as e:
    print(f"\nОшибка во время обучения: {e}")
    raise

model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model_merged = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    model_merged = model_merged.merge_and_unload()

    model_merged.save_pretrained(f"{MODEL_OUTPUT_DIR}_merged")
    tokenizer.save_pretrained(f"{MODEL_OUTPUT_DIR}_merged")

except Exception as e:
    print(f"Не удалось сохранить merged модель: {e}")