import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# 1. Mock Dataset Structure 
mock_data = {
    "messages": [
        [
            {"role": "system", "content": "You are a Booking.com Claims Expert."},
            {"role": "user", "content": "Hotel charged $50 for parking when booking said free."},
            {"role": "assistant", "content": "Reviewing voucher... Found discrepancy. Issuing $50 Booking Wallet credit immediately."}
        ]
    ] * 10000
}
dataset = Dataset.from_dict(mock_data)

model_id = "meta-llama/Llama-3-8b-Instruct" # 
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. Configure Quantization & Load Base Model (Using BFloat16 for stability)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. Define LoRA Configuration
peft_config = LoraConfig(
    r=16,                         # Rank: Higher rank captures more complexity but uses more memory
    lora_alpha=32,                # Scaling factor (usually 2x Rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. Training Parameters
training_args = TrainingArguments(
    output_dir="./booking_sft_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    bf16=True,
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    report_to="none" # Switch to "wandb" for live visualization tracking
)

# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset, 
    peft_config=peft_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# Start training run
# trainer.train()