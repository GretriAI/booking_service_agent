import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

model_id = "./booking_sft_lora" # Load the weights from the previous step

# 1. Prepare DPO Triplet Dataset
# Prompt, Chosen (Good outcome), Rejected (Bad outcome)
mock_dpo_data = {
    "prompt": ["Customer demands refund for dynamic pricing changing after look."],
    "chosen": ["I understand your frustration. While prices fluctuate based on real-time hotel occupancy, I can offer a gesture-of-goodwill voucher of $20."],
    "rejected": ["Prices change all the time. There is nothing we can do about this. Read our policy terms."]
}
dpo_dataset = Dataset.from_dict(mock_dpo_data)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-Instruct")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
# Reference model acts as a regulatory boundary so the model doesn't deviate too far
ref_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

dpo_config = DPOConfig(
    output_dir="./booking_dpo_results",
    beta=0.1,  # Implicit reward scale factor controlling distance from reference model
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6, # DPO requires an extremely small learning rate
    num_train_epochs=1,
    bf16=True,
    logging_steps=1
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    max_length=512,
    max_prompt_length=256
)

# Start alignment
# trainer.train()