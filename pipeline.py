from transformers import TrainingArguments
from functools import partial

training_args = TrainingArguments(
    output_dir="./booking_custom_metrics_run",
    evaluation_strategy="steps",
    eval_steps=50,                  # <--- Runs evaluation and your metrics loop every 50 steps
    logging_strategy="steps",
    logging_steps=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,   # <--- Batch size for the validation run
    bf16=True,
    report_to="none"                # Change to "wandb" to see graphs of your custom metrics
)

# Use a partial function to pass the tokenizer into the metrics calculation loop
metrics_callback = partial(compute_booking_metrics, tokenizer=tokenizer)

trainer = BookingCustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,       # <--- Validation set evaluated here
    compute_metrics=metrics_callback, # <--- Tracks custom text execution metrics
)

# trainer.train()