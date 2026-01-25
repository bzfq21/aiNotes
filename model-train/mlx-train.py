from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import load_dataset

# Load any HuggingFace model (1B model for quick start)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Qwen3-0.6B-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Load a dataset (or create your own)
train_dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")
eval_dataset = load_dataset("yahma/alpaca-cleaned", split="train[0:100]")

# Train with SFTTrainer (same API as TRL!)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_steps=50,
    ),
)
trainer.train()

model.save_pretrained("lora_model")  # Adapters only
# model.save_pretrained_merged("merged", tokenizer)  # Full model
# model.save_pretrained_gguf("model", tokenizer)  # GGUF (see note below)