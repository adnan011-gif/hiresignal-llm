"""
Supervised Fine-Tuning (SFT) Script for HireSignal.
Fine-tunes Qwen/Qwen2-1.5B-Instruct on candidate scoring and resume suggestions
using QLoRA quantization, PEFT, and W&B logging.
"""

import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Callback to capture and print final training stats
class LossPrinterCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_loss = None
        self.eval_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_loss = logs["loss"]
            if "eval_loss" in logs:
                self.eval_loss = logs["eval_loss"]

def formatting_prompts_func(example):
    """Formats raw instructions and job descriptions into the training prompt structure."""
    output_texts = []
    # If the input is a batch (list of values)
    if isinstance(example['instruction'], list):
        for inst, inp, resp in zip(example['instruction'], example['input'], example['response']):
            text = (
                f"### Instruction:\n{inst}\n\n"
                f"### Job Description:\n{inp}\n\n"
                f"### Response:\n{resp}"
            )
            output_texts.append(text)
    else:
        # Single example
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Job Description:\n{example['input']}\n\n"
            f"### Response:\n{example['response']}"
        )
        output_texts.append(text)
    return output_texts

def train_sft():
    print("=== Step 1: Initializing Weights & Biases ===")
    wandb.init(project="hiresignal-sft")

    # Paths to processed datasets
    train_path = "data/processed/train.json"
    val_path = "data/processed/val.json"

    print("=== Step 2: Loading Datasets ===")
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    print(f"Loaded train samples: {len(dataset['train'])}")
    print(f"Loaded validation samples: {len(dataset['validation'])}")

    print("=== Step 3: Loading Tokenizer ===")
    model_id = "Qwen/Qwen2-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Configure padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("=== Step 4: Configuring QLoRA Quantization ===")
    # Detect GPU/CUDA environment
    has_cuda = torch.cuda.is_available()
    
    if has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        device_map = "auto"
    else:
        print("WARNING: CUDA is not available. Running on CPU/MPS without 4-bit quantization.")
        bnb_config = None
        device_map = {"": "cpu"}

    print("=== Step 5: Loading Base Model ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    
    # Prepare model for 4-bit training if CUDA is active
    if has_cuda and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    print("=== Step 6: Configuring LoRA Adapter ===")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("=== Step 7: Configuring Training Arguments ===")
    training_args = TrainingArguments(
        output_dir="outputs/sft_model",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        logging_dir="outputs/sft_model/logs",
        report_to="wandb",
        fp16=has_cuda,  # Use fp16 training if running on CUDA
        use_mps_device=not has_cuda and torch.backends.mps.is_available(),
        remove_unused_columns=False
    )

    print("=== Step 8: Initializing SFTTrainer ===")
    # Initialize loss printer callback
    loss_printer = LossPrinterCallback()
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_prompts_func,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[loss_printer]
    )

    print("=== Step 9: Starting Supervised Fine-Tuning ===")
    train_result = trainer.train()

    print("=== Step 10: Saving Fine-Tuned LoRA Adapter ===")
    adapter_save_path = "outputs/sft_model/final_adapter"
    os.makedirs(adapter_save_path, exist_ok=True)
    trainer.model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    print(f"LoRA adapter successfully saved to {adapter_save_path}")

    # Evaluate the model
    print("=== Step 11: Running Validation Evaluation ===")
    eval_metrics = trainer.evaluate()
    
    print("\n=== Training Summary ===")
    final_train_loss = train_result.metrics.get("train_loss", loss_printer.train_loss)
    final_eval_loss = eval_metrics.get("eval_loss", loss_printer.eval_loss)
    print(f"Final Train Loss: {final_train_loss}")
    print(f"Final Eval Loss: {final_eval_loss}")
    
    wandb.finish()

if __name__ == "__main__":
    train_sft()
