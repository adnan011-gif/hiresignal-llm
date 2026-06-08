"""
Supervised Fine-Tuning (SFT) script for HireSignal.
Fine-tunes a base LLM on candidate scoring and resume feedback tasks.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

def train_sft():
    print("Initializing Supervised Fine-Tuning...")
    # TODO: Load model, dataset, configure LoRA, and initialize SFTTrainer
    pass

if __name__ == "__main__":
    train_sft()
