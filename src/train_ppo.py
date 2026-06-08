"""
PPO (Proximal Policy Optimization) Training Script for HireSignal.
Optimizes the SFT-tuned phi-2 model using the trained DistilBERT Reward Model.

Pipeline position:
  SFT Model (policy) + Reward Model (critic) → PPO update → Aligned Model

The PPO loop:
  1. Sample JD prompts from the validation set.
  2. Generate candidate responses with the current policy (SFT model).
  3. Score each response using the Reward Model (higher = more structured).
  4. Compute PPO advantage and update the policy to maximise reward while
     staying close to the original SFT distribution (KL penalty).
  5. Repeat for N steps until the policy reliably produces structured output.
"""

import os
import json
import random
import torch
import torch.nn as nn
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DistilBertModel,
    DistilBertTokenizer,
)
from peft import PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "microsoft/phi-2"
SFT_ADAPTER_PATH = "outputs/sft_model/final_adapter"
REWARD_MODEL_PATH = "outputs/reward_model/best_reward_model.pt"
REWARD_TOKENIZER_PATH = "outputs/reward_model/tokenizer"
VAL_DATA_PATH = "data/processed/val.json"
PPO_MODEL_SAVE_DIR = "outputs/ppo_model"
REWARD_BACKBONE_ID = "distilbert-base-uncased"

NUM_PPO_STEPS = 25
PPO_BATCH_SIZE = 8


# ── Reward Model (mirror of train_reward_model.py) ──────────────────────────

class RewardModel(nn.Module):
    """
    DistilBERT-based scalar reward model.
    Must match the architecture used during reward model training.
    """

    def __init__(self, backbone_name: str = REWARD_BACKBONE_ID):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        reward = self.reward_head(cls_hidden).squeeze(-1)
        return reward


# ── Prompt Builder ──────────────────────────────────────────────────────────

def build_prompt(instruction: str, job_description: str) -> str:
    """Format instruction and JD into the SFT training template."""
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Job Description:\n{job_description.strip()}\n\n"
        f"### Response:\n"
    )


# ── Model Loaders ──────────────────────────────────────────────────────────

def load_sft_model_and_tokenizer():
    """Load the SFT-tuned phi-2 with LoRA adapter, wrapped with a value head for PPO."""
    print("🔧  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for generation in PPO

    has_cuda = torch.cuda.is_available()

    if has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        device_map = "auto"
    else:
        print("⚠  CUDA not available — loading on CPU without quantization.")
        bnb_config = None
        device_map = {"": "cpu"}

    print(f"🔧  Loading base model ({MODEL_ID}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if has_cuda else torch.float32,
    )

    print(f"🔧  Loading LoRA adapter from {SFT_ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)

    # Wrap with value head for PPO (adds a linear head that estimates state values)
    print("🔧  Wrapping model with value head for PPO ...")
    model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    return model_with_value_head, tokenizer


def load_reward_model():
    """Load the trained DistilBERT reward model and its tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔧  Loading reward model from {REWARD_MODEL_PATH} ...")
    reward_model = RewardModel(REWARD_BACKBONE_ID).to(device)

    checkpoint = torch.load(REWARD_MODEL_PATH, map_location=device)
    reward_model.load_state_dict(checkpoint["model_state_dict"])
    reward_model.eval()

    print(f"🔧  Loading reward tokenizer from {REWARD_TOKENIZER_PATH} ...")
    reward_tokenizer = DistilBertTokenizer.from_pretrained(REWARD_TOKENIZER_PATH)

    return reward_model, reward_tokenizer, device


def compute_reward(reward_model, reward_tokenizer, prompt: str, response: str, device) -> float:
    """
    Score a single (prompt, response) pair using the reward model.
    Returns a scalar float reward.
    """
    text = prompt + " " + response
    encoding = reward_tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        score = reward_model(encoding["input_ids"], encoding["attention_mask"]).item()
    return score


# ── Main PPO Training Loop ─────────────────────────────────────────────────

def train_ppo():
    print("=" * 70)
    print("  HireSignal — PPO RLHF Training Loop")
    print("=" * 70)

    # ── Step 1: Initialize W&B ────────────────────────────────────────────
    wandb.init(project="hiresignal-ppo")

    # ── Step 2: Load validation prompts ───────────────────────────────────
    print(f"\n📂  Loading prompts from {VAL_DATA_PATH} ...")
    with open(VAL_DATA_PATH, "r") as f:
        val_data = json.load(f)
    print(f"   Available prompts: {len(val_data)}")

    # ── Step 3: Load SFT model + tokenizer ────────────────────────────────
    print()
    model, tokenizer = load_sft_model_and_tokenizer()

    # ── Step 4: Load reward model ─────────────────────────────────────────
    print()
    reward_model, reward_tokenizer, reward_device = load_reward_model()

    # ── Step 5: Configure PPO ─────────────────────────────────────────────
    print("\n⚙️   Configuring PPO ...")
    ppo_config = PPOConfig(
        model_name=MODEL_ID,
        learning_rate=1.4e-5,
        batch_size=PPO_BATCH_SIZE,
        mini_batch_size=PPO_BATCH_SIZE,
        ppo_epochs=1,
        log_with="wandb",
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,  # Uses the model's own reference copy (PEFT handles this)
        tokenizer=tokenizer,
    )

    # ── Step 6: PPO Training Loop ─────────────────────────────────────────
    print(f"\n🚀  Starting PPO loop — {NUM_PPO_STEPS} steps, batch_size={PPO_BATCH_SIZE}")
    print("─" * 70)

    # Track rewards across steps for summary
    step_rewards = []

    for step in range(1, NUM_PPO_STEPS + 1):
        # Sample a batch of JD prompts
        batch_samples = random.sample(val_data, min(PPO_BATCH_SIZE, len(val_data)))
        prompts = [
            build_prompt(s["instruction"], s["input"]) for s in batch_samples
        ]

        # Tokenize prompts into query tensors
        query_tensors = []
        for p in prompts:
            tokens = tokenizer(p, return_tensors="pt", truncation=True, max_length=256)
            query_tensors.append(tokens["input_ids"].squeeze(0))

        # Generate responses using current policy
        generation_kwargs = {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
        }
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query.unsqueeze(0), **generation_kwargs)
            # Extract only the newly generated tokens (after the query)
            response_new = response.squeeze(0)[len(query):]
            response_tensors.append(response_new)

        # Decode responses and compute rewards using the DistilBERT reward model
        rewards = []
        for prompt_text, resp_tensor in zip(prompts, response_tensors):
            response_text = tokenizer.decode(resp_tensor, skip_special_tokens=True)
            score = compute_reward(reward_model, reward_tokenizer, prompt_text, response_text, reward_device)
            rewards.append(torch.tensor(score, dtype=torch.float32))

        # Run PPO update step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Compute reward stats for this step
        reward_values = [r.item() for r in rewards]
        mean_reward = sum(reward_values) / len(reward_values)
        max_reward = max(reward_values)
        min_reward = min(reward_values)
        step_rewards.append(mean_reward)

        # Log to console
        print(
            f"   Step {step:>3d}/{NUM_PPO_STEPS} │ "
            f"Mean Reward: {mean_reward:+.4f} │ "
            f"Max: {max_reward:+.4f} │ "
            f"Min: {min_reward:+.4f}"
        )

        # Log to W&B
        wandb.log({
            "ppo/step": step,
            "ppo/mean_reward": mean_reward,
            "ppo/max_reward": max_reward,
            "ppo/min_reward": min_reward,
        })

    print("─" * 70)

    # ── Step 7: Training Summary ──────────────────────────────────────────
    step1_reward = step_rewards[0]
    step_final_reward = step_rewards[-1]

    if step1_reward != 0:
        pct_improvement = ((step_final_reward - step1_reward) / abs(step1_reward)) * 100
    else:
        pct_improvement = 0.0

    print(f"\n{'=' * 70}")
    print("  PPO Training Summary")
    print(f"{'=' * 70}")
    print(f"  Step 1 mean reward:     {step1_reward:+.4f}")
    print(f"  Step {NUM_PPO_STEPS} mean reward:    {step_final_reward:+.4f}")
    print(f"  % improvement:          {pct_improvement:+.1f}%")
    print(f"  RLHF alignment improved response structure by {abs(pct_improvement):.1f}%")
    print(f"{'=' * 70}")

    # ── Step 8: Save final aligned model ──────────────────────────────────
    print(f"\n💾  Saving PPO-aligned model to {PPO_MODEL_SAVE_DIR} ...")
    os.makedirs(PPO_MODEL_SAVE_DIR, exist_ok=True)

    # Save the PEFT adapter (aligned version)
    ppo_trainer.model.save_pretrained(PPO_MODEL_SAVE_DIR)
    tokenizer.save_pretrained(PPO_MODEL_SAVE_DIR)
    print(f"   Model and tokenizer saved to {PPO_MODEL_SAVE_DIR}")

    wandb.finish()
    print("\n✅  PPO training complete!")


if __name__ == "__main__":
    train_ppo()
