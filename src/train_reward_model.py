"""
Reward Model Training Script for HireSignal RLHF Pipeline.
Trains a DistilBERT-based scalar reward model on preference pairs
(chosen vs rejected JD analysis responses). The trained reward model
is later used during PPO to score LLM outputs and steer generation
toward structured, recruiter-friendly responses.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

# ── Constants ────────────────────────────────────────────────────────────────
PREFERENCE_DATA_PATH = "data/processed/preference_data.json"
MODEL_SAVE_DIR = "outputs/reward_model"
BACKBONE_ID = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5


# ── Reward Model Architecture ───────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Reward Model for RLHF.

    Architecture:
        DistilBERT (frozen or fine-tuned) → [CLS] hidden state
        → Dropout(0.1) → Linear(768 → 1) → scalar reward

    The model learns to output a higher scalar for "chosen" (structured)
    responses and a lower scalar for "rejected" (rambling) responses.
    """

    def __init__(self, backbone_name: str = BACKBONE_ID):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for distilbert-base

        self.dropout = nn.Dropout(0.1)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs        (batch_size, seq_len)
            attention_mask: Mask tensor  (batch_size, seq_len)

        Returns:
            Scalar reward scores         (batch_size,)
        """
        # Get [CLS] token representation from DistilBERT
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] = first token

        # Project to scalar reward
        cls_hidden = self.dropout(cls_hidden)
        reward = self.reward_head(cls_hidden).squeeze(-1)  # (batch_size,)
        return reward


# ── Preference Pair Dataset ─────────────────────────────────────────────────

class PreferencePairDataset(Dataset):
    """
    Converts preference pairs into flat (text, target) samples.

    Each preference pair yields two training examples:
      - positive: prompt + chosen response  → target = 1.0
      - negative: prompt + rejected response → target = 0.0
    """

    def __init__(self, pairs: list, tokenizer: DistilBertTokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for pair in pairs:
            prompt = pair["prompt"]

            # Positive example: prompt + chosen → 1.0
            chosen_text = prompt + " " + pair["chosen"]
            self.samples.append((chosen_text, 1.0))

            # Negative example: prompt + rejected → 0.0
            rejected_text = prompt + " " + pair["rejected"]
            self.samples.append((rejected_text, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, target = self.samples[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target": torch.tensor(target, dtype=torch.float32),
        }


# ── Training Loop ───────────────────────────────────────────────────────────

def train_reward_model():
    print("=" * 70)
    print("  HireSignal — Reward Model Training")
    print("=" * 70)

    # Step 1: Load preference data
    print(f"\n📂  Loading preference data from {PREFERENCE_DATA_PATH} ...")
    with open(PREFERENCE_DATA_PATH, "r") as f:
        preference_pairs = json.load(f)
    print(f"   Loaded {len(preference_pairs)} preference pairs")
    print(f"   Will create {len(preference_pairs) * 2} training samples (chosen + rejected)")

    # Step 2: Load tokenizer
    print(f"\n🔧  Loading tokenizer ({BACKBONE_ID}) ...")
    tokenizer = DistilBertTokenizer.from_pretrained(BACKBONE_ID)

    # Step 3: Build dataset and dataloader
    print("📦  Building training dataset ...")
    dataset = PreferencePairDataset(preference_pairs, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"   Total training samples: {len(dataset)}")
    print(f"   Batches per epoch:      {len(dataloader)}")

    # Step 4: Initialize model, optimizer, and loss
    print(f"\n🏗   Initializing RewardModel with {BACKBONE_ID} backbone ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model = RewardModel(BACKBONE_ID).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Track best model
    best_loss = float("inf")

    # Step 5: Training loop
    print(f"\n🚀  Starting training — {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("-" * 70)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            rewards = model(input_ids, attention_mask)
            loss = criterion(rewards, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Log every 10 steps
            if global_step % 10 == 0:
                print(f"   [Epoch {epoch+1}/{EPOCHS}] Step {global_step:>4d} | Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n   ✓ Epoch {epoch+1} complete — Avg Loss: {avg_epoch_loss:.4f}")

        # Save best model checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            save_path = os.path.join(MODEL_SAVE_DIR, "best_reward_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, save_path)
            print(f"   💾 Best model saved to {save_path} (loss: {best_loss:.4f})")

    print("-" * 70)

    # Step 6: Save final model
    final_path = os.path.join(MODEL_SAVE_DIR, "final_reward_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": EPOCHS,
        "loss": avg_epoch_loss,
    }, final_path)
    print(f"\n💾  Final model saved to {final_path}")

    # Also save tokenizer for inference convenience
    tokenizer.save_pretrained(os.path.join(MODEL_SAVE_DIR, "tokenizer"))
    print(f"   Tokenizer saved to {MODEL_SAVE_DIR}/tokenizer")

    # Step 7: Verification test on 5 pairs
    print(f"\n{'=' * 70}")
    print("  Verification — Scoring 5 Preference Pairs")
    print(f"{'=' * 70}")

    model.eval()
    test_pairs = preference_pairs[:5]
    correct = 0

    for i, pair in enumerate(test_pairs):
        prompt = pair["prompt"]

        # Score chosen response
        chosen_text = prompt + " " + pair["chosen"]
        chosen_enc = tokenizer(chosen_text, truncation=True, max_length=MAX_LENGTH,
                               padding="max_length", return_tensors="pt")
        chosen_enc = {k: v.to(device) for k, v in chosen_enc.items()}

        # Score rejected response
        rejected_text = prompt + " " + pair["rejected"]
        rejected_enc = tokenizer(rejected_text, truncation=True, max_length=MAX_LENGTH,
                                 padding="max_length", return_tensors="pt")
        rejected_enc = {k: v.to(device) for k, v in rejected_enc.items()}

        with torch.no_grad():
            chosen_score = model(chosen_enc["input_ids"], chosen_enc["attention_mask"]).item()
            rejected_score = model(rejected_enc["input_ids"], rejected_enc["attention_mask"]).item()

        is_correct = chosen_score > rejected_score
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"\n   Pair {i+1}: Chosen={chosen_score:.4f}  Rejected={rejected_score:.4f}  {status}")

    print(f"\n{'─' * 70}")
    print(f"   Reward model correctly ranked {correct}/5 pairs")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    train_reward_model()
