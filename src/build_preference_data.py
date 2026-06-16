"""
Build Preference Data for RLHF — HireSignal
=============================================

This script constructs a preference dataset used to train a Reward Model (RM)
or run Direct Preference Optimization (DPO). In the RLHF pipeline, the model
must learn *which* outputs humans prefer — not just how to generate text.

┌─────────────────────────────────────────────────────────────────────────────┐
│  WHAT IS PREFERENCE DATA IN RLHF?                                         │
│                                                                           │
│  In Reinforcement Learning from Human Feedback, we need pairs of          │
│  (prompt, chosen_response, rejected_response). The Reward Model learns    │
│  to assign higher scores to "chosen" responses and lower scores to        │
│  "rejected" ones. This reward signal is then used during PPO training     │
│  to steer the language model toward producing preferred outputs.          │
│                                                                           │
│  Pipeline:  Preference Pairs → Reward Model → PPO fine-tuning            │
│                                                                           │
│  WHY STRUCTURED = "CHOSEN" FOR RECRUITMENT?                               │
│                                                                           │
│  In a hiring/HR context, recruiters need responses that are:              │
│    • Consistently formatted (Skills, Level, Responsibilities, Red Flags)  │
│    • Concise and scannable — no rambling paragraphs                       │
│    • Actionable — clearly listing skills and levels                       │
│                                                                           │
│  A response that contains all expected headers in the right format is     │
│  objectively more useful than a verbose, unstructured wall of text.       │
│  By labelling structured outputs as "chosen", the Reward Model learns     │
│  to score structure and completeness highly — which is exactly what       │
│  we want the final RLHF-tuned model to produce.                          │
│                                                                           │
│  HOW THE REWARD MODEL LEARNS FROM THIS                                    │
│                                                                           │
│  The RM is a classifier trained with a ranking loss:                      │
│    loss = -log(sigmoid(r_chosen - r_rejected))                            │
│  It learns that structured, header-complete responses should get a        │
│  higher scalar reward than rambling ones. During PPO, the policy LLM      │
│  generates candidates and the RM scores them — the LLM is then updated   │
│  to maximize this reward while staying close to the SFT checkpoint        │
│  (via a KL-divergence penalty).                                           │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_PATH = "outputs/sft_model/final_adapter"
VAL_DATA_PATH = "data/processed/val.json"
OUTPUT_PATH = "data/processed/preference_data.json"
NUM_SAMPLES = 150  # Maximum samples to process from validation set


# ── Scoring Function ────────────────────────────────────────────────────────

def score_structure(response: str) -> int:
    """
    Score a response based on how well it follows the expected structured format.

    Checks for the presence and ordering of required section headers that
    a recruiter would expect. Higher score = more structured and useful.

    Scoring rubric (max 10 points):
      +2  Contains "Title:" header
      +2  Contains "Skills:" header
      +2  Contains "Level:" header
      +2  Contains "Responsibilities:" header
      +1  Contains "Red Flags:" header
      +1  Contains bullet points (lines starting with "- ")

    Returns:
        int: Structure quality score (0–10).
    """
    score = 0

    # Check for required section headers
    required_headers = {
        "Title:": 2,
        "Skills:": 2,
        "Level:": 2,
        "Responsibilities:": 2,
        "Red Flags:": 1,
    }
    for header, points in required_headers.items():
        if header.lower() in response.lower():
            score += points

    # Check for bullet point formatting (at least one "- " line)
    lines = response.strip().split("\n")
    if any(line.strip().startswith("- ") for line in lines):
        score += 1

    return score


# ── Prompt Builder ──────────────────────────────────────────────────────────

def build_prompt(instruction: str, job_description: str) -> str:
    """Format instruction and JD into the SFT training template."""
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Job Description:\n{job_description.strip()}\n\n"
        f"### Response:\n"
    )


# ── Response Generator ──────────────────────────────────────────────────────

def generate_response(model, tokenizer, prompt: str, temperature: float) -> str:
    """
    Generate a single response from the model at a given temperature.

    Lower temperature → more deterministic, structured output.
    Higher temperature → more creative but potentially rambling output.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip everything before "### Response:\n"
    marker = "### Response:\n"
    if marker in full_text:
        return full_text.split(marker, 1)[1].strip()
    return full_text[len(prompt):].strip()


# ── Model Loading ───────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    """Load base model with 4-bit quantization and apply LoRA adapter."""
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    print(f"Loading base model ({MODEL_ID}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if has_cuda else torch.float32,
    )

    print(f"Loading LoRA adapter from {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return model, tokenizer


# ── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HireSignal — Preference Data Builder for RLHF")
    print("=" * 70)

    # Step 1: Load validation data
    print(f"\n📂  Loading validation data from {VAL_DATA_PATH} ...")
    with open(VAL_DATA_PATH, "r") as f:
        val_data = json.load(f)

    # Cap at NUM_SAMPLES
    samples = val_data[:NUM_SAMPLES]
    print(f"   Using {len(samples)} samples (requested {NUM_SAMPLES}, available {len(val_data)})")

    # Step 2: Load fine-tuned model
    print()
    model, tokenizer = load_model_and_tokenizer()

    # Step 3 & 4: Generate paired responses and build preference dataset
    print(f"\n🔄  Generating preference pairs ...")
    preference_pairs = []
    a_naturally_better = 0  # Count where Response A (low temp) scored higher without swap
    total_processed = 0

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        job_description = sample["input"]
        prompt = build_prompt(instruction, job_description)

        # Generate Response A — low temperature (structured, precise)
        response_a = generate_response(model, tokenizer, prompt, temperature=0.3)

        # Generate Response B — high temperature (rambling, less structured)
        response_b = generate_response(model, tokenizer, prompt, temperature=1.1)

        # Score both responses on structural quality
        score_a = score_structure(response_a)
        score_b = score_structure(response_b)

        # Assign chosen/rejected based on actual structure scores
        # If A is genuinely more structured → A is chosen (natural order)
        # If B is accidentally more structured → swap so chosen is always better
        if score_a >= score_b:
            chosen = response_a
            rejected = response_b
            a_naturally_better += 1
        else:
            chosen = response_b
            rejected = response_a

        preference_pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": max(score_a, score_b),
            "rejected_score": min(score_a, score_b),
        })

        total_processed += 1

        # Progress logging every 10 samples
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"   ✓ Processed {i + 1}/{len(samples)} pairs "
                  f"(A better: {a_naturally_better}/{total_processed})")

    # Step 5: Save preference dataset
    print(f"\n💾  Saving {len(preference_pairs)} preference pairs to {OUTPUT_PATH} ...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(preference_pairs, f, indent=2)

    # Step 6: Print stats
    pct_a_better = (a_naturally_better / total_processed * 100) if total_processed > 0 else 0
    avg_chosen_score = sum(p["chosen_score"] for p in preference_pairs) / len(preference_pairs)
    avg_rejected_score = sum(p["rejected_score"] for p in preference_pairs) / len(preference_pairs)

    print(f"\n{'=' * 70}")
    print(f"  Preference Data Summary")
    print(f"{'=' * 70}")
    print(f"  Total pairs generated:               {len(preference_pairs)}")
    print(f"  Response A naturally better:          {a_naturally_better}/{total_processed} ({pct_a_better:.1f}%)")
    print(f"  Average chosen structure score:       {avg_chosen_score:.2f} / 10")
    print(f"  Average rejected structure score:     {avg_rejected_score:.2f} / 10")
    print(f"  Saved to:                             {OUTPUT_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
