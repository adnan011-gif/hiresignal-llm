"""
Inference script for HireSignal.
Loads the fine-tuned LLM and runs inference to score resume-job matches and generate improvements.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_feedback(job_description: str, resume: str, model_path: str):
    """
    Generate match score and improvements for a resume given a job description.
    """
    print(f"Loading model from {model_path} and running inference...")
    # TODO: Load tokenizer and model, format prompt, generate output
    return "Placeholder feedback: Match Score: 85/100. Key improvements: add metrics."

if __name__ == "__main__":
    jd = "Software Engineer with experience in Python and LLMs."
    cv = "Junior Developer with Python knowledge."
    print(generate_feedback(jd, cv, "outputs/sft_model"))
