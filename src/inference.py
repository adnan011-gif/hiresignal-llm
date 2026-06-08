"""
Inference Script for HireSignal.
Loads the base microsoft/phi-2 model and the fine-tuned LoRA adapter,
then runs side-by-side comparisons on realistic job descriptions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "microsoft/phi-2"
ADAPTER_PATH = "outputs/sft_model/final_adapter"

# ── Test Job Descriptions ────────────────────────────────────────────────────

JD_FULLSTACK = """
Full Stack Developer (Fresher Welcome!)
We are a fast-paced, dynamic startup disrupting the industry. We need a rockstar 
ninja developer who can wear many hats. You'll be responsible for building our 
entire platform from scratch — frontend, backend, DevOps, database design, mobile 
apps, AI/ML pipelines, and managing our cloud infrastructure. Must have 5+ years 
of experience with React, Angular, Vue, Node.js, Python, Go, Rust, Java, Kubernetes, 
Docker, AWS, GCP, Azure, MongoDB, PostgreSQL, Redis, Kafka, GraphQL, REST, 
WebSockets, CI/CD, Terraform, and blockchain. Competitive salary. We work hard, 
play hard. Self-starters only. Other duties as assigned.
"""

JD_DATA_ANALYST = """
Data Analyst — Marketing Intelligence Team
Location: Bangalore, India (Hybrid — 3 days in office)
Experience: 2-4 years

About the Role:
We are looking for a Data Analyst to join our Marketing Intelligence team. You will 
work closely with the marketing and product teams to analyze campaign performance, 
build dashboards, and generate actionable insights to optimize customer acquisition costs.

Responsibilities:
- Build and maintain Tableau dashboards tracking key marketing KPIs.
- Analyze A/B test results and provide statistical recommendations.
- Write complex SQL queries to extract and transform data from our data warehouse.
- Collaborate with data engineering to improve ETL pipeline quality.
- Present weekly insight reports to VP of Marketing.

Requirements:
- 2-4 years of experience in data analytics or business intelligence.
- Proficiency in SQL, Excel, and Tableau (or Power BI).
- Strong understanding of statistical analysis (hypothesis testing, regression).
- Experience with Python or R for data analysis is a plus.
- Excellent communication skills for presenting insights to non-technical stakeholders.

Nice to Have:
- Experience with Google Analytics and marketing attribution models.
- Familiarity with dbt or Airflow for data transformation pipelines.
"""

JD_AI_ENGINEER = """
AI/ML Engineer — Conversational AI Platform
Location: Remote (US timezone overlap required)
Experience: 3-5 years

About Us:
We are building the next generation of AI-powered hiring tools. Our platform uses 
large language models to analyze resumes, score candidate fit, and generate 
personalized interview questions.

Responsibilities:
- Design and implement LLM fine-tuning pipelines using Hugging Face Transformers and TRL.
- Build and optimize RAG (Retrieval-Augmented Generation) systems for document understanding.
- Develop evaluation frameworks to measure model quality, bias, and hallucination rates.
- Deploy models on AWS SageMaker with auto-scaling inference endpoints.
- Collaborate with product to translate business requirements into ML solutions.

Requirements:
- 3-5 years of experience in ML/NLP with at least 1 year working with LLMs.
- Strong proficiency in Python, PyTorch, and Hugging Face ecosystem.
- Experience with RLHF, DPO, or other alignment techniques.
- Familiarity with vector databases (Pinecone, Weaviate, or Chroma).
- Understanding of prompt engineering and evaluation best practices.
- MS or PhD in Computer Science, ML, or related field preferred.

Compensation:
- $150,000 - $200,000 base + equity
- Unlimited PTO, health insurance, home office stipend.
"""

# Collect test JDs in a list
TEST_JDS = [
    ("Full Stack Developer (Vague/Buzzword-heavy)", JD_FULLSTACK),
    ("Data Analyst (Clear & Specific)", JD_DATA_ANALYST),
    ("AI/ML Engineer (Domain-relevant)", JD_AI_ENGINEER),
]


def build_prompt(job_description: str) -> str:
    """Format a job description into the instruction template used during SFT training."""
    instruction = (
        "Analyze the following job description. Extract:\n"
        "1. Top 5 required skills\n"
        "2. Experience level expected (fresher/mid/senior)\n"
        "3. Key responsibilities in 3 bullet points\n"
        "4. Red flags or vague requirements if any"
    )
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Job Description:\n{job_description.strip()}\n\n"
        f"### Response:\n"
    )
    return prompt


def load_tokenizer():
    """Load and configure the tokenizer for phi-2."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(tokenizer):
    """Load the base phi-2 model with 4-bit quantization (or CPU fallback)."""
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
        print("⚠  CUDA not available — loading model on CPU without quantization.")
        bnb_config = None
        device_map = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if has_cuda else torch.float32,
    )
    return model


def load_finetuned_model(base_model):
    """Wrap the base model with the trained LoRA adapter."""
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    ft_model.eval()
    return ft_model


def generate_response(model, tokenizer, prompt: str) -> str:
    """Run model.generate() and return only the generated response text."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode full output and strip the instruction/prompt prefix
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only what comes after "### Response:\n"
    marker = "### Response:\n"
    if marker in full_text:
        response = full_text.split(marker, 1)[1].strip()
    else:
        # Fallback: strip the original prompt length
        response = full_text[len(prompt):].strip()

    return response


def analyze_jd(job_description: str, model, tokenizer) -> str:
    """Public-facing helper: format prompt → generate → return structured response."""
    prompt = build_prompt(job_description)
    return generate_response(model, tokenizer, prompt)


def main():
    print("=" * 80)
    print("  HireSignal — Base Model vs Fine-Tuned Model Comparison")
    print("=" * 80)

    # ── Load tokenizer ────────────────────────────────────────────────────
    print("\n🔧  Loading tokenizer ...")
    tokenizer = load_tokenizer()

    # ── Load base model ───────────────────────────────────────────────────
    print("🔧  Loading base model (microsoft/phi-2) ...")
    base_model = load_base_model(tokenizer)

    # ── Load fine-tuned model ─────────────────────────────────────────────
    print("🔧  Loading fine-tuned LoRA adapter ...")
    try:
        ft_model = load_finetuned_model(base_model)
        adapter_loaded = True
    except Exception as e:
        print(f"⚠  Could not load adapter from {ADAPTER_PATH}: {e}")
        print("   Fine-tuned comparison will be skipped.")
        adapter_loaded = False

    # ── Run comparison on each test JD ────────────────────────────────────
    for idx, (label, jd) in enumerate(TEST_JDS, start=1):
        prompt = build_prompt(jd)

        print(f"\n{'━' * 80}")
        print(f"  TEST {idx}: {label}")
        print(f"{'━' * 80}")
        print(f"📄  JD Preview: {jd.strip()[:200]} ...")

        # Base model response
        print(f"\n{'─' * 40}")
        print("🟡  BASE MODEL RESPONSE (no adapter):")
        print(f"{'─' * 40}")
        if adapter_loaded:
            # Disable adapter to get raw base model output
            ft_model.disable_adapter_layers()
            base_response = generate_response(ft_model, tokenizer, prompt)
            ft_model.enable_adapter_layers()
        else:
            base_response = generate_response(base_model, tokenizer, prompt)
        print(base_response)

        # Fine-tuned model response
        print(f"\n{'─' * 40}")
        print("🟢  FINE-TUNED MODEL RESPONSE (with adapter):")
        print(f"{'─' * 40}")
        if adapter_loaded:
            ft_response = generate_response(ft_model, tokenizer, prompt)
            print(ft_response)
        else:
            print("[Skipped — adapter not available]")

        print(f"\n{'═' * 80}\n")


if __name__ == "__main__":
    main()
