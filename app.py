"""
HireSignal — AI Job Description Analyzer
Gradio application with 3 tabs: JD Analyzer, Fit Scorer, Resume Tip Generator.
Loads the PPO-aligned model (with SFT fallback) for structured recruitment insights.
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.file_extractor import extract_text_from_file

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "microsoft/phi-2"
PPO_ADAPTER_PATH = "outputs/ppo_model"
SFT_ADAPTER_PATH = "outputs/sft_model/final_adapter"

# ── Example JDs ──────────────────────────────────────────────────────────────

EXAMPLE_JD_VAGUE = """Full Stack Developer (Fresher Welcome!)
We are a fast-paced, dynamic startup disrupting the industry. We need a rockstar ninja developer who can wear many hats. You'll build our entire platform from scratch — frontend, backend, DevOps, database, mobile apps, AI/ML pipelines, and cloud infra. Must have 5+ years experience with React, Angular, Vue, Node.js, Python, Go, Rust, Java, Kubernetes, Docker, AWS, GCP, Azure, MongoDB, PostgreSQL, Redis, Kafka, GraphQL, REST, WebSockets, CI/CD, Terraform, and blockchain. Competitive salary. Work hard, play hard. Self-starters only. Other duties as assigned."""

EXAMPLE_JD_MNC = """Senior Data Engineer — Enterprise Data Platform
Location: Hyderabad, India | Experience: 5-8 years

About the Role:
Join our Enterprise Data Platform team to build scalable data pipelines processing 50TB+ daily. You will design real-time streaming architectures and mentor junior engineers.

Responsibilities:
- Design and maintain Apache Spark and Kafka-based ETL pipelines.
- Optimize data warehouse queries on Snowflake for analytics workloads.
- Implement data quality frameworks with Great Expectations.
- Collaborate with ML teams to serve feature stores for model training.
- Lead code reviews and establish engineering best practices.

Requirements:
- 5-8 years in data engineering with production Spark experience.
- Strong SQL skills and experience with Snowflake or BigQuery.
- Proficiency in Python and Scala.
- Experience with Airflow or Dagster for orchestration.
- Familiarity with CI/CD for data pipelines (dbt, Terraform).

Compensation: ₹30L - ₹48L + RSUs + annual bonus."""

EXAMPLE_JD_AI = """AI/ML Engineer — InteligenAI
Location: Remote (US timezone overlap) | Experience: 3-5 years

About Us:
At InteligenAI, we build AI-powered hiring tools. Our platform uses large language models to analyze resumes, score candidate fit, and generate personalized interview questions.

Responsibilities:
- Design and implement LLM fine-tuning pipelines using Hugging Face Transformers and TRL.
- Build and optimize RAG systems for document understanding.
- Develop evaluation frameworks to measure model quality, bias, and hallucination rates.
- Deploy models on AWS SageMaker with auto-scaling inference endpoints.
- Collaborate with product to translate business requirements into ML solutions.

Requirements:
- 3-5 years in ML/NLP with at least 1 year working with LLMs.
- Strong proficiency in Python, PyTorch, and Hugging Face ecosystem.
- Experience with RLHF, DPO, or other alignment techniques.
- Familiarity with vector databases (Pinecone, Weaviate, or Chroma).
- MS or PhD in Computer Science, ML, or related field preferred.

Compensation: $150,000 - $200,000 base + equity."""

EXAMPLE_FIT_1_JD = """Backend Engineer — Payments Team
Requirements: 3+ years Python, experience with PostgreSQL, Redis, REST APIs, microservices architecture, payment gateway integrations (Stripe/Razorpay). AWS deployment experience required."""

EXAMPLE_FIT_1_RESUME = """2 years Python development, built REST APIs with Flask, used PostgreSQL and MongoDB, deployed on Heroku, basic understanding of Redis caching, no payment integration experience."""

EXAMPLE_FIT_2_JD = """ML Engineer — Computer Vision
Requirements: 3+ years experience, PyTorch, CNNs, object detection (YOLO, Detectron2), model deployment with TensorRT, experience with medical imaging datasets preferred."""

EXAMPLE_FIT_2_RESUME = """4 years ML experience, strong PyTorch and TensorFlow skills, built image classification models, experience with YOLO v5 for retail analytics, deployed models on AWS Lambda, no medical imaging experience."""

EXAMPLE_RESUME_1_JD = """Product Manager — SaaS Growth
Requirements: data-driven decision making, A/B testing, user journey mapping, cross-functional collaboration, SQL proficiency."""

EXAMPLE_RESUME_1_BULLETS = """- Managed product roadmap for a B2B tool
- Worked with engineering and design teams
- Analyzed user feedback to prioritize features
- Launched 3 product updates in 6 months"""

EXAMPLE_RESUME_2_JD = """DevOps Engineer — FinTech
Requirements: Kubernetes, Docker, Terraform, CI/CD (GitHub Actions), monitoring (Prometheus, Grafana), AWS, security compliance (SOC2)."""

EXAMPLE_RESUME_2_BULLETS = """- Set up deployment pipelines using Jenkins
- Managed AWS EC2 and S3 resources
- Monitored application health with CloudWatch
- Assisted in Docker containerization of services"""


# ── Model Loading ────────────────────────────────────────────────────────────


def load_model():
    """Load the best available model: PPO-aligned → SFT fallback → base model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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
        bnb_config = None
        device_map = {"": "cpu"}

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if has_cuda else torch.float32,
    )

    # Try PPO adapter first, then SFT adapter
    adapter_path = None
    if os.path.isdir(PPO_ADAPTER_PATH) and os.listdir(PPO_ADAPTER_PATH):
        adapter_path = PPO_ADAPTER_PATH
        print(f"✅  Loaded PPO-aligned adapter from {PPO_ADAPTER_PATH}")
    elif os.path.isdir(SFT_ADAPTER_PATH) and os.listdir(SFT_ADAPTER_PATH):
        adapter_path = SFT_ADAPTER_PATH
        print(f"✅  Loaded SFT adapter from {SFT_ADAPTER_PATH}")
    else:
        print("No fine-tuned adapter found, using base model")

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    else:
        model = base_model

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    """Run real model inference and return only the response portion."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    marker = "### Response:\n"
    if marker in full_text:
        return full_text.split(marker, 1)[1].strip()
    return full_text[len(prompt):].strip()


# ── File Upload Handler ──────────────────────────────────────────────────────

def extract_file(file) -> tuple:
    """Extract text from an uploaded file and return (text, status_message)."""
    if file is None:
        return "", "⚠ No file uploaded. Please select a file first."
    try:
        text = extract_text_from_file(file.name)
        if not text.strip():
            return "", "⚠ File appears empty. Please paste JD manually."
        return text, "✅ Text extracted! Review and click Analyze JD."
    except Exception as e:
        return "", f"❌ Could not extract text from file. Please paste JD manually. ({e})"


# ── Tab 1: JD Analyzer ──────────────────────────────────────────────────────

def analyze_jd(job_description: str) -> str:
    if not job_description.strip():
        return "⚠ Please paste a job description to analyze."

    yield "⏳ Analyzing JD, please wait..."

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
    try:
        result = generate(model, tokenizer, prompt)
        yield result
    except Exception as e:
        yield f"❌ Model inference failed: {e}"


# ── Tab 2: Fit Scorer ───────────────────────────────────────────────────────

def score_fit(job_description: str, resume_summary: str) -> str:
    if not job_description.strip() or not resume_summary.strip():
        return "⚠ Please provide both a job description and your skills/resume summary."

    yield "⏳ Scoring fit, please wait..."

    instruction = (
        "You are an expert career coach. Given a job description and a candidate's "
        "skills/resume summary, provide:\n"
        "1. A fit score out of 10\n"
        "2. Matching skills the candidate already has\n"
        "3. Missing skills the candidate needs to develop\n"
        "4. How to position themselves for this role despite any gaps"
    )
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Job Description:\n{job_description.strip()}\n\n"
        f"### Candidate Profile:\n{resume_summary.strip()}\n\n"
        f"### Response:\n"
    )
    try:
        result = generate(model, tokenizer, prompt, max_new_tokens=450)
        yield result
    except Exception as e:
        yield f"❌ Model inference failed: {e}"


# ── Tab 3: Resume Tip Generator ─────────────────────────────────────────────

def improve_resume(job_description: str, resume_bullets: str) -> str:
    if not job_description.strip() or not resume_bullets.strip():
        return "⚠ Please provide both a job description and your current resume bullet points."

    yield "⏳ Improving resume bullets, please wait..."

    instruction = (
        "You are a professional resume writer. Given a target job description and the "
        "candidate's current resume bullet points, rewrite each bullet to:\n"
        "1. Mirror keywords from the job description\n"
        "2. Add quantifiable metrics where possible\n"
        "3. Use strong action verbs\n"
        "4. Highlight transferable skills relevant to the target role\n"
        "Return the improved bullet points."
    )
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Target Job Description:\n{job_description.strip()}\n\n"
        f"### Current Resume Bullets:\n{resume_bullets.strip()}\n\n"
        f"### Response:\n"
    )
    try:
        result = generate(model, tokenizer, prompt, max_new_tokens=450)
        yield result
    except Exception as e:
        yield f"❌ Model inference failed: {e}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container { max-width: 960px !important; margin: auto; }
.app-header { text-align: center; margin-bottom: 8px; }
.app-header h1 { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2em; font-weight: 800; margin-bottom: 0;
}
.app-header p { color: #6b7280; font-size: 1.05em; margin-top: 4px; }
.footer-text { text-align: center; color: #9ca3af; font-size: 0.85em; margin-top: 16px; }
"""


def build_app():
    """Construct the Gradio Blocks interface with 3 tabs."""

    with gr.Blocks(title="HireSignal — AI Job Description Analyzer") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <h1>🎯 HireSignal — AI Job Description Analyzer</h1>
            <p>Fine-tuned using SFT + RLHF on recruitment data. Paste any JD to extract skills, score your fit, and improve your resume.</p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: JD Analyzer ────────────────────────────────────────
            with gr.TabItem("JD Analyzer"):
                gr.Markdown("### Analyze any job description for skills, level, responsibilities & red flags")
                with gr.Row():
                    with gr.Column():
                        jd_input = gr.Textbox(
                            label="Job Description",
                            placeholder="Paste a job description here...",
                            lines=12,
                            elem_id="jd-analyzer-input",
                        )
                        jd_file_upload = gr.File(
                            label="Or upload JD as PDF / Image / DOCX",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".docx", ".txt"],
                            elem_id="jd-file-upload",
                        )
                        extract_btn = gr.Button("Extract from File", variant="secondary", elem_id="extract-btn")
                        extract_status = gr.Textbox(
                            label="Upload Status",
                            interactive=False,
                            lines=1,
                            elem_id="extract-status",
                        )
                        analyze_btn = gr.Button("Analyze JD", variant="primary", elem_id="analyze-btn")
                    with gr.Column():
                        jd_output = gr.Textbox(
                            label="Structured Analysis",
                            lines=16,
                            interactive=False,
                            elem_id="jd-analyzer-output",
                        )

                gr.Examples(
                    examples=[
                        [EXAMPLE_JD_VAGUE],
                        [EXAMPLE_JD_MNC],
                        [EXAMPLE_JD_AI],
                    ],
                    inputs=[jd_input],
                    label="Try these example JDs",
                )

                extract_btn.click(fn=extract_file, inputs=[jd_file_upload], outputs=[jd_input, extract_status])
                analyze_btn.click(fn=analyze_jd, inputs=[jd_input], outputs=[jd_output])

            # ── Tab 2: Fit Scorer ─────────────────────────────────────────
            with gr.TabItem("Fit Scorer"):
                gr.Markdown("### See how well you match a role — get a score, gap analysis & positioning tips")
                with gr.Row():
                    with gr.Column():
                        fit_jd_input = gr.Textbox(
                            label="Job Description",
                            placeholder="Paste the target job description...",
                            lines=8,
                            elem_id="fit-jd-input",
                        )
                        fit_resume_input = gr.Textbox(
                            label="Your Skills / Resume Summary",
                            placeholder="List your skills, experience, and relevant background...",
                            lines=5,
                            elem_id="fit-resume-input",
                        )
                        fit_btn = gr.Button("Score My Fit", variant="primary", elem_id="fit-btn")
                    with gr.Column():
                        fit_output = gr.Textbox(
                            label="Fit Analysis",
                            lines=16,
                            interactive=False,
                            elem_id="fit-output",
                        )

                gr.Examples(
                    examples=[
                        [EXAMPLE_FIT_1_JD, EXAMPLE_FIT_1_RESUME],
                        [EXAMPLE_FIT_2_JD, EXAMPLE_FIT_2_RESUME],
                    ],
                    inputs=[fit_jd_input, fit_resume_input],
                    label="Try these examples",
                )

                fit_btn.click(fn=score_fit, inputs=[fit_jd_input, fit_resume_input], outputs=[fit_output])

            # ── Tab 3: Resume Tip Generator ───────────────────────────────
            with gr.TabItem("Resume Tip Generator"):
                gr.Markdown("### Rewrite your resume bullets to match a target JD — with metrics & keywords")
                with gr.Row():
                    with gr.Column():
                        resume_jd_input = gr.Textbox(
                            label="Target Job Description",
                            placeholder="Paste the JD you're applying to...",
                            lines=6,
                            elem_id="resume-jd-input",
                        )
                        resume_bullets_input = gr.Textbox(
                            label="Current Resume Bullet Points",
                            placeholder="Paste your existing resume bullets (one per line)...",
                            lines=5,
                            elem_id="resume-bullets-input",
                        )
                        resume_btn = gr.Button("Improve My Resume", variant="primary", elem_id="resume-btn")
                    with gr.Column():
                        resume_output = gr.Textbox(
                            label="Improved Resume Bullets",
                            lines=14,
                            interactive=False,
                            elem_id="resume-output",
                        )

                gr.Examples(
                    examples=[
                        [EXAMPLE_RESUME_1_JD, EXAMPLE_RESUME_1_BULLETS],
                        [EXAMPLE_RESUME_2_JD, EXAMPLE_RESUME_2_BULLETS],
                    ],
                    inputs=[resume_jd_input, resume_bullets_input],
                    label="Try these examples",
                )

                resume_btn.click(
                    fn=improve_resume,
                    inputs=[resume_jd_input, resume_bullets_input],
                    outputs=[resume_output],
                )

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-text">
            Built with Phi-2 + QLoRA + PPO | Not a substitute for human judgment
        </div>
        """)

    return demo


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀  Loading HireSignal model ...")
    model, tokenizer = load_model()
    print("🚀  Building Gradio interface ...")
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
