# 🎯 HireSignal — SFT + RLHF Fine-Tuned LLM for Job Description Analysis & Candidate Fit Scoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-FFBE00?style=flat&logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Problem Statement
Freshers apply to jobs blindly without knowing if they qualify, what skills they're missing, or how to position their resume. HireSignal solves this with a fine-tuned, RLHF-aligned LLM trained specifically on recruitment data.

---

## 🏗️ Architecture & Pipeline

```text
   Job Description Dataset
              ↓
   Data Preparation + Rule-Based Response Generation
              ↓
   SFT Training with QLoRA on Phi-2
              ↓
   Preference Data with Structure-Based Scoring
              ↓
   Reward Model Training (DistilBERT)
              ↓
   PPO / RLHF Alignment Loop
              ↓
   Final Model → Gradio UI (3 tabs)
```

---

## 🛠️ Tech Stack

| Component | Tools / Libraries Used | Purpose |
| :--- | :--- | :--- |
| **Core Frameworks** | PyTorch, Hugging Face Transformers | Base deep learning and transformer utilities |
| **Fine-Tuning** | PEFT (QLoRA), TRL (Transformer Reinforcement Learning) | Parameter-efficient fine-tuning and reinforcement learning optimization |
| **Base LLM** | `microsoft/phi-2` (2.7B parameters) | Causal language model fine-tuned for generation tasks |
| **Reward Model** | `distilbert-base-uncased` | Sequence classifier trained to predict human/recruiter preference scores |
| **User Interface** | Gradio 6.x | Web-based interface for interactive analysis |
| **Experiment Tracking** | Weights & Biases (W&B) | Hyperparameter tuning, loss tracking, and run comparisons |
| **Data Processing** | Pandas, Hugging Face Datasets, Scikit-learn | Data ingestion, cleaning, feature engineering, and train/val splits |

---

## 🌟 Key Features

| Feature | Input | Output | Description |
| :--- | :--- | :--- | :--- |
| **📋 JD Analyzer** | Job Description (12 lines) | Structured Analysis | Extracts top 5 required skills, expected experience level, key responsibilities in 3 concise bullets, and highlights hidden red flags. |
| **🎯 Fit Scorer** | Target JD (8 lines) + Candidate Profile (5 lines) | Fit score, Gaps, and Positioning | Calculates a match score out of 10, does a gap analysis of missing skills, and advises on how the candidate can position their background. |
| **✍️ Resume Tip Generator** | Target JD (6 lines) + Current resume bullets (5 lines) | Tailored resume bullets | Rewrites existing bullets using target keywords, strong action verbs, and quantifiable metrics. |

---

## 💻 Step-by-Step Local Setup

Follow these commands to clone, configure, run the training scripts, and launch the interactive web application:

### 1. Clone the Repository
```bash
git clone https://github.com/adnan011-gif/hiresignal-llm.git
cd hiresignal-llm
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute Data Preparation
Prepare datasets, parse fields, and output the SFT training records:
```bash
python src/data_prep.py
```

### 5. Run Supervised Fine-Tuning (SFT)
Fine-tune the base `microsoft/phi-2` model using QLoRA:
```bash
python src/train_sft.py
```

### 6. Build Preference Data
Construct chosen vs. rejected outputs to train the Reward Model:
```bash
python src/build_preference_data.py
```

### 7. Train the Reward Model
Train the DistilBERT sequence classifier on preference data:
```bash
python src/train_reward_model.py
```

### 8. Run PPO Optimization
Align the model's responses to recruiter standards using PPO reinforcement learning:
```bash
python src/train_ppo.py
```

### 9. Launch the Gradio UI
```bash
python app.py
```
This will start the web app on `http://localhost:7860`.

---

## 📁 Project Folder Structure

```text
hiresignal-llm/
├── data/
│   ├── raw/                      # Raw resume and job description datasets
│   └── processed/                # Preprocessed and split JSON records (train.json, val.json)
├── src/
│   ├── data_prep.py              # Rule-based clean-up and formatting pipeline
│   ├── train_sft.py              # Supervised Fine-Tuning script using PEFT (LoRA)
│   ├── inference.py              # Command-line tool comparing base vs. fine-tuned outputs
│   ├── build_preference_data.py  # Generates chosen/rejected pair outputs for RLHF
│   ├── train_reward_model.py     # Reward model trainer (DistilBERT classifier)
│   └── train_ppo.py              # PPO alignment training loop with reward tracking
├── notebooks/                    # Prototyping and EDA notebooks
├── outputs/                      # Models and checkpoint outputs
│   ├── ppo_model/                # PPO aligned adapter files (highest priority)
│   └── sft_model/
│       └── final_adapter/        # SFT adapter files (fallback path)
├── app.py                        # Gradio 3-tab UI source code
├── requirements.txt              # Pip dependencies manifest
└── README.md                     # Project documentation
```

---

## 📊 Sample Output (JD Analyzer)

Below is an illustration of how the base model's unaligned output compares to HireSignal's aligned response:

### Input Job Description (Vague Startup JD)
```text
Full Stack Developer (Fresher Welcome!)
We are a fast-paced, dynamic startup disrupting the industry. We need a rockstar ninja developer who can wear many hats. You'll build our entire platform from scratch — frontend, backend, DevOps, database, mobile apps, AI/ML pipelines, and cloud infra. Must have 5+ years experience... Competitive salary. Work hard, play hard...
```

### 🔴 Base Model (microsoft/phi-2) Output
```text
Instruction: Analyze the following job description. Extract skills, experience level, responsibilities, and red flags.
Job Description: Full Stack Developer (Fresher Welcome!)
Response: If you want to apply, make sure you write code. You should write HTML, CSS, React, and build platforms. You also need to have five years of experience. We love self-starters. Why join? Because we are a startup! We wear many hats. (Loops or diverges into repetitive instructions)
```

### 🟢 HireSignal (SFT + RLHF Aligned) Output
```text
Title: Full Stack Developer (Fresher Welcome!)
Skills: React, Node.js, Python, AWS, Docker
Level: senior (Note: contradictions found with "Fresher Welcome!")
Responsibilities:
- Build entire platform from scratch (frontend, backend, DevOps, database, mobile, and cloud infra).
- Wear many hats in a fast-paced startup environment.
- Manage CI/CD pipelines, containerization, and vector databases.
Red Flags: Expected to handle multiple unrelated roles ('wear many hats'). Mentions high-pressure, 'fast-paced' environment. Vague compensation package advertised ('competitive salary'). Informal/unprofessional hiring terminology ('rockstar', 'ninja'). Potential toxic work culture indicator ('work hard play hard'). Possible lack of structured training/onboarding ('self-starter'). Vague/unbounded scope of work duties.
```

---

## 💡 Key Learnings

1. **QLoRA (Quantized Low-Rank Adaptation):** Training adapters in 4-bit quantization allowed us to run full causal language model fine-tuning on consumer-grade hardware, reducing VRAM footprint by ~75% without compromising downstream task performance.
2. **Preference Data Heuristics:** In the absence of expensive human recruiters, structure-based evaluation metrics (e.g., checking for strict bullet lists, presence of specific red-flag keywords, length constraints) allowed us to bootstrap a clean, reliable chosen/rejected preference dataset programmatically.
3. **Reward Modeling:** Using a lightweight classifier (`distilbert-base-uncased`) to score generated text sequence tokens proved effective at capturing structural features and alignment objectives (such as highlighting red flags vs. hallucinating recommendations).
4. **PPO (Proximal Policy Optimization):** RLHF alignment stabilized training outputs, aligning the causal generation towards strict formatting constraints. A KL-divergence penalty was crucial to prevent policy collapse and ensure the model maintained general language capabilities.

---

## 🔮 Future Work

* **DPO (Direct Preference Optimization):** Skip the intermediate reward modeling step entirely and align the LLM directly on chosen/rejected preference pairs to simplify hyperparameter tuning and save training compute.
* **LinkedIn/Indeed Scraper:** Introduce a browser companion or integration script to scrape and import target job descriptions directly into the Gradio UI.
* **Hugging Face Spaces Deployment:** Optimize model serialization to containerize and deploy the app demo on Hugging Face Spaces for public testing.
