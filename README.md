# 🎯 HireSignal LLM

HireSignal is a project focused on fine-tuning a small Large Language Model (LLM) to analyze job descriptions, score candidate fit, and suggest targeted resume improvements. The system uses a multi-stage training pipeline incorporating **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning from Human Feedback (RLHF)** (or Direct Preference Optimization - DPO).

---

## 🚀 Project Goal

The primary goal of HireSignal is to build a highly specialized assistant that can:
1. **Analyze Job Descriptions**: Extract key skills, qualifications, and experience requirements.
2. **Score Candidate Fit**: Provide a structured match score indicating how well a resume aligns with a specific job description.
3. **Suggest Resume Improvements**: Offer actionable feedback and recommendations (e.g., adding metrics, rephrasing experience, adding missing skills) to improve candidate matching.

We achieve this by starting with a small base model (e.g., Llama-3-8B or Mistral-7B) and applying:
- **SFT**: To teach the model the base domain language, resume structure, and formatting.
- **RLHF / DPO**: To align model outputs with recruiter-preferred guidelines, ensuring suggestions are helpful, specific, and accurate.

---

## 🛠️ Tech Stack

- **Core Frameworks**: [PyTorch](https://pytorch.org/), [Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index)
- **Fine-Tuning Utilities**: [PEFT](https://github.com/huggingface/peft) (Parameter-Efficient Fine-Tuning, e.g., LoRA, QLoRA), [TRL](https://github.com/huggingface/trl) (Transformer Reinforcement Learning)
- **Hardware Acceleration / Quantization**: [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) (8-bit and 4-bit quantization), [Accelerate](https://github.com/huggingface/accelerate)
- **Experiment Tracking**: [Weights & Biases (wandb)](https://wandb.ai/)
- **Data Manipulation**: [Datasets (Hugging Face)](https://huggingface.co/docs/datasets/index), [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/)
- **User Interface**: [Gradio](https://gradio.app/)

---

## 📁 Folder Structure

```text
hiresignal-llm/
├── data/
│   ├── raw/                      # Raw resume and job description datasets
│   └── processed/                # Preprocessed and tokenized prompt-response pairs
├── src/
│   ├── data_prep.py              # Prepares raw data into instruction formats
│   ├── train_sft.py              # Performs Supervised Fine-Tuning (SFT)
│   ├── inference.py              # Standard inference pipeline for scoring and suggestions
│   ├── build_preference_data.py  # Builds chosen/rejected preference pairs for RLHF
│   ├── train_reward_model.py     # Trains the Reward Model using preference datasets
│   └── train_ppo.py              # Runs PPO optimization on the SFT model using the Reward Model
├── notebooks/                    # Jupyter notebooks for EDA and prototyping
├── outputs/                      # Saved checkpoints, models, and evaluation outputs
├── app.py                        # Gradio web interface for interactive usage
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 💻 How to Run Locally

### 1. Set Up Environment
First, clone the repository and navigate to the directory:
```bash
git clone https://github.com/adnan011-gif/hiresignal-llm.git
cd hiresignal-llm
```

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
python src/data_prep.py
```

### 3. Run Fine-Tuning (SFT)
```bash
python src/train_sft.py
```

### 4. Run the Web Application
```bash
python app.py
```
This will launch a Gradio interface locally, accessible at `http://localhost:7860`.
