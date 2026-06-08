"""
Data Preparation Script for HireSignal LLM Fine-Tuning.
Downloads the job descriptions dataset from Hugging Face, explores, cleans,
extracts structured fields using rule-based heuristics, formats into
instruction-response pairs, and saves train/val splits.
"""

import os
import re
import json
import random
import pandas as pd
from datasets import load_dataset

def clean_text(text):
    """Clean extra spaces and newlines from text."""
    if not isinstance(text, str):
        return ""
    # Replace multiple newlines or spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(description, model_response):
    """
    Extract top 5 skills using model_response JSON if valid,
    falling back to keyword search in description.
    """
    skills = []
    
    # Try parsing model_response JSON
    if isinstance(model_response, str) and model_response.strip():
        try:
            parsed = json.loads(model_response)
            # Check common keys for skills
            for key in ["Required Skills", "Preferred Qualifications", "Core Responsibilities"]:
                if key in parsed and isinstance(parsed[key], str):
                    # Split by common delimiters
                    items = re.split(r'[,.;•\-\n]', parsed[key])
                    for item in items:
                        cleaned = clean_text(item)
                        if cleaned and len(cleaned) > 2 and len(cleaned) < 50:
                            skills.append(cleaned)
        except Exception:
            pass

    # Fallback/supplement: keyword matching for common technical & soft skills
    common_skills = [
        "python", "java", "javascript", "c\\+\\+", "sql", "aws", "docker", "kubernetes",
        "react", "angular", "node\\.js", "machine learning", "deep learning", "nlp",
        "excel", "tableau", "salesforce", "project management", "agile", "scrum",
        "communication", "leadership", "problem solving", "analytics", "marketing",
        "product management", "design", "git", "cloud", "saas", "devops", "linux"
    ]
    
    desc_lower = description.lower()
    for skill in common_skills:
        if re.search(r'\b' + skill + r'\b', desc_lower):
            # Format nicely
            skills.append(skill.title() if len(skill) > 3 else skill.upper())

    # Deduplicate and limit to top 5
    unique_skills = []
    for s in skills:
        if s not in unique_skills:
            unique_skills.append(s)
        if len(unique_skills) == 5:
            break
            
    # Default skills if empty
    if not unique_skills:
        unique_skills = ["Communication", "Problem Solving", "Teamwork", "Adaptability", "Organization"]
        
    return ", ".join(unique_skills)

def determine_experience_level(description, model_response):
    """
    Determine experience level (fresher/mid/senior) based on description and model response.
    """
    text_to_search = f"{description} {model_response}".lower()
    
    # Check for years of experience
    years = re.findall(r'(\d+)\s*\+?\s*years?', text_to_search)
    max_years = 0
    if years:
        try:
            max_years = max(int(y) for y in years if int(y) < 25)
        except ValueError:
            pass

    # Check titles/keywords
    senior_keywords = ["senior", "lead", "principal", "manager", "director", "architect", "head", "vp"]
    fresher_keywords = ["fresher", "junior", "entry level", "entry-level", "intern", "graduate", "assistant"]
    
    is_senior = any(re.search(r'\b' + kw + r'\b', text_to_search) for kw in senior_keywords) or max_years >= 5
    is_fresher = any(re.search(r'\b' + kw + r'\b', text_to_search) for kw in fresher_keywords) or (max_years > 0 and max_years <= 2)
    
    if is_senior and not is_fresher:
        return "senior"
    elif is_fresher and not is_senior:
        return "fresher"
    else:
        # Default or mixed case
        if max_years >= 5:
            return "senior"
        elif max_years > 0 and max_years <= 2:
            return "fresher"
        return "mid"

def extract_responsibilities(description, model_response):
    """
    Extract 3 core responsibilities in bullet points.
    """
    responsibilities = []
    
    # Try parsing model_response JSON first
    if isinstance(model_response, str) and model_response.strip():
        try:
            parsed = json.loads(model_response)
            if "Core Responsibilities" in parsed and isinstance(parsed["Core Responsibilities"], str):
                items = re.split(r'[;•\-\n]', parsed["Core Responsibilities"])
                for item in items:
                    cleaned = clean_text(item)
                    if cleaned and len(cleaned) > 15:
                        responsibilities.append(cleaned)
        except Exception:
            pass
            
    # Fallback: extract sentences from description that look like responsibilities
    if len(responsibilities) < 3:
        sentences = re.split(r'[.!?\n]', description)
        action_verbs = [
            "manage", "lead", "develop", "create", "build", "design", "maintain", 
            "support", "collaborate", "ensure", "drive", "coordinate", "implement",
            "responsible for", "oversee", "analyze", "monitor", "write", "deliver"
        ]
        for sent in sentences:
            sent_clean = clean_text(sent)
            if len(sent_clean) > 25:
                # Check if starts with or contains action verbs
                sent_lower = sent_clean.lower()
                if any(verb in sent_lower[:30] for verb in action_verbs):
                    responsibilities.append(sent_clean)
                    if len(responsibilities) == 5: # get a pool
                        break

    # Clean up and ensure we have exactly 3
    final_bullets = []
    for r in responsibilities:
        # Remove leading symbols/dashes
        r_clean = re.sub(r'^[•\-\*\d\.\s]+', '', r).strip()
        if r_clean and r_clean not in final_bullets:
            final_bullets.append(r_clean)
            
    if len(final_bullets) < 3:
        # Fallback list if we couldn't find enough sentences
        fallbacks = [
            "Execute day-to-day operations and task management for the role.",
            "Collaborate with cross-functional teams to deliver project objectives.",
            "Maintain high quality standards and operational excellence."
        ]
        while len(final_bullets) < 3:
            final_bullets.append(fallbacks[len(final_bullets)])
            
    return final_bullets[:3]

def detect_red_flags(description):
    """
    Identify potential vague demands or red flags in the job description.
    """
    flags = []
    desc_lower = description.lower()
    
    rules = {
        "wear many hats": "Expected to handle multiple unrelated roles ('wear many hats').",
        "fast-paced": "Mentions high-pressure, 'fast-paced' environment.",
        "fast paced": "Mentions high-pressure, 'fast-paced' environment.",
        "competitive salary": "Vague compensation package advertised ('competitive salary').",
        "rockstar": "Informal/unprofessional hiring terminology ('rockstar').",
        "ninja": "Informal/unprofessional hiring terminology ('ninja').",
        "guru": "Informal/unprofessional hiring terminology ('guru').",
        "work hard play hard": "Potential toxic work culture indicator ('work hard play hard').",
        "work hard, play hard": "Potential toxic work culture indicator ('work hard play hard').",
        "self-starter": "Possible lack of structured training/onboarding ('self-starter').",
        "self starter": "Possible lack of structured training/onboarding ('self-starter').",
        "other duties as assigned": "Vague/unbounded scope of work duties.",
        "flexible hours": "Potential expectation of 24/7 availability ('flexible hours')."
    }
    
    for kw, msg in rules.items():
        if kw in desc_lower:
            flags.append(msg)
            
    if not flags:
        return "None identified."
        
    return " ".join(flags)

def main():
    print("=== Step 1: Downloading Dataset from Hugging Face ===")
    dataset_name = "jacob-hugging-face/job-descriptions"
    # Download split train[:4000] for MVP size
    dataset = load_dataset(dataset_name, split="train[:4000]")
    
    # Convert to pandas DataFrame for easy cleaning/exploration
    df = pd.DataFrame(dataset)
    
    print("\n=== Step 2: Exploring Dataset ===")
    print(f"Shape of downloaded dataset: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nSample row 1:")
    print(df.iloc[0].to_dict())
    print("\nSample row 2:")
    print(df.iloc[1].to_dict())
    
    print("\n=== Step 3: Cleaning Dataset ===")
    initial_len = len(df)
    
    # 1. Keep only rows where job description and title are non-empty
    df = df.dropna(subset=['job_description', 'position_title'])
    df = df[(df['job_description'].str.strip() != '') & (df['position_title'].str.strip() != '')]
    
    # 2. Remove duplicates
    df = df.drop_duplicates(subset=['job_description'])
    
    # 3. Strip extra whitespace and newlines
    df['job_description'] = df['job_description'].apply(clean_text)
    df['position_title'] = df['position_title'].apply(clean_text)
    
    # 4. Remove rows where job description is under 100 characters
    df = df[df['job_description'].str.len() >= 100]
    
    print(f"Cleaned dataset size: {len(df)} (Removed {initial_len - len(df)} rows)")
    
    print("\n=== Step 4 & 5: Formatting and Auto-generating Responses ===")
    formatted_data = []
    
    for idx, row in df.iterrows():
        jd = row['job_description']
        title = row['position_title']
        model_resp = row.get('model_response', '')
        
        # Extract fields using rule-based functions
        skills = extract_skills(jd, model_resp)
        level = determine_experience_level(jd, model_resp)
        bullets = extract_responsibilities(jd, model_resp)
        red_flags = detect_red_flags(jd)
        
        # Format responsibilities
        resp_bullets = "\n".join([f"- {b}" for b in bullets])
        
        # Instruction block template
        instruction = (
            "Analyze the following job description. Extract:\n"
            "1. Top 5 required skills\n"
            "2. Experience level expected (fresher/mid/senior)\n"
            "3. Key responsibilities in 3 bullet points\n"
            "4. Red flags or vague requirements if any"
        )
        
        # Response block template
        response = (
            f"Title: {title}\n"
            f"Skills: {skills}\n"
            f"Level: {level}\n"
            f"Responsibilities:\n"
            f"{resp_bullets}\n"
            f"Red Flags: {red_flags}"
        )
        
        # SFT formatted dict
        formatted_data.append({
            "instruction": instruction,
            "input": jd,
            "response": response
        })
        
    # Convert to DataFrame
    formatted_df = pd.DataFrame(formatted_data)
    
    print("\n=== Step 6: Splitting into Train and Validation Sets ===")
    # Split 90% train and 10% val
    shuffled_df = formatted_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.9 * len(shuffled_df))
    
    train_df = shuffled_df.iloc[:split_idx]
    val_df = shuffled_df.iloc[split_idx:]
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    
    print("\n=== Step 7: Saving Processed Datasets ===")
    os.makedirs("data/processed", exist_ok=True)
    
    # Save as JSON records
    train_path = "data/processed/train.json"
    val_path = "data/processed/val.json"
    
    train_df.to_json(train_path, orient="records", indent=2)
    val_df.to_json(val_path, orient="records", indent=2)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved validation data to {val_path}")
    
    print("\n=== Step 8: Final Stats ===")
    total_samples = len(shuffled_df)
    avg_len = shuffled_df['input'].str.len().mean()
    print(f"Total processed samples: {total_samples}")
    print(f"Average job description length: {avg_len:.2f} characters")
    
    print("\nSample SFT Instance:")
    print(json.dumps(formatted_data[0], indent=2))

if __name__ == "__main__":
    main()
