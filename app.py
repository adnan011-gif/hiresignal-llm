"""
HireSignal Gradio Application.
Provides a web interface to paste a Job Description and a Resume,
then displays the matching score and recommended improvements.
"""

import gradio as gr
from src.inference import generate_feedback

def analyze_resume(job_description, resume):
    if not job_description.strip() or not resume.strip():
        return "Please provide both a Job Description and a Resume."
    
    # Run inference (currently placeholder)
    feedback = generate_feedback(job_description, resume, "outputs/sft_model")
    return feedback

# Build modern Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎯 HireSignal
        ### AI-Powered Resume Scoring and Optimization Platform (SFT + RLHF)
        """
    )
    with gr.Row():
        with gr.Column():
            jd_input = gr.Textbox(
                label="Job Description",
                placeholder="Paste the target job description here...",
                lines=8
            )
            resume_input = gr.Textbox(
                label="Resume / CV",
                placeholder="Paste the candidate resume text here...",
                lines=10
            )
            submit_btn = gr.Button("Analyze Match", variant="primary")
        with gr.Column():
            output_display = gr.Textbox(
                label="HireSignal Feedback",
                placeholder="Analysis results will appear here...",
                lines=20,
                interactive=False
            )
            
    submit_btn.click(
        fn=analyze_resume,
        inputs=[jd_input, resume_input],
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
