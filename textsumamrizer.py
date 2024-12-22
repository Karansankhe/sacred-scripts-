import os
from dotenv import load_dotenv
import gradio as gr
from transformers import pipeline

load_dotenv()

summarizer = pipeline("summarization", model="t5-small")
def generate_summary(forensic_data):
    if not forensic_data.strip():
        return "Please provide valid forensic log data."
    
    prompt = f"Summarize the following forensic activities: {forensic_data}"
    
    summary = summarizer(prompt, max_length=150, min_length=30, do_sample=False)
    
    return summary[0]['summary_text']

iface = gr.Interface(
    fn=generate_summary,
    inputs=gr.Textbox(label="Forensic Log Data", lines=10, placeholder="Enter forensic data (e.g., system logs, file activity)"),
    outputs="text",
    title="Automated Forensic Data Summarization",
    description="This tool summarizes forensic activities from raw log data using T5."
)
iface.launch(share=True)
