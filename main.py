from ortools.sat.python import cp_model
import google.generativeai as genai
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure GenAI API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Make sure it's set in your environment variables.")
else:
    genai.configure(api_key=api_key)

# Initialize GenAI
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Task Scheduling Optimization Function
def optimize_schedule(workers, tasks, priorities):
    """
    Optimizes task allocation to workers based on priorities using OR-Tools.
    """
    model = cp_model.CpModel()
    # Decision variables: Assign workers to tasks
    x = {}
    for w in range(len(workers)):
        for t in range(len(tasks)):
            x[(w, t)] = model.NewBoolVar(f"x[{w},{t}]")

    # Constraints: Each task must be assigned to exactly one worker
    for t in range(len(tasks)):
        model.Add(sum(x[w, t] for w in range(len(workers))) == 1)

    # Objective: Minimize priority-weighted allocation
    model.Minimize(
        sum(priorities[t] * x[w, t] for w in range(len(workers)) for t in range(len(tasks)))
    )

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        schedule = {}
        for w in range(len(workers)):
            for t in range(len(tasks)):
                if solver.Value(x[w, t]):
                    schedule[workers[w]] = tasks[t]
        return schedule
    else:
        return None

# AI-Powered Q&A Function
def get_gemini_response(question):
    """
    Get a response from the GenAI Gemini model for warehouse scheduling questions.
    """
    response = chat.send_message(question, stream=True)
    return " ".join([chunk.text for chunk in response])

# Gradio Task Optimization Interface
def gradio_optimize_schedule(workers, tasks, priorities):
    try:
        # Parse inputs
        workers = [w.strip() for w in workers.split(",")]
        tasks = [t.strip() for t in tasks.split(",")]
        priorities = list(map(int, priorities.split(",")))

        # Validate inputs
        if len(tasks) != len(priorities):
            return "Error: The number of tasks must match the number of priorities."

        # Optimize schedule
        schedule = optimize_schedule(workers, tasks, priorities)
        if schedule:
            return schedule
        else:
            return "No optimal schedule found."
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Q&A Interface
def gradio_qa(question):
    if question.strip():
        return get_gemini_response(question)
    else:
        return "Please enter a valid question."

# Create Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Scheduling and Optimization Tool")

    # Section 1: Task Scheduling
    with gr.Tab("Task Scheduling"):
        gr.Markdown("### Optimize Task Scheduling")
        workers_input = gr.Textbox(label="Workers (comma-separated)", placeholder="e.g., Alice, Bob, Charlie")
        tasks_input = gr.Textbox(label="Tasks (comma-separated)", placeholder="e.g., Pick Orders, Load Truck")
        priorities_input = gr.Textbox(label="Priorities (comma-separated)", placeholder="e.g., 3, 1, 2")
        schedule_output = gr.Textbox(label="Optimized Schedule", interactive=False)
        schedule_button = gr.Button("Optimize Schedule")

        schedule_button.click(
            gradio_optimize_schedule,
            inputs=[workers_input, tasks_input, priorities_input],
            outputs=schedule_output,
        )

    # Section 2: AI-Powered Q&A
    with gr.Tab("AI Q&A"):
        gr.Markdown("### Ask Questions About Scheduling")
        question_input = gr.Textbox(label="Your Question", placeholder="e.g., How do I handle peak warehouse hours?")
        response_output = gr.Textbox(label="AI Response", interactive=False)
        qa_button = gr.Button("Get AI Response")

        qa_button.click(gradio_qa, inputs=question_input, outputs=response_output)

# Launch Gradio Interface
demo.launch(share=True)
