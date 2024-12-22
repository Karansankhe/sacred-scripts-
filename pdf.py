from dotenv import load_dotenv
load_dotenv()  # load all the environment variables from .env

import os
from PIL import Image
import google.generativeai as genai
import gradio as gr
from io import BytesIO

# Configure Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the new Gemini model (gemini-1.5-flash)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(input, image, user_prompt):
    response = model.generate_content([input, image[0], user_prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Convert PIL Image to byte array
        img_byte_array = BytesIO()
        uploaded_file.save(img_byte_array, format='PNG')  # Save as PNG (or other formats like JPEG)
        img_byte_array = img_byte_array.getvalue()  # Get byte data from the byte stream

        image_parts = [
            {
                "mime_type": "image/png",  # Set the MIME type (or you can use jpeg if needed)
                "data": img_byte_array
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to handle the invoice extraction
def extract_invoice_details(input_prompt, uploaded_file):
    if uploaded_file is not None:
        image_data = input_image_details(uploaded_file)
        response = get_gemini_response(input_prompt, image_data, input_prompt)
        return response
    else:
        return "Please upload an invoice image."

# Define Gradio interface components
input_prompt = """
You are an expert in understanding invoices. We will upload an image as an invoice,
and you will have to answer any questions based on the uploaded invoice image.
"""

# Gradio Interface
iface = gr.Interface(
    fn=extract_invoice_details,
    inputs=[
        gr.Textbox(label="Input Prompt", placeholder="Enter your question about the invoice", value=""),
        gr.Image(type="pil", label="Upload Invoice Image")
    ],
    outputs="text",
    live=True,
    title="MultiLanguage Invoice Extractor",
    description="This tool extracts information from uploaded invoice images using AI."
)

# Launch the Gradio app
iface.launch(share=True)
