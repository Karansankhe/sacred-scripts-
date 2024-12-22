import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langdetect import detect  # Language detection library

# Load environment variables from .env file
load_dotenv()

# Configure Google API with the provided API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize the Google Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to get a response from Google Gemini
def get_gemini_response(content, prompt, language):
    """
    This function generates a response using Google Gemini model.
    
    Parameters:
    - content (str): The user's message.
    - prompt (str): A fixed or dynamic prompt to guide the response generation.
    - language (str): The detected language of the user's input.
    
    Returns:
    - str: The generated response text.
    """
    # If the language is not English, include a translation instruction
    if language != 'en':
        prompt += f" Answer in {language}."

    # Generate a response using the Gemini model
    response = model.generate_content([content, prompt])
    return response.text

# Function to handle the chat and respond in the user's input language
def handle_chat(user_message):
    """
    This function handles the user's chat input, detects language, and generates a response.
    
    Parameters:
    - user_message (str): The user's chat message.
    
    Returns:
    - str: The chatbot's response to the user's message.
    """
    # Detect the language of the user's input
    language = detect(user_message)
    
    # Fixed prompt for generating a response
    prompt = f"""
    You are a helpful chatbot providing advice.
    User's Message: {user_message}
    """
    
    # Get the response from Google Gemini
    response = get_gemini_response(user_message, prompt, language)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Interactive Multilingual Chatbot")

st.markdown("""
This chatbot is powered by Google Gemini. It can respond to your messages in **multiple languages**.
You can enter your message in any language, and the chatbot will understand and reply in the same language.
""")

# Chat input section
user_message = st.text_input("Enter your message:")

# When the user clicks the button to get a response
if st.button("Get Response"):
    if user_message:
        # Generate response using the handle_chat function
        response = handle_chat(user_message)
        
        # Display chatbot's response
        st.subheader("Chatbot Response:")
        st.write(response)
    else:
        st.error("Please enter a message.")
