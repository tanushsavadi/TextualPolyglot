# Q&A Chatbot
# from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap
import time
import pandas as pd
from PIL import Image
import numpy as np

import google.generativeai as genai

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load OpenAI model and get responses

def get_gemini_responses(inputs, images, prompts):
    model = genai.GenerativeModel('gemini-1.5-flash')
    responses = []
    for input, image, prompt in zip(inputs, images, prompts):
        response = model.generate_content([input, image, prompt])
        response_parts = response.parts  # Use response.parts instead of response.text
        text_content = response_parts[0].text if response_parts else "No text found"
        responses.append(text_content)
    return responses

def input_images_setup(uploaded_files):
    image_parts_list = []
    for uploaded_file in uploaded_files:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = {
            "mime_type": uploaded_file.type,
            "data": bytes_data
        }
        image_parts_list.append(image_parts)
    return image_parts_list

##initialize our streamlit app

st.set_page_config(page_title="Gemini Image Demo")

st.header("TextualPolyglot")
##add a smaller description text below
st.write("TextualPolyglot is a tool that helps you ask questions about information from invoices, bills, newspapers, and other forms of imagery mediums in various languages. Google's Gemini engine powers the tool.")

inputs = st.text_area("Input Prompts (comma-separated):", key="inputs")
uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

images = []
captions = []

if uploaded_files is not None:
    for file in uploaded_files:
        image = Image.open(file)
        images.append(image)
        captions.append(f"Uploaded Image - {file.name}")

    st.image(images, caption=captions, use_column_width=True)

submit = st.button("Tell me about the images")

input_prompt = """
               You are an expert in understanding invoices, bills, newspapers and articles.
               You will receive input images as invoices, bills, newspapers, and other such mediums &
               you will have to answer questions based on the input image
               """

if submit:
    input_list = [input.strip() for input in inputs.split(',')]
    image_data_list = input_images_setup(uploaded_files)
    
    total_images = len(images)
    
    # Initialize progress bar
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    
    responses = get_gemini_responses(input_prompt, image_data_list, input_list)
    
    st.subheader("The Responses are")
    
    # Create a DataFrame to store the extracted information
    df_data = {"Image": [], "Response": []}
    
    for i, (response, image) in enumerate(zip(responses, images)):
        st.write(f"Response {i + 1}: {response}")
        
        # Update progress bar
        progress_value = (i + 1) / total_images
        progress_bar.progress(progress_value)
        # Add a small sleep to allow the UI to update
        time.sleep(0.2)
        
        # Append data to the DataFrame
        df_data["Image"].append(f"Image_{i + 1}")
        df_data["Response"].append(response)
    
    progress_placeholder.empty()  # Clear the progress bar when processing is complete
    
    # Convert the DataFrame to CSV
    df = pd.DataFrame(df_data)
    
    # Create a download link for the CSV file
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode(),
        file_name="extracted_data.csv",
        mime="text/csv",
    )
