
import streamlit as st
import torch
import numpy as np
import joblib
from transformers import DistilBertTokenizer, DistilBertModel

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load the trained classifier and label encoder
rf_classifier = joblib.load('/content/random_forest_model.pkl')
label_encoder = joblib.load('/content/label_encoder.pkl')

# Function to extract DistilBERT embeddings
def get_distilbert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.numpy().flatten()

# Streamlit app
st.title("Resume Job Classification")
st.write("Upload your resume and find out which job category it matches best!")

# File upload widget
uploaded_file = st.file_uploader("Upload a resume", type=["txt"])

if uploaded_file is not None:
    # Read the resume text
    try:
        resume_text = uploaded_file.read().decode('utf-8')

        # Extract features using DistilBERT
        embedding = get_distilbert_embedding(resume_text).reshape(1, -1)

        # Make a prediction
        prediction = rf_classifier.predict(embedding)
        job_category = label_encoder.inverse_transform(prediction)[0]

        # Display the result
        st.success(f"The predicted job category is: {job_category}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
    