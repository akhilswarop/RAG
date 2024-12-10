import streamlit as st
import torch
import spacy
import pandas as pd
from typing import List
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
import os
import subprocess
import sys
import faiss
import numpy as np

# Added Imports for pyresparser
sys.path.append('C:/Users/swaro/OneDrive/Documents/GitHub/RAG/pyresparser')  # Adjust the path as needed
sys.path.append('C:/Users/swaro/OneDrive/Documents/GitHub/RAG/linkedin_scraper')  # Adjust the path as needed
from pyresparser import ResumeParser

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Set page configuration
st.set_page_config(page_title="Career Guidance System using RAG with Mistral", layout="wide")
st.title("Career Guidance System using RAG with Mistral")

# Initialize variables with default values
skills = []
academic_history = ""
psychometric_profile = ""
jobs_data = []

# Load spaCy model for NLP tasks
@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Load Sentence Transformer model for embeddings
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sentence_transformer()

# Load O*NET job titles and descriptions
@st.cache_data
def load_onet_data():
    try:
        df = pd.read_csv('Occupation Data.csv')  # Ensure this file contains 'Title' and 'Description' columns
        return df
    except FileNotFoundError:
        st.error("The file 'Occupation Data.csv' was not found. Please ensure it's in the correct directory.")
        return pd.DataFrame()

onet_titles_df = load_onet_data()
onet_titles = onet_titles_df['Title'].tolist()

# Function to preprocess documents for FAISS
def preprocess_documents(df, text_column='Description'):
    documents = []
    for idx, row in df.iterrows():
        text = row[text_column]
        sentences = sent_tokenize(text)
        for sentence in sentences:
            documents.append({
                'job_title': row['Title'],
                'soc_code': row['O*NET-SOC Code'],  # Include the SOC code
                'text': sentence
            })
        
    return pd.DataFrame(documents)

@st.cache_resource
def prepare_faiss_index():
    documents_df = preprocess_documents(onet_titles_df)
    if documents_df.empty:
        st.error("No documents available for indexing.")
        return None, None

    # Initialize the Sentence Transformer model
    retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    embeddings = retrieval_model.encode(documents_df['text'].tolist(), convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity after normalization

    # Add embeddings to the index
    index.add(embeddings)

    return index, documents_df

# Load or prepare FAISS index and documents
index, documents_df = prepare_faiss_index()

# Function to retrieve relevant documents using FAISS
def retrieve_relevant_documents(query, top_k=10):
    if index is None or documents_df is None:
        st.error("The FAISS index or documents dataframe is not available.")
        return []

    # Encode the query
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = documents_df.iloc[indices[0]]
        # Collect the SOC Code, Title, and Text
    
    results = []
    for _, row in retrieved_docs.iterrows():
        results.append(
        f"SOC Code: {row['soc_code']}, Title: {row['job_title']}, Description: {row['text']}"
        )
    return results

# Function to process the resume using pyresparser
def process_resume(file_path):
    try:
        data = ResumeParser(file_path).get_extracted_data()
        return data
    except Exception as e:
        st.error(f"An error occurred while parsing the resume: {e}")
        return {}

# Function to generate career guidance using RAG with Mistral
def generate_career_guidance_rag_mistral(skills, academic_history, psychometric_profile, top_job_titles):
    skills_text = ', '.join(skills) if skills else "No skills provided"
    top_job_titles_text = ', '.join(top_job_titles) if top_job_titles else "No job titles provided"

    # Create a comprehensive query based on user profile
    query = f"Skills: {skills_text}. Academic History: {academic_history}. Psychometric Profile: {psychometric_profile}. Top Job Titles: {top_job_titles_text}."

    # Retrieve relevant documents
    retrieved_docs = retrieve_relevant_documents(query, top_k=10)
    context = "\n".join(retrieved_docs)

    # Construct the prompt for Mistral
    prompt = f"""
You are an AI career advisor. Use the following context to provide personalized career guidance.

Context:
{context}

User Profile:
Skills: {skills_text}
Academic History: {academic_history}
Top Recommended Job Titles: {top_job_titles_text}


Provide a comprehensive analysis including:
- Recommended career paths with SOC codes and titles
- Skill development suggestions
- Potential industries to explore
- Next steps for job applications
"""

    try:
        with st.spinner("Generating personalized career guidance..."):
            result_mistral = subprocess.run(
                ["ollama", "run", "mistral", prompt],
                capture_output=True,
                text=True,
                check=True
            )
            result_gemma_2b = subprocess.run(
                ["ollama", "run", "gemma2:2b", prompt],
                capture_output=True,
                text=True,
                check=True
            )
            result_gemma_9b = subprocess.run(
                ["ollama", "run", "gemma2:9b", prompt],
                capture_output=True,
                text=True,
                check=True
            )
            guidance_mistral = result_mistral.stdout.strip()
            guidance_gemma_2b = result_gemma_2b.stdout.strip()
            guidance_gemma_9b = result_gemma_9b.stdout.strip()
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred while generating guidance: {e.stderr}")
        guidance = "We're sorry, but we couldn't generate your career guidance at this time. Please try again later."

    return guidance_mistral, guidance_gemma_2b, guidance_gemma_9b

# Main application logic

# File Upload and Automatic Resume Processing
st.header("Complete Your Profile and Generate Guidance")
st.subheader("Upload Your Resume")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"], key="resume_upload")

# Initialize session state variables
if "resume_data" not in st.session_state:
    st.session_state.resume_data = {}
if "name" not in st.session_state:
    st.session_state.name = ""
if "email" not in st.session_state:
    st.session_state.email = ""
if "mobile_number" not in st.session_state:
    st.session_state.mobile_number = ""
if "skills" not in st.session_state:
    st.session_state.skills = ""
if "degree" not in st.session_state:
    st.session_state.degree = ""
if "experience" not in st.session_state:
    st.session_state.experience = ""
if "college_name" not in st.session_state:
    st.session_state.college_name = ""

# Process the resume and update session state
if uploaded_file:
    st.success("Resume uploaded successfully!")
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract resume data
    resume_data = process_resume(uploaded_file.name)
    if resume_data:
        st.session_state.resume_data = resume_data
        st.session_state.name = resume_data.get("name", "")
        st.session_state.email = resume_data.get("email", "")
        st.session_state.mobile_number = resume_data.get("mobile_number", "")
        st.session_state.skills = resume_data.get("skills", [])
        st.session_state.degree = resume_data.get("degree", [])
        st.session_state.experience = resume_data.get("experience", "")
        st.session_state.college_name = resume_data.get("college_name", "")
        st.success("Resume processed successfully!")
    else:
        st.error("Failed to process the resume. Please try again.")

# Combined User Profile and Location Form
with st.form("career_guidance_form"):
    # Academic History Autofill
    st.subheader("Academic History")
    name = st.text_input("Name", value=st.session_state.name)
    email = st.text_input("Email", value=st.session_state.email)
    mobile_number = st.text_input("Mobile Number", value=st.session_state.mobile_number)
    degree = st.text_area("Degree", value=', '.join(st.session_state.degree) if isinstance(st.session_state.degree, list) else st.session_state.degree)
    institution = st.text_input("Institution", value=st.session_state.college_name)
    graduation_year = st.number_input("Graduation Year", min_value=1950, max_value=2100, step=1)

    # Skills Autofill
    st.subheader("Skills")
    skills = st.text_area("Skills", value=', '.join(st.session_state.skills) if isinstance(st.session_state.skills, list) else st.session_state.skills)

    # Experience Autofill
    st.subheader("Experience")
    experience = st.text_area("Experience", value=st.session_state.experience)

    # Submit Button
    submit = st.form_submit_button("Submit and Generate Guidance")


# Handle form submission
if submit:
    if not (degree and institution and graduation_year):
        st.error("Please fill in all the required profile fields.")
    else:
        # Process academic history
        academic_history = f"Degree: {degree}, Institution: {institution}, Graduation Year: {graduation_year}"

        # Process psychometric profile

        st.success("Profile information submitted successfully!")

        # Ensure skills are in list format
        user_skills = [skill.strip() for skill in skills.split(',')] if skills else []

        # Generate career guidance using RAG with Mistral
        guidance_mistral, guidance_gemma_2b, guidance_gemma_9b = generate_career_guidance_rag_mistral(
            skills=user_skills,
            academic_history=academic_history,
            psychometric_profile=psychometric_profile,
            top_job_titles=[],  # Since skill matching is removed
        )

        st.subheader("Career Guidance [Mistral]:")
        st.write(guidance_mistral)

        st.subheader("Career Guidance [Gemma 2B]:")
        st.write(guidance_gemma_2b)

        st.subheader("Career Guidance [Gemma 9B]:")
        st.write(guidance_gemma_9b)
