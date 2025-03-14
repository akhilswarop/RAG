from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import os
import json
import unicodedata
import subprocess
from typing import List
from pydantic import BaseModel
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
from ollama import chat  # Assuming you are using Ollama's API for LLM interaction
import spacy
import pandas as pd
from typing import List
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import subprocess
import faiss
import numpy as np
import time
import json
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from pdfminer.high_level import extract_text
from serpapi import GoogleSearch
from ollama import chat
from pydantic import BaseModel
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, JsonCorrectnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from deepeval.models import DeepEvalBaseLLM
import json
import winsound
import transformers
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from rouge_score import rouge_scorer
from bert_score import score as score_bert
from bleurt import score as score_bleurt

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Temporary storage for emails (use a database in production)
emails = []
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def get_first_file():
    """Get the first PDF file in the uploads folder."""
    print(f"Checking directory: {os.path.abspath(UPLOAD_FOLDER)}")
    print("Folder exists:", os.path.exists(UPLOAD_FOLDER))
    print("Files in folder:", os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else "Folder does not exist")
    files = [f for f in os.listdir(UPLOAD_FOLDER)]
    print("Files in uploads folder:", files)
    return os.path.join(UPLOAD_FOLDER, files[0]) if files else None

def clean_text(text: str) -> str:
            """
            Cleans the input text by normalizing and removing unwanted characters.
            """
            # Normalize the text to NFKC form
            normalized_text = unicodedata.normalize('NFKC', text)

            # Remove non-printable characters
            cleaned_text = ''.join(c for c in normalized_text if c.isprintable())

            return cleaned_text


def get_resume_text(file_path: str) -> str:
            """
            Extracts and cleans text from a PDF resume.

            Parameters:
            - file_path (str): Path to the PDF file.

            Returns:
            - str: Cleaned text extracted from the PDF.
            """
            print("File Path:", file_path)
            print(f"Checking file: {file_path}")
            print(f"Normalized path: {os.path.normpath(file_path)}")
            print(f"File exists: {os.path.exists(file_path)}")
            
            if file_path.lower().endswith('.pdf'):
                try:
                    # Extract text using pdfminer.high_level.extract_text
                    text = extract_text(file_path)
        
                    # Clean the extracted text
                    cleaned_text = clean_text(text)
        
                    return cleaned_text
                except Exception as e:
                    print(f"Error extracting text from PDF: {e}")
                    return ""
            else:
                raise ValueError("Unsupported file format. Only .pdf is supported.")



class Experience(BaseModel):
    role: str
    company: str
    description: str


class Project(BaseModel):
    title: str
    description: str


class Education(BaseModel):
    degree: str
    institution: str


class ExtracurricularActivity(BaseModel):
    activity: str
    description: str


class Resume(BaseModel):
    name: str
    location: str
    email: str
    phone: str
    linkedin: str
    skills: List[str]
    experience: List[Experience]
    projects: List[Project]
    education: List[Education]
    extracurricular_activities: List[ExtracurricularActivity]


# Function to process the resume using pyresparser
def parse_resume_with_ollama(resume_text):
    '''
    Function to parse resume text using Ollama's gemma2:2b model and return structured JSON data.
    '''

   
            # Use pyresparser to extract data from the resume text
    extracted_data = resume_text
    # Define the prompt with strict instructions
    prompt = f"""
You are a helpful assistant that extracts structured data from the given resume text.

Important Instructions:
1. Output Format: Return only a single JSON object that strictly follows the requested structure.
2. No Extra Text: Do not include any additional text, explanations, code fences, triple backticks, or any formatting beyond the JSON object.
3. No Missing Keys: Include all keys listed below, even if their values are empty or blank.
4. No Trailing Commas: Ensure that there are no trailing commas after the last item in arrays or objects.
5. Data Structure:
   - name: string
   - location: string
   - email: string
   - phone: string
   - linkedin: string
   - skills: an array of strings
   - experience: an array of objects, each with keys "role", "company", "location", "start_date", "end_date", "description"
   - projects: an array of objects, each with keys "title", "start_date", "end_date", "description", "tech_stack" (where "tech_stack" is an array of strings)
   - education: an array of objects, each with keys "degree", "institution", "start_date", "end_date", "gpa"
   - extracurricular_activities: an array of objects, each with keys "activity" and "description"
6. Strictly follow the structure in step 5. Do not create new keys by yourself. Use only the keys I mentioned in step 5. 
7. You are part of a resume parsing pipeline so it's really important you return a json only object and again. Strictly follow the key names in step 5. 
If a field is not found in the resume, return an empty string "" for strings or an empty array [] for lists.

Resume Text:
{extracted_data}

Your task:
Extract the requested information from the resume text and return only one valid JSON object, strictly following the structure and instructions above. Do not add extra fields or omit the ones given in the structure.
"""

    # Call Ollama's gemma2:2b model via subprocess
    try:
            response = chat(
                    messages=[
                        {
                        'role': 'user',
                        'content': prompt,
                        }
                    ],
                    model='finetuned',
                    format=Resume.model_json_schema(),
                    )
            result = Resume.model_validate_json(response.message.content)
    except subprocess.CalledProcessError as e:
            print("Error running ollama:", e.stderr)
            return None
    # Try to parse the cleaned JSON
    try:
        parsed = result
        return parsed
    except json.JSONDecodeError as e:
        print("JSON parsing failed:")
        print(e)
        

def load_spacy_model():
    return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Load Sentence Transformer model for embeddings
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sentence_transformer()

# Load O*NET job titles and descriptions
def load_onet_data():
    try:
        df = pd.read_csv('../Occupation Data.csv')
        print(df)
        return df
# Ensure this file contains 'Title' and 'Description' columns        return df
    except FileNotFoundError:
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

def prepare_faiss_index():
    documents_df = preprocess_documents(onet_titles_df)
    if documents_df.empty:
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
        return []

    # Encode the query
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = documents_df.iloc[indices[0]]
        # Collect the SOC Code, Title, and Text
    print("==========RETRIEVED DOCUMENTS===============")
    print(retrieved_docs)
    results = []
    results_json = []
    for _, row in retrieved_docs.iterrows():
        results.append(
        f"SOC Code: {row['soc_code']}, Title: {row['job_title']}, Description: {row['text']}"
        )
        results_json.append({
            "soc_code": row['soc_code'],
            "title": row['job_title'],
            "description": row['text']
        })
    print("==========TOP K CLOSEST DOCUMENTS===============")
    print(results_json)
    return results, results_json

def generate_career_guidance(skills, academic_history):
    
    # Generations and Evaluations 
    generations = {}
            
    skills_text = ', '.join(skills) if skills else "No skills provided"

    # Create a comprehensive query based on user profile
    query = f"Skills: {skills_text}. Academic History: {academic_history}."

    # Retrieve relevant documents
    retrieved_docs, top_job_titles = retrieve_relevant_documents(query, top_k=10)
    context = "\n".join(retrieved_docs)
    print("=============== CONTEXT ================")
    print(context)
    # Construct the prompt for Mistral
    prompt = f"""
You are an AI career advisor. Use the following context to provide personalized career guidance.

Context:
{context}

User Profile:
Skills: {skills_text}
Academic History: {academic_history}


Provide a comprehensive analysis including:
- Recommended career paths with SOC codes and titles
- Skill development suggestions
- Potential industries to explore
- Next steps for job applications
"""

    try:
            generations["gemma2_2b"] = subprocess.run(
                ["ollama", "run", "gemma2:2b", prompt],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            
            
            generations["gemma2_9b"] = subprocess.run(
                ["ollama", "run", "gemma2:9b", prompt],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            generations["mistral"] = subprocess.run(
                ["ollama", "run", "mistral", prompt],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            generations["deepseek-r1:14b"] = subprocess.run(
                ["ollama", "run", "mistral", prompt],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            


        
            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")  # Print the actual error message
                
    return  generations, top_job_titles   
        
        
        
def google_jobs_search(job_title, location = "Remote"):

    params = {
    "engine": "google_jobs",
    "q": f"{job_title} jobs in {location}",
    "hl": "en",
    "api_key": "22c744e7201db68cce330bf72f58d1c9a81529af3361865d333daa41a32e1551" 
    }

    

    search = GoogleSearch(params)
    results = search.get_dict()
    search_results = results["jobs_results"]

    jobs_data = []
    for job in search_results:
        jobs_data.append({
            "Title": job.get("title", ""),
            "Company": job.get("company_name", ""),
            "Location": job.get("location", ""),
            "Via": job.get("via", ""),
            "Description (truncated)": job.get("description", "")[:100] + "...",
            "Apply Options": job.get("apply_options", "")
        })

    return jobs_data


































        

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith(".pdf"):
        # Delete any existing file in uploads/
        for existing_file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, existing_file))

        # Save the new file
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(file_path)

        return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

    return jsonify({"error": "Unsupported file format. Only .pdf is allowed."}), 400














@app.route("/process-resume", methods=["POST"])
def process_resume():
    """Finds the first resume in 'uploads' folder and processes it."""
    resume_file = get_first_file()
    print("Resume file:", resume_file)
    if not resume_file:
        return jsonify({"error": "No files found in uploads folder"}), 400

    resume_text = get_resume_text(resume_file)

    if not resume_text:
        return jsonify({"error": "Failed to extract text from resume"}), 500

    parsed_resume = parse_resume_with_ollama(resume_text)

    print("Parsed resume:", parsed_resume)
    
    return jsonify(parsed_resume.model_dump() if parsed_resume else {"error": "Failed to parse resume"}), 200














@app.route("/generate-guidance", methods=["POST"])

def generate_guidance():
    
    data = request.json
    skills = data.get("skills", [])
    academic_history = data.get("academic_history", "")

    generations, top_job_titles = generate_career_guidance(skills, academic_history)
        
    return jsonify({"generations": generations, "top_job_titles": top_job_titles})











@app.route("/job-search", methods=["POST"])

def search_jobs():
    data = request.json
    job_titles = data.get("jobs")
    print("Job Titles", job_titles)
    postings = []
    postings.append(google_jobs_search(job_titles[0]))
    print("Postings", postings)
    return jsonify(postings)  
    
if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
