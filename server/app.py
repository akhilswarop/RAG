from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import unicodedata
import subprocess
import numpy as np
import faiss
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdfminer.high_level import extract_text
from serpapi import GoogleSearch
from pydantic import BaseModel
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
nltk.download('punkt')

# Load models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
onet_titles_df = pd.read_csv('Occupation Data.csv')

def preprocess_documents(df):
    documents = []
    for _, row in df.iterrows():
        for sentence in sent_tokenize(row['Description']):
            documents.append({'title': row['Title'], 'text': sentence})
    return pd.DataFrame(documents)

documents_df = preprocess_documents(onet_titles_df)
embeddings = sentence_model.encode(documents_df['text'].tolist()).astype('float32')
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

def retrieve_documents(query, top_k=10):
    query_embedding = sentence_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return documents_df.iloc[indices[0]].to_dict(orient='records')

def get_resume_text(file_path):
    return unicodedata.normalize('NFKC', extract_text(file_path))

def parse_resume(resume_text):
    prompt = f"""Extract structured JSON data from this resume text:\n{resume_text}"""
    response = subprocess.run(["ollama", "run", "gemma2:2b", prompt], capture_output=True, text=True)
    return json.loads(response.stdout)

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Invalid file"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)
    return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

@app.route("/process-resume", methods=["POST"])
def process_resume():
    resume_file = next(iter(os.listdir(UPLOAD_FOLDER)), None)
    if not resume_file:
        return jsonify({"error": "No files found"}), 400
    resume_text = get_resume_text(os.path.join(UPLOAD_FOLDER, resume_file))
    parsed_resume = parse_resume(resume_text)
    return jsonify(parsed_resume), 200

@app.route("/career-guidance", methods=["POST"])
def career_guidance():
    data = request.json
    skills = ', '.join(data.get("skills", []))
    academic_history = data.get("academic_history", "")
    retrieved_docs = retrieve_documents(f"Skills: {skills}. Academic History: {academic_history}.")
    return jsonify(retrieved_docs), 200

@app.route("/google-jobs", methods=["POST"])
def google_jobs():
    data = request.json
    job_title, location = data.get("job_title"), data.get("location")
    search = GoogleSearch({"engine": "google_jobs", "q": f"{job_title} jobs in {location}", "hl": "en", "api_key": "YOUR_API_KEY"})
    return jsonify(search.get_dict().get("jobs_results", [])), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
