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
import streamlit as st  # Assuming you are using Streamlit for some UI interactions
from ollama import chat  # Assuming you are using Ollama's API for LLM interaction


app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Temporary storage for emails (use a database in production)
emails = []
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/store-email', methods=['POST'])
def store_email():
    data = request.json
    email = data.get("email")
    
    if email:
        emails.append(email)  # Store email
        print("Stored Emails:", emails)  # Debugging print statement
        return jsonify({"message": "Email stored successfully"}), 200
    
    return jsonify({"error": "Invalid email"}), 400

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
    extracted_data = get_resume_text(resume_text)
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
    with st.spinner('Parsing resume...'):
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
        
        
if __name__ == '__main__':
    app.run(debug=True, port=5000)
