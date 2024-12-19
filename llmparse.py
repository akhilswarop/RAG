import os
import argparse
import pandas as pd
import json
from typing import List
from pydantic import BaseModel, ValidationError
from pdfminer.high_level import extract_text
import unicodedata
import subprocess
from urllib.parse import urlparse
from ollama import chat  # Ensure you have
import csv

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
    try:
        response = chat(
                messages=[
                    {
                    'role': 'user',
                    'content': prompt,
                    }
                ],
                model='gemma2:2b',
                format=Resume.model_json_schema(),
                )
        result = Resume.model_validate_json(response.message.content)
    except subprocess.CalledProcessError as e:
        print("Error running ollama:", e.stderr)
        return None  
    # Try to parse the cleaned JSON
    try:
        parsed = result
        return extracted_data, parsed
    except json.JSONDecodeError as e:
        print("JSON parsing failed:")
        print(e)




def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process PDF resumes and generate JSON outputs.')
    parser.add_argument('--resume_folder', type=str, required=True,
                        help='Folder containing downloaded PDF resumes')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to the output CSV file (e.g., output.csv)')
    args = parser.parse_args()

    resume_folder = args.resume_folder
    output_csv = args.output_csv

    # Check if output CSV exists to determine if headers need to be written
    file_exists = os.path.isfile(output_csv)

    # Open the CSV file in append mode
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Filename', 'Question', 'Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Iterate through each PDF file in the resume folder
        for filename in os.listdir(resume_folder):
                file_path = os.path.join(resume_folder, filename)
                print(f"Processing: {filename}")


                # Parse the resume text to JSON
                resume_text, parsed_json = parse_resume_with_ollama(f"dataset/resumes/{filename}")

                    # Convert the Pydantic model to JSON string with indentation for readability
                try:
                    answer_json = parsed_json.model_dump_json(indent=2)
                except Exception as e:
                    print(f"Error converting parsed resume to JSON for {filename}: {e}")
                    answer_json = "{}"

                # Append to CSV
                writer.writerow({
                    'Filename': filename,
                    'Question': resume_text,
                    'Answer': answer_json
                })
                print(f"Successfully parsed: {filename}")

    print(f"\nAll resumes processed. Output saved to '{output_csv}'.")

# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    main()