import streamlit as st
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


from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
# LLM Evaluation Logic 

class Gemma2_2B(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # Same as the previous example above
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=10000,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        # Output and load valid JSON
        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)
    
    def get_model_name(self):
        return "Gemma-2 2B"


def initialize_evaluator():
    # Configure bitsandbytes quantization
    quantization_config = QuantoConfig(weights="int8")

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        device_map="cuda",            # automatically distribute layers across GPUs / CPU
        quantization_config=quantization_config,
        torch_dtype=torch.float16,     # float16 for weights outside the 4-bit layers
        trust_remote_code=True,        # required if the Gemma-2 code is custom
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        trust_remote_code=True,
    )

    gemma2_2b = Gemma2_2B(model=model, tokenizer=tokenizer)
    return gemma2_2b

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Set page configuration
st.set_page_config(page_title="Career Guidance System using RAG with Mistral", layout="wide")
st.title("Career Guidance System using RAG with Mistral")

# Initialize variables with default values
skills = []
academic_history = ""
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
        


    

# Function to generate career guidance using RAG with Mistral
def generate_career_guidance(skills, academic_history):
    
    # Generations and Evaluations 
    generations = {}
    evaluations = {}
    
    for model in ["gemma2_2b", "gemma2_9b", "mistral"]:
        if model not in evaluations:
            evaluations[model] = {
                "bleu": None,
                "rouge": None, 
                "bert": None,
                "sentence_movers_similarity": None,
                "bleurt": None,
                "answer_relevancy": {
                    "score": None,
                    "reason": None
                }, 
                "faithfulness": {
                    "score": None,
                    "reason": None
                },
                
                "hallucination": {
                    "score": None,
                    "reason": None
                }, 
                
                
                
            }
            
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
        with st.spinner("Generating personalized career guidance..."):

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
            


        models = ["gemma2_2b", "gemma2_9b", "mistral"]
        reference_model = "deepseek-r1:14b"

        # Compute BLEU, ROUGE, and BERTScore for each model
        for model in models:
            evaluations[model]["bleu"] = calculate_bleu_score(generations[reference_model], generations[model])
            evaluations[model]["rouge"] = calculate_rouge_score(generations[reference_model], generations[model])
            evaluations[model]["bert"] = calculate_bertscore(generations[reference_model], generations[model])
            evaluations[model]["sentence_movers_similarity"] = sentence_movers_similarity(generations[reference_model], generations[model])
            evaluations[model]["bleurt"] = compute_bleurt(generations[reference_model], generations[model])

            # Print evaluation scores
            print(f"{model.upper()} - BLEU: {evaluations[model]['bleu']}")
            print(f"{model.upper()} - ROUGE: {evaluations[model]['rouge']}")
            print(f"{model.upper()} - BERT: {evaluations[model]['bert']}")

        # Evaluate LLM outputs for each model
        for model in models:
            (
                evaluations[model]["answer_relevancy"]["score"],
                evaluations[model]["answer_relevancy"]["reason"],
                evaluations[model]["faithfulness"]["score"],
                evaluations[model]["faithfulness"]["reason"],
                evaluations[model]["hallucination"]["score"],
                evaluations[model]["hallucination"]["reason"]
            ) = evaluate_llm(prompt, context, generations[model]) 
            
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred while generating guidance: {e.stderr}")


    return  generations, evaluations, top_job_titles

def google_jobs_search(job_title, location):

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


##########################################################################
# Evaluation Functions
##########################################################################
def calculate_bleu_score(reference, candidate):
    reference = [nltk.word_tokenize(reference.lower())]
    candidate = nltk.word_tokenize(candidate.lower())
    
    # Use smoothing to handle brevity
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU score
    score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
    print(f"BLEU SCORE: {round(score, 2)}")
    return round(score, 2)

def calculate_rouge_score(reference, candidate):
    """
    Compute ROUGE-L score between reference and candidate text.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    rouge_score = scores["rougeL"].fmeasure
    
    # Announce and print the ROUGE score
    print(f"ROUGE SCORE: {round(rouge_score, 2)}")
    
    return round(rouge_score, 2)

def calculate_bertscore(reference, candidate):
    """
    Compute BERTScore (F1) between reference and candidate text.
    """
    P, R, F1 = score_bert([candidate], [reference], lang="en")
    precision = P.item()
    recall = R.item()
    f1 = F1.item()
    
    # Announce and print the BERTScore components
    print(f"BERT SCORE - Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}")
    
    return round(f1, 2)
def sentence_movers_similarity(reference, hypothesis):
    """
    Compute Sentence Mover’s Similarity (SMS) using contextual embeddings.
    """
    
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    emb1 = sentence_model.encode(reference, convert_to_tensor=True)
    emb2 = sentence_model.encode(hypothesis, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    
    # Announce and print the Sentence Mover’s Similarity
    print(f"SMS SCORE: {round(similarity, 2)}")
    
    return round(similarity, 2)


def compute_bleurt(candidate, reference):
    # Initialize the BLEURT scorer (this may download the default model if not cached)
    scorer = score_bleurt.BleurtScorer()
    
    # BLEURT expects lists of candidates and references
    scores = scorer.score(
        references=[reference], 
        candidates=[candidate]
    )
    print(f"BLEURT SCORE: {np.mean(scores)}")
    return np.mean(scores)  # BLEURT returns a list of scores, one per candidate

def evaluate_llm(input, context, output):
    gemma2_2b = initialize_evaluator()
    
    # Test Cases
    
    answer_relevancy_test_case = LLMTestCase(
    input=input,
    actual_output=output
)
    faithfulness_test_case = LLMTestCase(
    input=input,
    actual_output=output,
    retrieval_context=list(context)
)
    hallucination_test_case = LLMTestCase(
    input=input,
    actual_output=output,
    context=list(context)
)

    
    
    # Metrics
    
    answer_relevancy_metric = AnswerRelevancyMetric(model=gemma2_2b, threshold=0.5, include_reason=True)

    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=gemma2_2b, include_reason=True)
    
    hallucination_metric = HallucinationMetric(threshold= 0.7, model=gemma2_2b, include_reason= True)
    
 
    hallucination_metric.measure(hallucination_test_case)
    answer_relevancy_metric.measure(answer_relevancy_test_case)
    faithfulness_metric.measure(faithfulness_test_case)
    
    
    return answer_relevancy_metric.score, answer_relevancy_metric.reason, faithfulness_metric.score, faithfulness_metric.reason, hallucination_metric.score, hallucination_metric.reason

def play_beeps():
    # Play a simple beep sound
    frequency = 2500  # Set frequency to 2500 Hertz
    duration = 1000  # Set duration to 1000 milliseconds (1 second)

    while True:
        winsound.Beep(frequency, duration)
        
            
# Main application logic

# File Upload and Automatic Resume Processing
st.header("Complete Your Profile and Generate Guidance")
st.subheader("Upload Your Resume")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"], key="resume_upload")

# Initialize session state variables
if "resume_data" not in st.session_state:
    st.session_state.resume_data = {}
if "resume_file" not in st.session_state:
    st.session_state.resume_file = ""
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
    # Check if the uploaded file is different from the previously processed one
    if st.session_state.resume_file != uploaded_file.name:
        st.success("Resume uploaded successfully!")
        
                # Save the file temporarily for processing
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract resume data
        resume_data = parse_resume_with_ollama(uploaded_file.name)

        if resume_data:
            st.session_state.resume_data = resume_data
            st.session_state.resume_file = uploaded_file.name  
            st.session_state.name = resume_data.name if resume_data.name else "Not specified"
            st.session_state.email = resume_data.email if resume_data.email else "Not specified"
            st.session_state.mobile_number = resume_data.phone if resume_data.phone else "Not specified"
            st.session_state.skills = resume_data.skills if resume_data.skills else []
            st.session_state.degree = resume_data.education[0].degree if resume_data.education else "Not specified"
            st.session_state.college_name = resume_data.education[0].institution if resume_data.education else "Not specified"   
            st.session_state.experience = [f"Role:{exp.role} Company: {exp.company} Description: {exp.description}" for exp in resume_data.experience] if resume_data.experience else "No experience listed"        
            st.success("Resume processed successfully!")
        else:
            st.error("Failed to process the resume. Please try again.")
    else:
        # Optional: Inform the user that the resume has already been processed
        st.info("Resume already processed.")

# Combined User Profile and Location Form
with st.form("career_guidance_form"):
    # Academic History Autofill
    st.subheader("Academic History")
    name = st.text_input("Name", value=st.session_state.name)
    email = st.text_input("Email", value=st.session_state.email)
    mobile_number = st.text_input("Mobile Number", value=st.session_state.mobile_number)
    degree = st.text_area("Degree", st.session_state.degree)
    institution = st.text_input("Institution", value=st.session_state.college_name)

    # Skills Autofill
    st.subheader("Skills")
    skills = st.text_area("Skills", value=', '.join(st.session_state.skills) if isinstance(st.session_state.skills, list) else st.session_state.skills)

    # Experience Autofill
    st.subheader("Experience")
    experience = st.text_area("Experience", value=st.session_state.experience)

    location = st.text_input("Location", value="Remote")

    # Submit Button
    submit = st.form_submit_button("Submit and Generate Guidance")


# Handle form submission
if submit:
    if not (degree and institution and location):
        st.error("Please fill in all the required profile fields.")
    else:
        # Process academic history
        academic_history = f"Degree: {degree}, Institution: {institution}"

        # Process psychometric profile

        st.success("Profile information submitted successfully!")

        # Ensure skills are in list format
        user_skills = [skill.strip() for skill in skills.split(',')] if skills else []

        start_time = time.time()

        # Generate career guidance using RAG with Mistral
        generations, evaluations, top_job_titles = generate_career_guidance(
            skills=user_skills,
            academic_history=academic_history,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

# Format elapsed time in minutes and seconds
        minutes, seconds = divmod(elapsed_time, 60)
# Scoring 


        
        st.subheader("Career Guidance [Gemma 2B]:")
        st.write(generations["gemma2_2b"])
        st.markdown("---")
        
        data = {
                "Metric": ["Answer Relevancy", "Faithfulness", "Hallucination", "BLEU", "ROUGE", "BERT"],
                "Score": [
                    evaluations["gemma2_2b"]["answer_relevancy"]["score"],
                    evaluations["gemma2_2b"]["faithfulness"]["score"],
                    evaluations["gemma2_2b"]["hallucination"]["score"],
                    evaluations["gemma2_2b"]["bleu"],   # BLEU has a score
                    evaluations["gemma2_2b"]["rouge"],  # ROUGE has a score
                    evaluations["gemma2_2b"]["bert"],    # BERT has a score
                    evaluations["gemma2_2b"]["sentence_movers_similarity"],
                    evaluations["gemma2_2b"]["bleurt"]# BERT has a score
                ],
                "Reason": [
                    evaluations["gemma2_2b"]["answer_relevancy"]["reason"],
                    evaluations["gemma2_2b"]["faithfulness"]["reason"],
                    evaluations["gemma2_2b"]["hallucination"]["reason"],
                    None,  # BLEU does not have a reason field
                    None,  # ROUGE does not have a reason field
                    None   # BERT does not have a reason field
                ]
            }


        df = pd.DataFrame(data)

        # Display the table
        st.markdown("---")
        st.table(df)
    
        
        st.subheader("Career Guidance [Gemma 9B]:")
        st.write(generations["gemma2_9b"])
        st.markdown("---")
        data = {
                "Metric": ["Answer Relevancy", "Faithfulness", "Hallucination", "BLEU", "ROUGE", "BERT"],
                "Score": [
                    evaluations["gemma2_9b"]["answer_relevancy"]["score"],
                    evaluations["gemma2_9b"]["faithfulness"]["score"],
                    evaluations["gemma2_9b"]["hallucination"]["score"],
                    evaluations["gemma2_9b"]["bleu"],   # BLEU has a score
                    evaluations["gemma2_9b"]["rouge"],  # ROUGE has a score
                    evaluations["gemma2_9b"]["bert"],
                    evaluations["gemma2_9b"]["sentence_movers_similarity"],# BERT has a score
                    evaluations["gemma2_9b"]["bleurt"]# BERT has a score
                ],
                "Reason": [
                    evaluations["gemma2_9b"]["answer_relevancy"]["reason"],
                    evaluations["gemma2_9b"]["faithfulness"]["reason"],
                    evaluations["gemma2_9b"]["hallucination"]["reason"],
                    None,  # BLEU does not have a reason field
                    None,  # ROUGE does not have a reason field
                    None   # BERT does not have a reason field
                ]
            }

        df = pd.DataFrame(data)

        # Display the table
        st.markdown("---")
        st.table(df)
    
        
        st.subheader("Career Guidance [Mistral]:")
        st.write(generations["mistral"])
        st.markdown("---")
        data = {
                "Metric": ["Answer Relevancy", "Faithfulness", "Hallucination", "BLEU", "ROUGE", "BERT"],
                "Score": [
                    evaluations["mistral"]["answer_relevancy"]["score"],
                    evaluations["mistral"]["faithfulness"]["score"],
                    evaluations["mistral"]["hallucination"]["score"],
                    evaluations["mistral"]["bleu"],   # BLEU has a score
                    evaluations["mistral"]["rouge"],  # ROUGE has a score
                    evaluations["mistral"]["bert"],
                    evaluations["mistral"]["sentence_movers_similarity"],
                    evaluations["mistral"]["bleurt"]
                ],
                "Reason": [
                    evaluations["mistral"]["answer_relevancy"]["reason"],
                    evaluations["mistral"]["faithfulness"]["reason"],
                    evaluations["mistral"]["hallucination"]["reason"],
                    None,  # BLEU does not have a reason field
                    None,  # ROUGE does not have a reason field
                    None   # BERT does not have a reason field
                ]
            }

        df = pd.DataFrame(data)

        # Display the table
        st.markdown("---")
        st.table(df)
    
        # Play beeps
        play_beeps()
        
        st.write(f"Time taken for generation: {int(minutes)} minutes and {seconds:.2f} seconds")

        st.write("Top Job Titles:")
        relevant_job_titles = [job["title"] for job in top_job_titles]
        for title in relevant_job_titles:
            st.write(title)
        
        st.title("Job Listings")
        # jobs_data = google_jobs_search(relevant_job_titles[0], location)
        for job in jobs_data:
            st.subheader(job["Title"])
            st.write(f"**Company:** {job['Company']}")
            st.write(f"**Location:** {job['Location']}")
            st.write(f"**Via:** {job['Via']}")
            st.write(job["Description (truncated)"])
            
            # Render each apply option as a clickable link
            if job["Apply Options"]:
                st.write("**Apply Links:**")
                for option in job["Apply Options"]:
                    title = option.get("title", "Apply Here")
                    link = option.get("link", "#")
                    st.write(f"[{title}]({link})")
            
            st.markdown("---")  # A line to separate jobs