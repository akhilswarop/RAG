import streamlit as st
import torch  # Make sure to import torch
import spacy
import pandas as pd
from typing import List
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import subprocess
# Added Imports for pyresparser
import sys
sys.path.append('C:/Users/swaro/OneDrive/Documents/GitHub/RAG/pyresparser')  # Adjust the path as needed
sys.path.append('C:/Users/swaro/OneDrive/Documents/GitHub/RAG/linkedin_scraper')  # Adjust the path as needed
from pyresparser import ResumeParser
from linkedin_scraper import actions
from linkedin_scraper import JobSearch

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Set page configuration
st.set_page_config(page_title="Career Guidance System using RAG", layout="wide")
st.title("Career Guidance System using RAG")

from pyresparser import ResumeParser
from linkedin_scraper import actions, JobSearch

# Initialize variables with default values
skills = []
academic_history = ""
psychometric_profile = ""

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

# Function to load or compute job title embeddings
@st.cache_resource
def load_or_compute_embeddings(job_titles, embeddings_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(embeddings_file):
        with st.spinner("Computing job title embeddings..."):
            onet_title_embeddings = model.encode(job_titles, convert_to_tensor=True)
            onet_title_embeddings = onet_title_embeddings.to(device)
            torch.save(onet_title_embeddings, embeddings_file)
    else:
        onet_title_embeddings = torch.load(embeddings_file, map_location=device)

    return onet_title_embeddings

# Function to rank job titles based on semantic similarity
def rank_job_titles_semantic(skills, job_titles, job_title_embeddings, top_k=5):
    if not skills or job_title_embeddings is None:
        return []

    skills_text = ' '.join(skills)
    skills_embedding = model.encode(skills_text, convert_to_tensor=True).cpu().numpy()
    job_title_embeddings_cpu = job_title_embeddings.cpu().numpy()
    similarities = cosine_similarity([skills_embedding], job_title_embeddings_cpu)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_job_titles = [job_titles[i] for i in top_indices]

    return top_job_titles

# Function to process the resume using pyresparser
def process_resume(file_path):
    try:
        data = ResumeParser(file_path).get_extracted_data()
        return data
    except Exception as e:
        st.error(f"An error occurred while parsing the resume: {e}")
        return {}

# Function to extract skills or keywords from text
def extract_skills(text):
    doc = nlp(text)
    skills_extracted = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'SKILL', 'TECHNOLOGY']]

    if not skills_extracted:
        words = word_tokenize(text.lower())
        skills_extracted = [word for word in words if word not in stopwords.words('english') and len(word) > 2]

    return skills_extracted

# Function to calculate skill match
def calculate_skill_match(user_skills, job_titles, onet_titles_df):
    job_skills = {}
    for job in job_titles:
        job_desc = onet_titles_df[onet_titles_df['Title'] == job]['Description'].values
        if len(job_desc) > 0:
            job_text = job_desc[0]
            job_tokens = extract_skills(job_text)
            job_skills[job] = set(job_tokens)
        else:
            job_skills[job] = set()

    match_scores = {}
    for job, required_skills in job_skills.items():
        if required_skills:
            match = set(user_skills).intersection(required_skills)
            score = len(match) / len(required_skills) * 100  # Percentage match
            match_scores[job] = round(score, 2)
        else:
            match_scores[job] = 0.0

    return match_scores, job_skills

# Function to visualize skill match
def visualize_skill_match(match_scores):
    if not match_scores:
        st.write("No match scores to display.")
        return

    df = pd.DataFrame({
        'Job Title': list(match_scores.keys()),
        'Match Score (%)': list(match_scores.values())
    })

    fig = px.bar(df, x='Job Title', y='Match Score (%)',
                 title='Skill Match Percentage for Recommended Jobs',
                 labels={'Job Title': 'Job Title', 'Match Score (%)': 'Match Score (%)'},
                 text='Match Score (%)')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

# Function to identify and visualize skill gaps
def identify_skill_gaps(user_skills, job_skills):
    all_required_skills = set()
    for required in job_skills.values():
        all_required_skills.update(required)
    missing_skills = all_required_skills - set(user_skills)
    return missing_skills

def visualize_skill_gaps(missing_skills):
    if missing_skills:
        st.subheader("Skill Gap Analysis")
        st.write("You may consider developing the following skills to enhance your career prospects:")
        for skill in sorted(missing_skills):
            st.write(f"- {skill}")
    else:
        st.subheader("Great Job!")
        st.write("You possess all the skills required for the recommended job titles.")

# Function to generate career guidance using an AI model
def generate_career_guidance(skills, academic_history, psychometric_profile, top_job_titles, job_listings):
    skills_text = ', '.join(skills) if skills else "No skills provided"
    top_job_titles_text = ', '.join(top_job_titles) if top_job_titles else "No job titles provided"
    job_listings_text = job_listings if job_listings else "No job listings available"

    prompt = f"""
    Based on the following information, provide personalized career guidance:

    **Skills:** {skills_text}

    **Academic History:** {academic_history}

    **Psychometric Profile:** {psychometric_profile}

    **Top Recommended Job Titles:** {top_job_titles_text}

    **Available Job Listings:**
    {job_listings_text}

    Provide a comprehensive analysis including:
    - Recommended career paths
    - Skill development suggestions
    - Potential industries to explore
    - Next steps for job applications
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        guidance = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        guidance = f"An error occurred while generating guidance: {e}"

    return guidance

# Function to perform LDA on O*NET data
def perform_lda_on_onet(onet_titles_df):
    if 'Description' not in onet_titles_df.columns:
        st.error("The O*NET dataset does not contain a 'Description' column.")
        return None
    if onet_titles_df.empty:
        st.error("O*NET dataset is empty. Please check the data file.")
        return None

    texts = onet_titles_df['Description'].fillna('').apply(preprocess_text).tolist()
    if not texts:
        st.error("No text data available for LDA.")
        return None

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    if not corpus:
        st.error("Corpus is empty after preprocessing. Cannot perform LDA.")
        return None

    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None

    temp = dictionary[0]
    id2word = dictionary.id2token

    try:
        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )
        topics = lda_model.print_topics(num_words=10)
        return topics
    except Exception as e:
        st.error(f"An error occurred while performing LDA: {e}")
        return None

# Function to preprocess text for LDA
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    tokens = [token for token in tokens if len(token) > 2]
    return tokens

# Function to display job listings in Streamlit
def display_jobs(jobs_data):
    if not jobs_data:
        st.write("No job listings found.")
        return

    st.subheader("Job Listings Based on Your Profile:")
    for job in jobs_data:
        st.markdown(f"### {job['Job Title']}")
        st.write(f"**Company:** {job['Company']}")
        st.write(f"**Location:** {job['Location']}")
        st.write(f"**Sector:** {job['Job Sector']}")
        st.markdown(f"[Apply Here]({job['Job URL']})")
        st.markdown("---")

# Function to perform LinkedIn job search (credentials should be handled securely)
def perform_linkedin_job_search(job_titles: List[str]):
    st.warning("LinkedIn scraping may violate their Terms of Service. Proceed at your own risk.")
    email = "akhiltheswarop@gmail.com"
    password = "Vaazhkai@12"

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    actions.login(driver, email, password)
    st.info("Please complete any additional verification in the browser window.")
    st.write("Waiting for user to complete login...")

    # Wait for the user to complete login (you may implement a better method)
    st.warning("Press Enter in the console to continue after successful login.")
    input("Press Enter to continue after successful login")

    job_search = JobSearch(driver=driver, close_on_complete=False, scrape=False)
    jobs_data = []

    for title in job_titles[:1]:
        job_listings = job_search.search(title)
        for job in job_listings:
            jobs_data.append({
                'Job Title': job.job_title,
                'Company': job.company,
                'Location': job.location,
                'Job URL': job.linkedin_url,
                'Job Sector': title
            })

    df = pd.DataFrame(jobs_data)
    df.to_csv('linkedin_job_listings.csv', mode='a', index=False)
    driver.quit()
    return jobs_data

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
        st.session_state.experience = resume_data.get("experience", [])
        st.session_state.college_name = resume_data.get("college_name", "")
        st.success("Resume processed successfully!")
    else:
        st.error("Failed to process the resume. Please try again.")

# Combined User Profile, Academic, and Location Form
with st.form("career_guidance_form"):
    # Academic History Autofill
    st.subheader("Academic History")
    name = st.text_input("Name", value=st.session_state.name)
    email = st.text_input("Email", value=st.session_state.email)
    mobile_number = st.text_input("Mobile Number", value=st.session_state.mobile_number)
    degree = st.text_area("Degree", value=st.session_state.degree)
    institution = st.text_input("Institution", value=st.session_state.college_name)
    graduation_year = st.number_input("Graduation Year", min_value=1950, max_value=2100, step=1)

    # Skills Autofill
    st.subheader("Skills")
    skills = st.text_area("Skills", value=', '.join(st.session_state.skills))

    # Experience Autofill
    st.subheader("Experience")
    experience = st.text_area("Experience", value=st.session_state.experience)

    # Psychometric Profile
    st.subheader("Psychometric Profile")
    personality_traits = st.text_input("Personality Traits (e.g., Analytical, Creative)")
    strengths = st.text_input("Strengths (e.g., Problem-solving, Leadership)")
    preferences = st.text_area("Work Preferences (e.g., Remote work, Team-oriented)")

    # Location Preference
    st.subheader("Job Location Preference")
    location = st.text_input("Preferred job location (e.g., New York, NY) or leave blank for any location:")

    # Submit Button
    submit = st.form_submit_button("Submit and Generate Guidance")

# Handle form submission
if submit:
    if not (degree and institution and graduation_year and personality_traits and strengths and preferences):
        st.error("Please fill in all the required profile fields.")
    else:
        # Process academic history
        academic_history = f"Degree: {degree}, Institution: {institution}, Graduation Year: {graduation_year}"

        # Process psychometric profile
        psychometric_profile = f"Personality Traits: {personality_traits}, Strengths: {strengths}, Preferences: {preferences}"

        st.success("Profile information submitted successfully!")

        # Ensure skills are in list format
        user_skills = skills.split(', ') if skills else []

        # Load or compute embeddings
        embeddings_file = 'onet_title_embeddings.pkl'
        onet_title_embeddings = load_or_compute_embeddings(onet_titles, embeddings_file)

        # Rank job titles using semantic similarity
        top_job_titles = rank_job_titles_semantic(user_skills, onet_titles, onet_title_embeddings, top_k=5)
        if top_job_titles:
            st.subheader("Top Job Titles Based on Your Profile:")
            st.write(", ".join(top_job_titles))
        else:
            st.warning("No job titles could be ranked based on the provided skills.")

        # Search Jobs and Generate Guidance
        # with st.spinner("Searching for jobs and generating career guidance..."):
        #     job_results = perform_linkedin_job_search(top_job_titles)

        #     # Display jobs
        #     if job_results:
        #         display_jobs(job_results)
        #     else:
        #         st.write("No job listings found.")

        #     # Generate career guidance
        #     job_listings_text = "\n".join([f"{job['Job Title']} at {job['Company']} in {job['Location']}" for job in job_results])
        #     guidance = generate_career_guidance(user_skills, academic_history, psychometric_profile, top_job_titles, job_listings_text)
        #     st.subheader("Personalized Career Guidance:")
        #     st.write(guidance)

        #     # Skill Match Visualization
        #     match_scores, job_skills = calculate_skill_match(user_skills, top_job_titles, onet_titles_df)
        #     visualize_skill_match(match_scores)

        #     # Skill Gap Analysis
        #     missing_skills = identify_skill_gaps(user_skills, job_skills)
        #     visualize_skill_gaps(missing_skills)

        # # Perform LDA unsupervised clustering and display the results
        # if not onet_titles_df.empty:
        #     st.subheader("LDA Unsupervised Clustering on O*NET Data")
        #     st.write("Performing LDA to find topics in O*NET occupations...")
        #     topics = perform_lda_on_onet(onet_titles_df)

        #     if topics:
        #         st.write("Here are the topics found:")
        #         for idx, topic in enumerate(topics):
        #             st.write(f"**Topic {idx+1}:** {topic}")
        #     else:
        #         st.write("LDA could not be performed due to insufficient data or errors.")
        # else:
        #     st.write("O*NET data is unavailable. Please check the data file.")
