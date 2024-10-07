import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
import spacy
import pandas as pd
from selenium import webdriver
from linkedin_scraper import JobSearch, actions
from typing import List
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Set page configuration
st.set_page_config(page_title="Career Guidance System using RAG", layout="wide")

st.title("Career Guidance System using RAG")

# Load spaCy model for skill extraction
nlp = spacy.load('en_core_web_sm')

# Function to extract text from resume
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)
    else:
        text = ""
    return text

# Function to extract skills or keywords from the text
def extract_skills(text):
    doc = nlp(text)
    skills = set()
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN']:
            skills.add(token.lemma_.lower())
    return list(skills)

# Load O*NET job titles for ranking purposes
onet_titles_df = pd.read_csv('Occupation Data.csv')  # Ensure this file is in your working directory
onet_titles = onet_titles_df['Title'].tolist()

# Function to rank job titles based on skills
def rank_job_titles(skills, onet_titles):
    skills_text = ' '.join(skills)
    documents = [skills_text] + onet_titles
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:])[0]
    top_indices = cosine_similarities.argsort()[-2:][::-1]
    top_job_titles = [onet_titles[i] for i in top_indices]
    return top_job_titles

# Function to perform LinkedIn job search using Selenium
def perform_linkedin_job_search(job_titles: List[str], email, password):
    driver = webdriver.Chrome()
    email = "akhiltheswarop@gmail.com"
    password = "Vaazhkai@12"

    actions.login(driver, email, password) 

    input("Press Enter to continue after successful login")

    job_search = JobSearch(driver=driver, close_on_complete=False, scrape=False)

    jobs_titles = job_titles

    for title in jobs_titles:

        job_listings = job_search.search(title)  

        jobs_data = []

    for job in job_listings:

        jobs_data.append({
            'Job Title': job.job_title,
            'Company': job.company,
            'Location': job.location,
            'Job URL': job.linkedin_url,
            'Job Sector': title  # The job title that was used for the search
        })

        df = pd.DataFrame(jobs_data)

        df.to_csv('linkedin_job_listings.csv', mode ='a', index=False)
    return jobs_data

# Function to display job listings in Streamlit
def display_jobs(jobs_data):
    st.subheader("LinkedIn Job Listings Based on Your Resume:")
    for job in jobs_data:
        st.markdown(f"### {job['Job Title']}")
        st.write(f"**Company:** {job['Company']}")
        st.write(f"**Location:** {job['Location']}")
        st.write(f"**Sector:** {job['Job Sector']}")
        # Display Apply button that links to the job URL
        st.link_button("Apply", job['Job URL'])
        st.markdown("---")

# Process the uploaded resume
uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)
    if resume_text:
        # Extract skills
        skills = extract_skills(resume_text)
        st.subheader("Extracted Skills and Interests from Resume:")
        st.write(skills)

        # Rank job titles
        top_job_titles = rank_job_titles(skills, onet_titles)
        st.subheader("Top 5 Job Titles Based on Your Resume:")
        st.write(top_job_titles)

        # Perform job search using LinkedIn
        st.subheader("Searching for Jobs on LinkedIn...")
        # Collect user email and password for LinkedIn login
        email = st.text_input("Enter your LinkedIn email:", type="default")
        password = st.text_input("Enter your LinkedIn password:", type="password")

        if st.button("Search LinkedIn Jobs"):
            if email and password:
                job_results = perform_linkedin_job_search(top_job_titles, email, password)

                # Display jobs
                if job_results:
                    display_jobs(job_results)
                else:
                    st.write("No job listings found.")
            else:
                st.error("Please enter both LinkedIn email and password to proceed.")
    else:
        st.error("Could not extract text from the uploaded file.")
