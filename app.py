import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
import spacy
import openai
import pandas as pd
from selenium import webdriver
from linkedin_scraper import JobSearch, actions
from typing import List
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Additional imports for LDA
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')  # Ensure this is 'punkt'

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

# Function for preprocessing text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Ensure 'word_tokenize' is used correctly
    tokens = [token for token in tokens if token.isalpha()]  # Remove punctuation and numbers
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stop words
    tokens = [token for token in tokens if len(token) > 2]  # Remove short words
    return tokens

# Load O*NET job titles and descriptions for ranking purposes
onet_titles_df = pd.read_csv('Occupation Data.csv')  # Ensure this file contains 'Title' and 'Description' columns
onet_titles = onet_titles_df['Title'].tolist()

# Function to rank job titles based on skills
def rank_job_titles(skills, onet_titles):
    skills_text = ' '.join(skills)
    documents = [skills_text] + onet_titles
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:])[0]
    top_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 job titles
    top_job_titles = [onet_titles[i] for i in top_indices]
    return top_job_titles

# Function to perform LinkedIn job search using Selenium
def perform_linkedin_job_search(job_titles: List[str], email, password):
    driver = webdriver.Chrome()
    actions.login(driver, email, password) 

    # Wait for manual login if two-factor authentication is enabled
    st.info("Please complete the login process in the browser window that has opened.")
    input("Press Enter to continue after successful login")

    job_search = JobSearch(driver=driver, close_on_complete=False, scrape=False)

    jobs_data = []

    for title in job_titles:
        job_listings = job_search.search(title)  

        for job in job_listings:
            jobs_data.append({
                'Job Title': job.job_title,
                'Company': job.company,
                'Location': job.location,
                'Job URL': job.linkedin_url,
                'Job Sector': title  # The job title that was used for the search
            })

    df = pd.DataFrame(jobs_data)
    df.to_csv('linkedin_job_listings.csv', mode='a', index=False)
    return jobs_data

# Function to display job listings in Streamlit
def display_jobs(jobs_data):
    st.subheader("LinkedIn Job Listings Based on Your Resume:")
    for job in jobs_data:
        st.markdown(f"### {job['Job Title']}")
        st.write(f"**Company:** {job['Company']}")
        st.write(f"**Location:** {job['Location']}")
        st.write(f"**Sector:** {job['Job Sector']}")
        st.markdown(f"[Apply Here]({job['Job URL']})")
        st.markdown("---")

# Function to perform LDA on O*NET data
def perform_lda_on_onet(onet_titles_df):
    # Check if 'Description' column exists
    if 'Description' not in onet_titles_df.columns:
        st.error("The O*NET dataset does not contain a 'Description' column.")
        return None
    # Preprocess the descriptions
    texts = onet_titles_df['Description'].fillna('').apply(preprocess_text).tolist()
    
    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(texts)
    
    # Filter out extremes to limit the number of features
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # Create a Bag-of-Words representation of the documents.
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Set training parameters.
    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    
    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    
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

# Perform LDA unsupervised clustering and display the results
st.subheader("LDA Unsupervised Clustering on O*NET Data")
st.write("Performing LDA to find topics in O*NET occupations...")
topics = perform_lda_on_onet(onet_titles_df)

if topics:
    st.write("Here are the topics found:")
    for idx, topic in enumerate(topics):
        st.write(f"**Topic {idx+1}:** {topic}")
