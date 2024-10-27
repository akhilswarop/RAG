import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
import spacy
import pandas as pd
from typing import List
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai
import requests
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')  # Ensure this is 'punkt'

# Set OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

# Set page configuration
st.set_page_config(page_title="Career Guidance System using RAG", layout="wide")

st.title("Career Guidance System using RAG")

# Load spaCy model for skill extraction
nlp = spacy.load('en_core_web_sm')

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a lightweight model for efficiency

# Load O*NET job titles and descriptions for ranking purposes
onet_titles_df = pd.read_csv('Occupation Data.csv')  # Ensure this file contains 'Title' and 'Description' columns
onet_titles = onet_titles_df['Title'].tolist()

# Precompute or load job title embeddings
embeddings_file = 'onet_title_embeddings.pkl'

if not os.path.exists(embeddings_file):
    with st.spinner("Computing job title embeddings..."):
        onet_title_embeddings = model.encode(onet_titles, convert_to_tensor=True)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(onet_title_embeddings, f)
else:
    with open(embeddings_file, 'rb') as f:
        onet_title_embeddings = pickle.load(f)
# Function to rank job titles based on semantic similarity to provided skills
def rank_job_titles_semantic(skills, job_titles, job_title_embeddings, top_k=5):
    # Convert the list of skills into a single string
    skills_text = ' '.join(skills)
    
    # Generate embeddings for the skills using the SentenceTransformer model and move to CPU
    skills_embedding = model.encode(skills_text, convert_to_tensor=True).cpu().numpy()
    
    # Ensure job title embeddings are on the CPU and convert to NumPy array
    job_title_embeddings_cpu = job_title_embeddings.cpu().numpy()
    
    # Calculate cosine similarity between the skills embedding and each job title embedding
    similarities = cosine_similarity([skills_embedding], job_title_embeddings_cpu)[0]
    
    # Get the top K job titles based on similarity scores
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_job_titles = [job_titles[i] for i in top_indices]
    
    return top_job_titles
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

# Function to perform job search using Arbeitnow API
def search_jobs_arbeitnow(keywords: List[str], location: str = ""):
    jobs_data = []
    for keyword in keywords:
        params = {
            'description': keyword,
            'location': location,
        }
        response = requests.get("https://arbeitnow.com/api/job-board-api", params=params)
        if response.status_code == 200:
            data = response.json()
            for job in data.get('data', []):
                jobs_data.append({
                    'Job Title': job.get('title'),
                    'Company': job.get('company_name'),
                    'Location': job.get('location'),
                    'Job URL': job.get('url'),
                    'Job Sector': keyword
                })
        else:
            st.error(f"Failed to fetch jobs for {keyword}.")
    return jobs_data

# Function to display job listings in Streamlit
def display_jobs(jobs_data):
    st.subheader("Job Listings Based on Your Profile:")
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

# Function to calculate skill match
def calculate_skill_match(skills, job_titles, onet_titles_df):
    # For each job title, find required skills from O*NET data
    job_skills = {}
    for job in job_titles:
        job_desc = onet_titles_df[onet_titles_df['Title'] == job]['Description'].values
        if len(job_desc) > 0:
            job_text = job_desc[0]
            job_tokens = extract_skills(job_text)
            job_skills[job] = set(job_tokens)
        else:
            job_skills[job] = set()
    
    # Calculate match score
    match_scores = {}
    for job, required_skills in job_skills.items():
        if required_skills:
            match = set(skills).intersection(required_skills)
            score = len(match) / len(required_skills) * 100  # Percentage match
            match_scores[job] = score
        else:
            match_scores[job] = 0
    
    return match_scores, job_skills

# Function to visualize skill match
def visualize_skill_match(match_scores):
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
def identify_skill_gaps(skills, job_skills):
    all_required_skills = set()
    for required in job_skills.values():
        all_required_skills.update(required)
    missing_skills = all_required_skills - set(skills)
    return missing_skills

def visualize_skill_gaps(missing_skills):
    if missing_skills:
        st.subheader("Skill Gap Analysis")
        st.write("You may consider developing the following skills to enhance your career prospects:")
        for skill in missing_skills:
            st.write(f"- {skill}")
    else:
        st.subheader("Great Job!")
        st.write("You possess all the skills required for the recommended job titles.")

# Function to generate personalized career guidance using OpenAI
def generate_career_guidance(skills, academic_history, psychometric_profile, top_job_titles, job_listings):
    prompt = f"""
    Based on the following information, provide personalized career guidance:

    **Skills:** {', '.join(skills)}
    
    **Academic History:** {academic_history}
    
    **Psychometric Profile:** {psychometric_profile}
    
    **Top Recommended Job Titles:** {', '.join(top_job_titles)}
    
    **Available Job Listings:**
    {job_listings}

    Provide a comprehensive analysis including:
    - Recommended career paths
    - Skill development suggestions
    - Potential industries to explore
    - Next steps for job applications
    """

    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose appropriate engine
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    guidance = response.choices[0].text.strip()
    return guidance



# User Profile Input Form
with st.form("user_profile"):
    st.header("Your Profile Information")
    
    # Academic History
    st.subheader("Academic History")
    degree = st.text_input("Degree (e.g., B.Sc. in Computer Science)")
    institution = st.text_input("Institution (e.g., MIT)")
    graduation_year = st.number_input("Graduation Year", min_value=1950, max_value=2100, step=1)
    certifications = st.text_area("Certifications (separated by commas):")
    
    # Psychometric Profile
    st.subheader("Psychometric Profile")
    personality_traits = st.text_input("Personality Traits (e.g., Analytical, Creative)")
    strengths = st.text_input("Strengths (e.g., Problem-solving, Leadership)")
    preferences = st.text_area("Work Preferences (e.g., Remote work, Team-oriented)")
    
    submitted = st.form_submit_button("Submit Profile")

if submitted:
    # Process academic history
    academic_history = f"Degree: {degree}, Institution: {institution}, Graduation Year: {graduation_year}, Certifications: {certifications}"
    
    # Process psychometric profile
    psychometric_profile = f"Personality Traits: {personality_traits}, Strengths: {strengths}, Work Preferences: {preferences}"
    
    st.success("Profile information submitted successfully!")

# Process the uploaded resume
uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)
    if resume_text:
        # Extract skills
        skills = extract_skills(resume_text)
        st.subheader("Extracted Skills and Interests from Resume:")
        st.write(skills)

        # Rank job titles using semantic similarity
        top_job_titles = rank_job_titles_semantic(skills, onet_titles, onet_title_embeddings, top_k=5)
        st.subheader("Top 5 Job Titles Based on Your Resume:")
        st.write(top_job_titles)

        # Collect user location preference
        location = st.text_input("Enter preferred job location (e.g., New York, NY) or leave blank for any location:")

        # Search Jobs and Generate Guidance Button
        if st.button("Search Jobs and Generate Guidance"):
            if skills and academic_history and psychometric_profile:
                with st.spinner("Searching for jobs and generating career guidance..."):
                    job_results = search_jobs_arbeitnow(top_job_titles, location)
                    
                    # Display jobs
                    if job_results:
                        display_jobs(job_results)
                    else:
                        st.write("No job listings found.")

                    # Generate career guidance
                    job_listings_text = "\n".join([f"{job['Job Title']} at {job['Company']} in {job['Location']}" for job in job_results])
                    guidance = generate_career_guidance(skills, academic_history, psychometric_profile, top_job_titles, job_listings_text)
                    st.subheader("Personalized Career Guidance:")
                    st.write(guidance)
                    
                    # Skill Match Visualization
                    match_scores, job_skills = calculate_skill_match(skills, top_job_titles, onet_titles_df)
                    visualize_skill_match(match_scores)
                    
                    # Skill Gap Analysis
                    missing_skills = identify_skill_gaps(skills, job_skills)
                    visualize_skill_gaps(missing_skills)
            else:
                st.error("Please provide your skills, academic history, and psychometric profile.")

# Perform LDA unsupervised clustering and display the results
st.subheader("LDA Unsupervised Clustering on O*NET Data")
st.write("Performing LDA to find topics in O*NET occupations...")
topics = perform_lda_on_onet(onet_titles_df)

if topics:
    st.write("Here are the topics found:")
    for idx, topic in enumerate(topics):
        st.write(f"**Topic {idx+1}:** {topic}")
