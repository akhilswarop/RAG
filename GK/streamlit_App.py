import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Job Matcher Chatbot", layout="wide")

st.markdown("""
## Job Matcher: Get Career Insights

This chatbot uses a job title CSV file and a resume PDF to recommend job options based on skill extraction. Upload your resume in PDF format and a CSV file containing job titles and descriptions to receive customized job suggestions or ask broader questions based on the uploaded files.
""")

# Load resume text from PDF
def get_pdf_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def generate_llm_response(resume_text, user_question, job_data):
    jobs_text = "\n".join([f"{row['Job Title']}: {row['Description']}" for _, row in job_data.iterrows()])
    prompt = f"""
    Below is the resume text and available job descriptions. Please answer the question based on the information provided.
    
    Resume:
    {resume_text}

    Job Descriptions:
    {jobs_text}

    Question: {user_question}

    Answer:
    """

    model_name = "google/flan-t5-large"  # You can use "flan-t5-xl" or "flan-t5-xxl" for even larger models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize and generate a response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    outputs = model.generate(inputs.input_ids, max_new_tokens=200, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def get_csv_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if "Job Title" not in df.columns or "Description" not in df.columns:
        st.warning("CSV must contain 'Job Title' and 'Description' columns.")
        return None
    return df

def main():
    st.header("Job Matcher Chatbot 💁")

    col1, col2, col3 = st.columns([1, 2, 1])  # This centers the content on the screen
    with col2:
        uploaded_pdf = st.file_uploader("Upload your resume (PDF)", type=['pdf'], key="pdf_uploader")
        uploaded_csv = st.file_uploader("Upload job descriptions (CSV)", type=['csv'], key="csv_uploader")

        user_question = st.text_input("Ask a question based on your resume and job options:")

        if uploaded_pdf and uploaded_csv:
            with st.spinner("Processing files..."):
                # Extract resume text
                resume_text = get_pdf_text([uploaded_pdf])

                # Load job d escriptions from CSV
                job_data = get_csv_data(uploaded_csv)

                if job_data is not None and user_question:
                    # Use the LLM (Flan-T5) to answer the user's question based on the resume and job data
                    response = generate_llm_response(resume_text, user_question, job_data)

                    # Display the LLM's response
                    st.write(f"**Reply to your query: {user_question}**")
                    st.write(response)
        else:
            st.warning("Please upload both a resume PDF and job descriptions CSV to proceed.")

if __name__ == "__main__":
    main()
