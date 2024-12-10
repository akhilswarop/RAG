import os
import sys
import pandas as pd
from pyresparser import ResumeParser
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='resume_processing.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def add_pyresparser_path():
    """
    Adds the pyresparser path to sys.path.
    Adjust the path according to your setup.
    """
    pyresparser_path = 'pyresparser'  # Adjust as needed
    if pyresparser_path not in sys.path:
        sys.path.append(pyresparser_path)
        logging.info(f"Added {pyresparser_path} to sys.path")

def extract_resume_data(resume_path):
    """
    Extracts data from a single resume file.
    
    Parameters:
        resume_path (str): The file path to the resume.
    
    Returns:
        dict: Extracted resume data.
    """
    try:
        data = ResumeParser(resume_path).get_extracted_data()
        logging.info(f"Successfully processed: {resume_path}")
        return data
    except Exception as e:
        logging.error(f"Error processing {resume_path}: {e}")
        return {}

def process_resumes_in_folder(folder_path):
    """
    Processes all resume files in the specified folder.
    
    Parameters:
        folder_path (str): The directory containing resume files.
    
    Returns:
        pd.DataFrame: DataFrame containing extracted data from all resumes.
    """
    supported_extensions = ['.pdf', '.docx', '.doc']
    resumes = [file for file in Path(folder_path).iterdir() if file.suffix.lower() in supported_extensions]
    
    if not resumes:
        logging.warning(f"No resume files found in {folder_path}. Supported formats: {supported_extensions}")
        return pd.DataFrame()
    
    all_resume_data = []
    
    for resume in resumes:
        data = extract_resume_data(str(resume))
        if data:
            data['file_name'] = resume.name  # Keep track of which file the data came from
            all_resume_data.append(data)
    
    if all_resume_data:
        df = pd.DataFrame(all_resume_data)
        return df
    else:
        logging.warning("No data extracted from resumes.")
        return pd.DataFrame()

def main():
    # Add pyresparser path
    add_pyresparser_path()
    
    # Specify the folder containing resumes
    folder_path = 'resumes'  # Replace with your actual folder path
    
    # Process resumes
    resume_df = process_resumes_in_folder(folder_path)
    
    if not resume_df.empty:
        # Save the aggregated data to a CSV file
        output_csv = 'extracted_resume_data.csv'
        resume_df.to_csv(output_csv, index=False)
        logging.info(f"Extracted data saved to {output_csv}")
        print(f"Extracted data saved to {output_csv}")
    else:
        print("No data extracted from resumes.")

if __name__ == "__main__":
    main()
