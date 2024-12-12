import os
import sys
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

def main():
    # Add pyresparser path
    add_pyresparser_path()
    
    # Prompt the user to input the resume file path
    resume_path = input("GaneshkaranM_RESUME (1).pdf").strip()
    
    # Validate the file path and format
    if not os.path.isfile(resume_path):
        print("Error: The specified file does not exist. Please check the path and try again.")
        logging.error(f"Invalid file path: {resume_path}")
        return
    
    supported_extensions = ['.pdf', '.docx', '.doc']
    if not Path(resume_path).suffix.lower() in supported_extensions:
        print(f"Error: Unsupported file format. Supported formats are: {supported_extensions}")
        logging.error(f"Unsupported file format: {resume_path}")
        return
    
    # Process the single resume file
    resume_data = extract_resume_data(resume_path)
    
    if resume_data:
        # Print the extracted data in the terminal
        print("\nExtracted Resume Data:")
        for key, value in resume_data.items():
            print(f"{key}: {value}")
    else:
        print("No data extracted from the resume.")

if __name__ == "__main__":
    main()
