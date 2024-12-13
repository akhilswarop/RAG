import io
import subprocess
import json
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import docx2txt

def extract_text_from_pdf(pdf_path):
    '''
    Helper function to extract the plain text from .pdf files

    :param pdf_path: path to PDF file to be extracted (remote or local)
    :return: iterator of string of extracted text
    '''
    if not isinstance(pdf_path, io.BytesIO):
        # Extract text from local pdf file
        with open(pdf_path, 'rb') as fh:
            try:
                for page in PDFPage.get_pages(
                        fh,
                        caching=True,
                        check_extractable=True
                ):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(
                        resource_manager,
                        fake_file_handle,
                        laparams=LAParams()
                    )
                    page_interpreter = PDFPageInterpreter(
                        resource_manager,
                        converter
                    )
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()

                    yield text

                    # Close open handles
                    converter.close()
                    fake_file_handle.close()
            except PDFSyntaxError:
                print(f"Error: PDFSyntaxError encountered while processing {pdf_path}")
                return
    else:
        # Extract text from remote pdf file
        try:
            for page in PDFPage.get_pages(
                    pdf_path,
                    caching=True,
                    check_extractable=True
            ):
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    fake_file_handle,
                    laparams=LAParams()
                )
                page_interpreter = PDFPageInterpreter(
                    resource_manager,
                    converter
                )
                page_interpreter.process_page(page)

                text = fake_file_handle.getvalue()
                yield text

                converter.close()
                fake_file_handle.close()
        except PDFSyntaxError:
            print("Error: PDFSyntaxError encountered while processing remote PDF")
            return

def extract_text_from_docx(doc_path):
    '''
    Helper function to extract plain text from .docx files

    :param doc_path: path to .docx file to be extracted
    :return: string of extracted text
    '''
    try:
        temp = docx2txt.process(doc_path)
        text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        return ' '.join(text)
    except KeyError:
        print(f"Error: KeyError encountered while processing {doc_path}")
        return ' '

def get_resume_text(file_path):
    '''
    Helper function to return all text from pdf or docx
    '''
    if file_path.endswith('.pdf'):
        full_text = []
        for page_text in extract_text_from_pdf(file_path):
            full_text.append(page_text)
        return "\n".join(full_text)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Supported formats are .pdf and .docx.")

def parse_resume_with_ollama(resume_text):
    '''
    Function to parse resume text using Ollama's gemma2:2b model and return structured JSON data.
    '''
    # Define the prompt with strict instructions
    prompt = f"""
You are a helpful assistant that extracts structured data from the given resume text.

Important Instructions:
1. Output Format: Return only a single JSON object that strictly follows the requested structure.
2. No Extra Text: Do not include any additional text, explanations, code fences, triple backticks, or any formatting beyond the JSON object.
3. No Missing Keys: Include all keys listed below, even if their values are empty or blank.
4. Data Structure:
   - name: string
   - location: string
   - email: string
   - phone: string
   - linkedin: string
   - experience: an array of objects, each with keys "role", "company", "location", "start_date", "end_date", "description"
   - projects: an array of objects, each with keys "title", "start_date", "end_date", "description", "tech_stack" (where "tech_stack" is an array of strings)
   - education: an array of objects, each with keys "degree", "institution", "start_date", "end_date", "gpa"
   - skills: an array of strings
   - extracurricular_activities: an array of objects, each with keys "activity" and "description"

If a field is not found in the resume, return an empty string "" for strings or an empty array [] for lists.

Resume Text:
{resume_text}

Your task:
Extract the requested information from the resume text and return only one valid JSON object, strictly following the structure and instructions above.
"""

    # Call Ollama's gemma2:2b model via subprocess
    try:
        result = subprocess.run(
            ["ollama", "run", "gemma2:2b"],
            input=prompt,
            text=True,
            capture_output=True,
            check=True  # This will raise CalledProcessError if the command fails
        )
    except subprocess.CalledProcessError as e:
        print("Error running ollama:", e.stderr)
        return None

    # The model should return JSON. We can load it directly.
    response_text = result.stdout.strip()

    # Debug: Print the raw response
    print("=======")
    print(response_text)
    print("=======")

    # Clean the response by removing any code fences or markdown formatting
    cleaned_text = response_text.replace("```json", "").replace("```", "").strip()

    # Debug: Print the cleaned JSON
    print("Cleaned JSON:")
    print(cleaned_text)
    print("=======")

    # Try to parse the cleaned JSON
    try:
        parsed = json.loads(cleaned_text)
        return parsed
    except json.JSONDecodeError as e:
        print("JSON parsing failed:")
        print(e)
        print("Extracted JSON:")
        print(cleaned_text)
        return None

# Example usage:
if __name__ == "__main__":
    # Specify the path to your resume file (PDF or DOCX)
    file_path = "resumes/GaneshkaranM_RESUME (1).pdf"  # Update this path accordingly

    try:
        # Extract text from the resume
        text = get_resume_text(file_path)
        print("Resume text extracted successfully.")

        # Parse the resume text to JSON using Ollama
        extracted_data = parse_resume_with_ollama(text)

        if extracted_data:
            print("Extracted Resume Data in JSON:")
            print(json.dumps(extracted_data, indent=4))
        else:
            print("No valid JSON extracted.")
    except Exception as e:
        print(f"An error occurred: {e}")
