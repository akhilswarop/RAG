import wikipediaapi
import re
import csv

def extract_education(page_name):
    # Initialize Wikipedia API with user-agent
    user_agent = "MyScrapingBot/1.0 (ganeshkaran2008@gmail.com)"
    wiki = wikipediaapi.Wikipedia('en', user_agent=user_agent)
    
    # Fetch the page
    page = wiki.page(page_name)
    
    # Check if the page exists
    if not page.exists():
        print(f"Page '{page_name}' does not exist.")
        return None
    
    # Extract the page summary and sections
    education = ""
    
    for section in page.sections:
        # Some pages may have 'Early Life' or 'Education' sections
        if 'education' in section.title.lower() or 'early life' in section.title.lower():
            education += section.text

    # Simple regex to extract degree keywords
    degrees = re.findall(r'(Bachelor|Master|PhD|Doctorate|Degree|Diploma|B\.S\.|M\.S\.)', education, re.IGNORECASE)
    
    if education:
        return education, degrees
    else:
        print(f"No education information found on '{page_name}'")
        return None

def save_to_csv(data, filename='education_details.csv'):
    # Define CSV header
    header = ['Name', 'Education Details', 'Degrees Found']

    # Open the file in write mode
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Write the data rows
        for row in data:
            writer.writerow(row)
    
    print(f"Data successfully saved to {filename}")

# Example usage:
famous_people = ["Albert Einstein", "Elon Musk", "Marie Curie"]
scraped_data = []

for person in famous_people:
    education_info = extract_education(person)
    if education_info:
        # Append person's name, education details, and degrees to the list
        scraped_data.append([person, education_info[0], ', '.join(education_info[1])])

# Save the data to a CSV file
save_to_csv(scraped_data)