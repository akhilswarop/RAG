import sys
sys.path.append('C:/Users/swaro/OneDrive/Documents/GitHub/RAG/pyresparser')



import pyresparser

data = pyresparser.ResumeParser('White Turqoise Creative Simple Medical Personnel Nurse Resume.pdf').get_extracted_data()
print(data)