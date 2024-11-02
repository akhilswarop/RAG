import sys
sys.path.append('C:/Users/swaro/OneDrive/Documents/GitHub/RAG/pyresparser')



import pyresparser

data = pyresparser.ResumeParser('Akhil_Swarop.docx').get_extracted_data()
print(data)