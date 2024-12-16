from serpapi import GoogleSearch

params = {
  "engine": "google_jobs",
  "q": "barista new york",
  "hl": "en",
  "api_key": "22c744e7201db68cce330bf72f58d1c9a81529af3361865d333daa41a32e1551"
}

search = GoogleSearch(params)
results = search.get_dict()
jobs_results = results["jobs_results"]
print(jobs_results)