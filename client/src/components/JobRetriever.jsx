import { useState } from "react";
import { terminal } from 'virtual:terminal';
import JobListingsDashboard from "./JobListingsDashboard";

const JobRetriever = ({ jobs }) => {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchedJobs, setSearchedJobs] = useState([]);

  const searchJobs = async () => {
    if (!jobs || jobs.trim() === "") {
      setError("Please enter at least one job title");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    const jobTitles = jobs.split(",").map(job => job.trim()).filter(job => job !== "");
    
    if (jobTitles.length === 0) {
      setLoading(false);
      setError("Please enter valid job titles");
      return;
    }
    
    setSearchedJobs(jobTitles);
    terminal.log("Searching for job titles:", jobTitles);
    
    try {
      const response = await fetch("http://localhost:5000/job-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobs: jobTitles }),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setResponse(data);
        terminal.log("Job response data:", data);
      } else {
        setError(`Error: ${data.message || "Failed to retrieve jobs"}`);
        terminal.log("API error:", data);
      }
    } catch (error) { 
      setError("Failed to connect to the server. Please try again later.");
      console.error("Request failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      searchJobs();
    }
  };

  return (
    <div className="p-4 max-w-4xl mx-auto bg-gray-50 rounded-lg shadow-sm">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">Job Retriever</h2>
      
      <div className="mb-4">
        <p className="text-sm text-gray-600 mb-2">
          Enter job titles separated by commas
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={jobs}
            onKeyDown={handleKeyDown}
            className="flex-1 border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Software Engineer, Data Scientist, Product Manager"
            disabled={loading}
          />
          <button 
            onClick={searchJobs} 
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors disabled:bg-blue-400"
          >
            {loading ? "Searching..." : "Search Jobs"}
          </button>
        </div>
      </div>
      
      {error && (
        <div className="p-3 mb-4 bg-red-100 border border-red-300 text-red-700 rounded">
          {error}
        </div>
      )}
      
      {loading && (
        <div className="flex justify-center items-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
          <p className="ml-3 text-blue-600">Searching for jobs, please wait...</p>
        </div>
      )}
      
      {!loading && searchedJobs.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Searched job titles:</h3>
          <div className="flex flex-wrap gap-2">
            {searchedJobs.map((job, index) => (
              <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm">
                {job}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {!loading && response && (
        <>
          {terminal.log("Response data being passed to component:", response)}
          <JobListingsDashboard jobData={response} />
        </>
      )}
      
      {!loading && !response && !error && (
        <p className="text-center text-gray-500 py-8">
          Enter job titles and click "Search Jobs" to find listings
        </p>
      )}
    </div>
  );
};

export default JobRetriever;