import { useState } from "react";
import { terminal } from 'virtual:terminal';
import JobListingsDashboard from "./JobListingsDashboard";

const JobRetriever = ({ jobs }) => {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);


  const searchJobs = async () => {
    if (!jobs) return;
    
    setLoading(true);
    const jobTitles = jobs.split(",").map(job => job.trim());
    terminal.log(jobTitles)
    try {
      const response = await fetch("http://localhost:5000/job-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobs: jobTitles }),
      });

      const data = await response.json();
      if (response.ok) {
        setResponse(data);
        terminal.log("Job response", response);
      }
    } catch (error) { 
      console.error("Request failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h2 className="text-xl font-semibold mb-4">Job Retriever</h2>

      <button 
        onClick={searchJobs} 
        className="bg-blue-600 text-white px-4 py-2 rounded mb-4 w-full"
      >
        {loading ? "Searching..." : "Search Jobs"}
      </button>

      {loading ? (
  <p className="text-blue-600">Searching for jobs, please wait...</p>
) : response ? (
  <>
    {terminal.log("Response data being passed to component:", response)}
    <JobListingsDashboard jobData={response} />
  </>
) : null}
    </div>
  );
}

export default JobRetriever;