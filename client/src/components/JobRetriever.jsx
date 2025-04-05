import { useState } from "react";
import { terminal } from 'virtual:terminal';
import JobListingsDashboard from "./JobListingsDashboard";

const JobRetriever = ( {jobs} ) => {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchedJobs, setSearchedJobs] = useState([]);
  const [location, setLocation] = useState("");
  const [showSelectionPanel, setShowSelectionPanel] = useState(true);

  // Mocked from resume — replace this with dynamic resume-matched jobs
  const suggestedJobs = [...new Set(jobs.split(",").map((j) => j.trim()))];

  terminal.log(suggestedJobs)
  terminal.log(suggestedJobs.length)
  const [selectedJobs, setSelectedJobs] = useState([]);

  const searchJobs = async () => {
    setShowSelectionPanel(false)
    if (selectedJobs.length === 0) {
      setError("Please select at least one job title");
      return;
    }

    if (!location || location.trim() === "") {
      setError("Please enter a preferred location");
      return;
    }

    setLoading(true);
    setError(null);
    setSearchedJobs(selectedJobs);
    terminal.log("Searching for job titles:", selectedJobs, "in", location);

    try {
      const response = await fetch("http://localhost:5000/job-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobs: selectedJobs,
          location: location.trim(),
        }),
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

  return (
    <div className="p-4 max-w-4xl mx-auto bg-gray-50 rounded-lg shadow-sm">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">Job Retriever</h2>

      {/* Suggested Job Selection */}
      {showSelectionPanel && suggestedJobs.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            Based on your resume, select jobs you want to search:
          </h3>
          <div className="flex flex-wrap gap-2">
            {suggestedJobs.map((job, index) => {
              const isSelected = selectedJobs.includes(job);
              return (
                <button
                  key={index}
                  onClick={() =>
                    setSelectedJobs((prev) =>
                      isSelected
                        ? prev.filter((j) => j !== job)
                        : [...prev, job]
                    )
                  }
                  className={`px-3 py-1 rounded-full border text-sm transition-colors ${
                    isSelected
                      ? "bg-blue-600 text-white border-blue-700"
                      : "bg-gray-100 text-gray-700 border-gray-300"
                  }`}
                >
                  {job}
                </button>
              );
            })}
          </div>
          {/* Location Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Preferred Job Location
        </label>
        <input
          type="text"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="e.g. Bangalore, Remote, New York"
          disabled={loading}
        />
      </div>

      {/* Search Button */}
      <div className="mb-4">
        <button
          onClick={searchJobs}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors disabled:bg-blue-400"
        >
          {loading ? "Searching..." : "Search Jobs"}
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-3 mb-4 bg-red-100 border border-red-300 text-red-700 rounded">
          {error}
        </div>
      )}
        </div>
      )}

      {/* Loading Spinner */}
        {loading && (
        <div className="flex justify-center items-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
          <p className="ml-3 text-blue-600">Searching for jobs, please wait...</p>
        </div>
      )}

      {/* Display selected jobs */}
      {!loading && searchedJobs.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            Selected job titles:
          </h3>
          <div className="flex flex-wrap gap-2">
            {searchedJobs.map((job, index) => (
              <span
                key={index}
                className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm"
              >
                {job}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {!loading && response && (
        <>
          {terminal.log("Response data being passed to component:", response)}
          <JobListingsDashboard jobData={response} />
        </>
      )}

      {!loading && !response && !error && (
        <p className="text-center text-gray-500 py-8">
          Select job roles and location to find listings.
        </p>
      )}
    </div>
  );
};

export default JobRetriever;
