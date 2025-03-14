import React from 'react';
import { ExternalLink } from 'lucide-react';

const JobListingsDashboard = ({ jobData }) => {
  // Check if jobData exists and has the expected structure
  if (!jobData || !Array.isArray(jobData) || jobData.length === 0 || !Array.isArray(jobData[0])) {
    return <div className="text-red-500 p-4">No job data available</div>;
  }

  // Extract the job listings array from the nested structure
  const jobListings = jobData[0];
  
  // Limit to 5 jobs as requested
  const limitedJobListings = jobListings.slice(0, 5);
  
  // For debugging - log the data being rendered
  console.log("Rendering job listings:", limitedJobListings);

  return (
    <div className="w-full max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Job Listings</h1>
      
      <div className="grid gap-6">
        {limitedJobListings.map((job, index) => (
          <div key={index} className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-xl font-semibold text-blue-600">{job.Title}</h2>
                <p className="text-gray-700 font-medium mt-1">{job.Company}</p>
                <p className="text-gray-500 mt-1 flex items-center">
                  <span className="mr-2">📍</span> {job.Location || "Remote"}
                </p>
              </div>
              <span className="bg-blue-100 text-blue-800 text-xs font-semibold px-2.5 py-0.5 rounded">
                via {job.Via}
              </span>
            </div>
            
            <div className="mt-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Description</h3>
              <p className="text-gray-600 text-sm">
                {job["Description (truncated)"] || "No description available"}
              </p>
            </div>
            
            <div className="mt-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Apply Options</h3>
              <div className="flex flex-wrap gap-2">
                {job["Apply Options"] && job["Apply Options"].slice(0, 3).map((option, optionIndex) => (
                  <a 
                    key={optionIndex} 
                    href={option.link}
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded text-sm text-gray-700"
                  >
                    {option.title}
                    <ExternalLink size={14} className="ml-1" />
                  </a>
                ))}
                {job["Apply Options"] && job["Apply Options"].length > 3 && (
                  <span className="text-sm text-gray-500 flex items-center">
                    +{job["Apply Options"].length - 3} more
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default JobListingsDashboard;