import { terminal } from 'virtual:terminal'
import React, { useState } from "react";
import ReactMarkdown from 'react-markdown'
import JobRetriever from "./JobRetriever";

const GuidanceGenerator = ({ parsedResume }) => {
  const [guidance, setGuidance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showJobRetriever, setShowJobRetriever] = useState(false)
  const generateGuidance = async () => {
    if (!parsedResume || !parsedResume.skills || !parsedResume.education) {
      alert("No valid resume data found.");
      return;
    }

    
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/generate-guidance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          skills: parsedResume.skills,
          academic_history: parsedResume.education.map((edu) => edu.degree).join(", "),
        }),
      });

      const data = await response.json();
      if (response.ok) {
        setGuidance(data);
        
                
        
        } 
    } catch (error) { 
      console.error("Request failed:", error);
    } finally {
      setLoading(false);
    }
  };
  if (showJobRetriever) {
    return <JobRetriever jobs={guidance.top_job_titles.map(job => job.title).join(",")} />
    ;
  }
  return (
    <div className="bg-white p-8 rounded-xl shadow-md">
      <h2 className="text-2xl font-semibold mb-4 text-blue-800">Career Guidance</h2>

      {loading ? (
        <p className="text-blue-600">Generating guidance, please wait...</p>
      ) : guidance ? (
        <div className="space-y-8">
          {Object.entries(guidance.generations).map(([model, response]) => (            
            <div key={model} className="p-4 border rounded-lg">
              <h3 className="text-lg font-semibold text-gray-700 text-center">{model.toUpperCase()}'s Advice</h3>
              <hr className="pt-4 pb-6"></hr>              
              <p><ReactMarkdown>{response}</ReactMarkdown></p>
            </div>
          ))}
          <button 
            onClick={() => setShowJobRetriever(true)} 
            className="bg-blue-600 text-white px-4 py-2 rounded w-full"
          >
            Search for Jobs
          </button>
        </div>
      ) : (
        <button onClick={generateGuidance} className="bg-green-600 text-white p-2 rounded w-full">
          Generate Career Guidance
        </button>
      )}
    </div>
  );
};

export default GuidanceGenerator;