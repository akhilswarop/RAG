import React, { useState } from "react";
import GuidanceGenerator from "./GuidanceGenerator";

const FileProcess = () => {
  const [parsedResume, setParsedResume] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showGuidance, setShowGuidance] = useState(false);

  const handleProcessResume = async () => {
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/process-resume", {
        method: "POST",
      });

      const data = await response.json();

      if (response.ok) {
        setParsedResume(data);
      } else {
        console.error("Processing failed:", data.error);
      }
    } catch (error) {
      console.error("Error processing file:", error);
    } finally {
      setLoading(false);
    }
  };

  if (showGuidance) {
    return <GuidanceGenerator parsedResume={parsedResume} />;
  }

  return (
    <div className="p-8 bg-white rounded-lg shadow-lg max-w-3xl mx-auto">
      <h2 className="text-2xl font-semibold mb-4 text-blue-800">Process Resume</h2>

      {loading ? (
        <p className="text-blue-600">Processing file, please wait...</p>
      ) : parsedResume ? (
        <div className="space-y-6">
          <div className="p-4 border rounded-lg">
            <h3 className="text-lg font-semibold text-gray-700">Personal Details</h3>
            <p><strong>Name:</strong> {parsedResume.name || "N/A"}</p>
            <p><strong>Email:</strong> {parsedResume.email || "N/A"}</p>
            <p><strong>Phone:</strong> {parsedResume.phone || "N/A"}</p>
            <p><strong>Location:</strong> {parsedResume.location || "N/A"}</p>
            <p><strong>LinkedIn:</strong> {parsedResume.linkedin || "N/A"}</p>
          </div>

          <div className="p-4 border rounded-lg">
            <h3 className="text-lg font-semibold text-gray-700">Skills</h3>
            <ul className="list-disc pl-5">
              {parsedResume.skills.length > 0 ? (
                parsedResume.skills.map((skill, index) => <li key={index}>{skill}</li>)
              ) : (
                <p>N/A</p>
              )}
            </ul>
          </div>

          <div className="p-4 border rounded-lg">
            <h3 className="text-lg font-semibold text-gray-700">Experience</h3>
            {parsedResume.experience.length > 0 ? (
              parsedResume.experience.map((exp, index) => (
                <div key={index} className="mt-2">
                  <p><strong>{exp.role}</strong> at {exp.company}</p>
                  <p className="text-gray-600">{exp.description}</p>
                </div>
              ))
            ) : (
              <p>N/A</p>
            )}
          </div>

          <div className="p-4 border rounded-lg">
            <h3 className="text-lg font-semibold text-gray-700">Education</h3>
            {parsedResume.education.length > 0 ? (
              parsedResume.education.map((edu, index) => (
                <div key={index} className="mt-2">
                  <p><strong>{edu.degree}</strong> at {edu.institution}</p>
                </div>
              ))
            ) : (
              <p>N/A</p>
            )}
          </div>

          <div className="p-4 border rounded-lg">
            <h3 className="text-lg font-semibold text-gray-700">Projects</h3>
            {parsedResume.projects.length > 0 ? (
              parsedResume.projects.map((proj, index) => (
                <div key={index} className="mt-2">
                  <p><strong>{proj.title || "Untitled Project"}</strong></p>
                  <p className="text-gray-600">{proj.description || "No description available."}</p>
                </div>
              ))
            ) : (
              <p>N/A</p>
            )}
          </div>

          <button
            onClick={() => setShowGuidance(true)}
            className="bg-green-600 text-white px-4 py-2 rounded w-full"
          >
            Generate Guidance
          </button>
        </div>
      ) : (
        <button
          onClick={handleProcessResume}
          className="bg-blue-600 text-white p-2 rounded w-full"
          disabled={loading}
        >
          Process Resume
        </button>
      )}
    </div>
  );
};

export default FileProcess;
