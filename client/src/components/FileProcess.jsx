import React, { useState } from "react";
import FileUpload from "../components/FileUpload";
import ExtractedData from "../components/ExtractedData";

const FileProcess = () => {
  const [filename, setFilename] = useState(null);
  const [parsedResume, setParsedResume] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleProcessResume = async () => {
    if (!filename) {
      alert("No file uploaded!");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/process-resume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }), // Send only filename
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

  return (
    <div className="p-8 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-semibold mb-4 text-blue-800">Process Resume</h2>

      {!filename ? (
        <FileUpload onFileUpload={setFilename} /> // Get filename from FileUpload
      ) : loading ? (
        <p className="text-blue-600">Processing file, please wait...</p>
      ) : parsedResume ? (
        <ExtractedData data={parsedResume} />
      ) : (
        <button onClick={handleProcessResume} className="bg-blue-600 text-white p-2 rounded">
          Process Resume
        </button>
      )}
    </div>
  );
};

export default FileProcess;
