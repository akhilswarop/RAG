import React, { useState } from "react";

const GuidanceGenerator = ({ parsedResume }) => {
  const [guidance, setGuidance] = useState(null);
  const [loading, setLoading] = useState(false);

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
      } else {
        console.error("Error:", data.error);
      }
    } catch (error) {
      console.error("Request failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 bg-white rounded-lg shadow-lg max-w-3xl mx-auto mt-6">
      <h2 className="text-2xl font-semibold mb-4 text-blue-800">Career Guidance</h2>

      {loading ? (
        <p className="text-blue-600">Generating guidance, please wait...</p>
      ) : guidance ? (
        <div className="space-y-6">
          {Object.entries(guidance).map(([model, response]) => (
            <div key={model} className="p-4 border rounded-lg">
              <h3 className="text-lg font-semibold text-gray-700">{model.toUpperCase()}'s Advice</h3>
              <p>{response || "No response"}</p>
            </div>
          ))}
          <button onClick={() => setGuidance(null)} className="bg-red-500 text-white px-4 py-2 rounded w-full">
            Clear Guidance
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
