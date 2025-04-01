import React, { useEffect, useState } from "react";
import GuidanceGenerateComponent from "../components/GuidanceGenerator";
import { useLocation, useNavigate } from "react-router-dom";
import { AlertCircle } from "lucide-react";

const GuidanceGenerate = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const parsedResume = location.state?.parsedResume;

  useEffect(() => {
    // Check if resume data exists
    if (!parsedResume) {
      // Set a short timeout to avoid flash of error message
      const timer = setTimeout(() => {
        navigate("/upload", { 
          state: { error: "No resume data found. Please upload your resume first." }
        });
      }, 100);
      
      return () => clearTimeout(timer);
    }
    
    setIsLoading(false);
  }, [parsedResume, navigate]);

  if (isLoading) {
    return (
      <div className="container mx-auto p-8 flex items-center justify-center min-h-[50vh]">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-12 w-3/4 bg-gray-200 rounded-md mb-4"></div>
          <div className="h-64 w-full bg-gray-100 rounded-lg"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4 md:p-8">
      <div className="bg-white shadow-lg rounded-xl overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-blue-800 p-6">
          <h1 className="text-2xl md:text-3xl font-bold text-white">Career Guidance Generator</h1>
          <p className="text-blue-100 mt-2">Personalized recommendations based on your resume</p>
        </div>
        
        <div className="p-4 md:p-6 bg-gray-50 rounded-lg m-4 md:m-6 shadow-inner">
          {parsedResume && Object.keys(parsedResume).length > 0 ? (
            <GuidanceGenerateComponent parsedResume={parsedResume} />
          ) : (
            <div className="flex items-center p-4 bg-yellow-50 rounded-md border border-yellow-200">
              <AlertCircle className="h-5 w-5 text-yellow-500 mr-3" />
              <p className="text-yellow-700">Resume data appears incomplete. Some guidance may be limited.</p>
            </div>
          )}
        </div>
        
        <div className="p-4 bg-gray-50 border-t border-gray-100 text-sm text-gray-500 text-center">
          All guidance is generated based on the information provided in your resume.
        </div>
      </div>
    </div>
  );
};

export default GuidanceGenerate;