import React, { useState } from "react";
import GuidanceGenerator from "./GuidanceGenerator";
import BeatLoader from "react-spinners/BeatLoader";
import { useNavigate } from "react-router-dom";
import { FileText, User, Briefcase, GraduationCap, Code, Mail, Phone, MapPin, Linkedin, ChevronRight } from "lucide-react";

const FileProcess = () => {
  const [parsedResume, setParsedResume] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showGuidance, setShowGuidance] = useState(false);
  const navigate = useNavigate();

  const handleProcessResume = async () => {
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/process-resume", {
        method: "POST",
      });

      const data = await response.json();

      if (response.ok) {
        setParsedResume(data);
        window.scrollTo({ top: 0, behavior: "smooth" });
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
    <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Resume Analysis</h2>
        <p className="text-gray-600">Let AI extract and organize your resume information</p>
      </div>

      {!parsedResume ? (
        <div className="bg-white rounded-xl shadow-md overflow-hidden p-8 text-center max-w-md mx-auto border border-gray-100">
          <div className="mb-6">
            <FileText className="h-16 w-16 text-blue-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Ready to Analyze Your Resume</h3>
            <p className="text-gray-600 mb-6">Click the button below to extract key information from your resume</p>
          </div>

          <button
            onClick={handleProcessResume}
            className="w-full flex items-center justify-center gap-3 text-white py-4 px-6 rounded-lg transition bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 shadow-md hover:shadow-lg disabled:opacity-70 text-lg font-medium"
            disabled={loading}
          >
            {loading ? (
              <BeatLoader color="#ffffff" size={10} />
            ) : (
              <>
                <span>Process Resume</span>
                <ChevronRight className="h-5 w-5" />
              </>
            )}
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl shadow-sm border border-blue-100">
            <div className="flex items-center mb-4">
              <User className="h-6 w-6 text-blue-600 mr-3" />
              <h3 className="text-xl font-bold text-gray-800">Personal Details</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {parsedResume.email && parsedResume.email!="N/A" && (
                <div className="flex items-center">
                  <Mail className="h-5 w-5 text-gray-500 mr-2" />
                  <span className="text-gray-700">{parsedResume.email}</span>
                </div>
              )}
              {parsedResume.phone && parsedResume.phone!="N/A" && (
                <div className="flex items-center">
                  <Phone className="h-5 w-5 text-gray-500 mr-2" />
                  <span className="text-gray-700">{parsedResume.phone}</span>
                </div>
              )}
              {parsedResume.location && parsedResume.location != "N/A" && (
                <div className="flex items-center">
                  <MapPin className="h-5 w-5 text-gray-500 mr-2" />
                  <span className="text-gray-700">{parsedResume.location}</span>
                </div>
              )}
              {parsedResume.linkedin && parsedResume.linkedin != "N/A"  && (
                <div className="flex items-center">
                  <Linkedin className="h-5 w-5 text-gray-500 mr-2" />
                  <span className="text-gray-700">{parsedResume.linkedin}</span>
                </div>
              )}
            </div>          
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center mb-4">
              <Code className="h-6 w-6 text-green-600 mr-3" />
              <h3 className="text-xl font-bold text-gray-800">Skills</h3>
            </div>
            {parsedResume.skills.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {parsedResume.skills.map((skill, index) => (
                  <span key={index} className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
                    {skill}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 italic">No skills listed</p>
            )}
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center mb-4">
              <Briefcase className="h-6 w-6 text-indigo-600 mr-3" />
              <h3 className="text-xl font-bold text-gray-800">Experience</h3>
            </div>
            {parsedResume.experience.length > 0 ? (
              <div className="space-y-5">
                {parsedResume.experience.map((exp, index) => (
                  <div key={index} className="pb-4 border-b border-gray-100 last:border-0 last:pb-0">
                    <h4 className="font-semibold text-gray-800">{exp.role}</h4>
                    <p className="text-indigo-600 mb-2">{exp.company}</p>
                    <p className="text-gray-600 text-sm">{exp.description}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 italic">No experience listed</p>
            )}
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center mb-4">
              <GraduationCap className="h-6 w-6 text-purple-600 mr-3" />
              <h3 className="text-xl font-bold text-gray-800">Education</h3>
            </div>
            {parsedResume.education.length > 0 ? (
              <div className="space-y-4">
                {parsedResume.education.map((edu, index) => (
                  <div key={index} className="pb-3 border-b border-gray-100 last:border-0 last:pb-0">
                    <h4 className="font-semibold text-gray-800">{edu.degree}</h4>
                    <p className="text-purple-600">{edu.institution}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 italic">No education listed</p>
            )}
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center mb-4">
              <Code className="h-6 w-6 text-amber-600 mr-3" />
              <h3 className="text-xl font-bold text-gray-800">Projects</h3>
            </div>
            {parsedResume.projects.length > 0 ? (
              <div className="space-y-5">
                {parsedResume.projects.map((proj, index) => (
                  <div key={index} className="pb-4 border-b border-gray-100 last:border-0 last:pb-0">
                    <h4 className="font-semibold text-gray-800">{proj.title || "Untitled Project"}</h4>
                    <p className="text-gray-600 text-sm">{proj.description || "No description available."}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 italic">No projects listed</p>
            )}
          </div>

          <button
            onClick={() => navigate("/guidance-generate", { state: { parsedResume } })}
            className="w-full flex items-center justify-center gap-2 text-white py-4 px-6 rounded-lg transition bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 shadow-md hover:shadow-lg text-lg font-medium"
          >
            Generate Career Guidance
            <ChevronRight className="h-5 w-5" />
          </button>
        </div>
      )}
    </div>
  );
};

export default FileProcess;