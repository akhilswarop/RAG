import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { terminal } from "virtual:terminal";
import { CheckCircle, XCircle, Upload, File, FileText, Trash2, Info, ChevronRight, BarChart, Shield, Zap, Clock } from "lucide-react";

const FileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle, loading, success, error
  const [dragActive, setDragActive] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setUploadMessage("");
      setUploadStatus("idle");
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
      setUploadMessage("");
      setUploadStatus("idle");
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadMessage("");
    setUploadStatus("idle");
  };

  const getFileIcon = () => {
    if (!selectedFile) return null;
    
    const fileType = selectedFile.type;
    if (fileType.includes("pdf")) {
      return <FileText size={24} className="text-red-500" />;
    } else if (fileType.includes("doc") || fileType.includes("word")) {
      return <FileText size={24} className="text-blue-500" />;
    } else {
      return <File size={24} className="text-gray-500" />;
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMessage("Please select a file first.");
      setUploadStatus("error");
      return;
    }

    setUploadStatus("loading");
    console.log("Uploading file:", selectedFile);
    
    const formData = new FormData();
    formData.append("file", selectedFile);
    
    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });
      
      const data = await response.json();
      terminal.log(data);
      
      if (response.ok) {
        setUploadMessage("File uploaded successfully!");
        setUploadStatus("success");
        console.log("Uploaded file:", data.filename);
        
        // Redirect after a short delay to show success message
        setTimeout(() => {
          navigate("/file-process");
        }, 1500);
      } else {
        setUploadMessage(data.message || "Failed to upload file.");
        setUploadStatus("error");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setUploadMessage("Error occurred during upload.");
      setUploadStatus("error");
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-center text-blue-800 mb-2">Career Document Analyzer</h1>
      <p className="text-center text-gray-600 mb-8">Upload your resume and get AI-powered career insights and recommendations</p>
      
      <div className="grid md:grid-cols-5 gap-8">
        {/* Feature Highlights - Takes 2/5 of the space */}
        <div className="md:col-span-2">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl shadow-sm border border-blue-100 h-full">
            <h2 className="text-xl font-bold text-blue-800 mb-6">Powerful Career Analysis</h2>
            
            {/* Feature Cards */}
            <div className="space-y-4">
              <div className="bg-white p-4 rounded-lg shadow-sm border border-blue-100 transform transition-all hover:scale-105">
                <div className="flex items-start">
                  <div className="bg-blue-100 p-2 rounded-full mr-4">
                    <BarChart className="text-blue-600" size={20} />
                  </div>
                  <div>
                    <h3 className="font-medium text-blue-800">Skills Assessment</h3>
                    <p className="text-sm text-gray-600">AI-powered analysis of your strengths and improvement areas</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg shadow-sm border border-indigo-100 transform transition-all hover:scale-105">
                <div className="flex items-start">
                  <div className="bg-indigo-100 p-2 rounded-full mr-4">
                    <Zap className="text-indigo-600" size={20} />
                  </div>
                  <div>
                    <h3 className="font-medium text-indigo-800">Job Matching</h3>
                    <p className="text-sm text-gray-600">Find the perfect job opportunities that match your profile</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg shadow-sm border border-purple-100 transform transition-all hover:scale-105">
                <div className="flex items-start">
                  <div className="bg-purple-100 p-2 rounded-full mr-4">
                    <Clock className="text-purple-600" size={20} />
                  </div>
                  <div>
                    <h3 className="font-medium text-purple-800">Instant Feedback</h3>
                    <p className="text-sm text-gray-600">Get real-time suggestions to improve your resume</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg shadow-sm border border-green-100 transform transition-all hover:scale-105">
                <div className="flex items-start">
                  <div className="bg-green-100 p-2 rounded-full mr-4">
                    <Shield className="text-green-600" size={20} />
                  </div>
                  <div>
                    <h3 className="font-medium text-green-800">Privacy Focused</h3>
                    <p className="text-sm text-gray-600">Your documents are processed securely and privately</p>
                  </div>
                </div>
              </div>
            </div>
            
            
          </div>
        </div>
        
        {/* File Upload Area - Takes 3/5 of the space */}
        <div className="md:col-span-3">
          <div className="bg-white p-8 rounded-xl shadow-lg border border-gray-100">
            <div className="flex items-center justify-between mb-6">
              {/* <h2 className="text-2xl font-bold text-blue-800">Upload Career Documents</h2> */}
              <div className="flex items-center text-sm text-gray-500">
                <Info size={16} className="mr-1" />
                <span>Supported formats: PDF</span>
              </div>
            </div>
            
            {/* File Upload Area */}
            <div 
              className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors ${
                dragActive 
                  ? "border-blue-500 bg-blue-50" 
                  : selectedFile 
                    ? "border-green-200 bg-green-50" 
                    : "border-blue-200 hover:border-blue-300 hover:bg-blue-50"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                className="hidden"
                id="fileUpload"
                onChange={handleFileChange}
                accept=".pdf,.doc,.docx"
              />
              
              {!selectedFile ? (
                <label htmlFor="fileUpload" className="cursor-pointer block">
                  <Upload size={40} className="mx-auto mb-4 text-blue-500" />
                  <p className="text-gray-700 font-medium mb-2">
                    Drag and drop files here or click to browse
                  </p>
                  <p className="text-gray-500 text-sm">
                    Upload your resume, cover letter, or other career documents
                  </p>
                </label>
              ) : (
                <div className="py-4">
                  <div className="flex items-center justify-center mb-3">
                    {getFileIcon()}
                    <span className="ml-2 font-medium text-gray-700">{selectedFile.name}</span>
                  </div>
                  <div className="text-sm text-gray-500">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </div>
                  <button 
                    onClick={handleRemoveFile}
                    className="mt-4 text-red-500 flex items-center justify-center mx-auto hover:text-red-700"
                  >
                    <Trash2 size={16} className="mr-1" /> Remove file
                  </button>
                </div>
              )}
            </div>
            
            {/* Upload Status Message */}
            {uploadMessage && (
              <div className={`mt-4 p-3 rounded-lg ${
                uploadStatus === "success" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
              }`}>
                <p className="flex items-center">
                  {uploadStatus === "success" ? 
                    <CheckCircle size={16} className="mr-2" /> : 
                    <XCircle size={16} className="mr-2" />
                  }
                  {uploadMessage}
                </p>
              </div>
            )}
            
            {/* Upload Button */}
            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploadStatus === "loading"}
              className={`mt-6 w-full flex items-center justify-center gap-2 text-white py-4 rounded-lg transition-all ${
                !selectedFile 
                  ? "bg-gray-400 cursor-not-allowed" 
                  : uploadStatus === "success"
                    ? "bg-green-600 hover:bg-green-700"
                    : uploadStatus === "error"
                      ? "bg-red-600 hover:bg-red-700"
                      : uploadStatus === "loading"
                        ? "bg-blue-400 cursor-wait"
                        : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              {uploadStatus === "loading" ? (
                <div className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </div>
              ) : uploadStatus === "success" ? (
                <>
                  <CheckCircle size={20} /> Uploaded Successfully!
                </>
              ) : uploadStatus === "error" ? (
                <>
                  <XCircle size={20} /> Upload Failed
                </>
              ) : (
                <>
                  <Upload size={20} /> Upload Document
                </>
              )}
            </button>

            {/* Extra Information */}
            <div className="mt-6 bg-blue-50 p-4 rounded-lg">
              <h3 className="font-medium text-blue-800 mb-2">Why upload your documents?</h3>
              <ul className="text-sm text-gray-700 space-y-2">
                <li className="flex items-start">
                  <CheckCircle size={16} className="mr-2 text-green-500 mt-1 flex-shrink-0" />
                  <span>Get personalized career insights and recommendations</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle size={16} className="mr-2 text-green-500 mt-1 flex-shrink-0" />
                  <span>Identify skills gaps and opportunities for improvement</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle size={16} className="mr-2 text-green-500 mt-1 flex-shrink-0" />
                  <span>Match your profile with suitable job positions</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      

    </div>
  );
};

export default FileUpload;