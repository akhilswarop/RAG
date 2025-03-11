import React, { useState } from "react";
import { terminal } from 'virtual:terminal'
import { CheckCircle, XCircle, Upload } from "lucide-react"; // Import icons
import FileProcess from "./FileProcess"; 

const FileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [showFileProcess, setShowFileProcess] = useState(false); // New state

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      console.log("Please select a file first.");
      return;
    }

    console.log("Uploading file:", selectedFile);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      terminal.log(data)

      if (response.ok) {
        setUploadMessage("File uploaded successfully!");
        console.log("Uploaded file:", data.filename);
        setShowFileProcess(true); // Show FileProcess after upload

      } else {
        setUploadMessage("Failed to upload file.");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setUploadMessage("Error occurred during upload.");
    }
  };

  return (
    <div className="bg-white p-8 rounded-xl shadow-md">
      <h2 className="text-2xl font-semibold mb-6 text-blue-800">
        Upload Career Documents
      </h2>

      <div className="border-2 border-dashed border-blue-200 p-10 text-center">
        <input
          type="file"
          className="hidden"
          id="fileUpload"
          onChange={handleFileChange}
        />

        <label htmlFor="fileUpload" className="cursor-pointer">
          <p className="text-gray-600 mb-2">
            {selectedFile
              ? `Selected: ${selectedFile.name}`
              : "Click to upload files"}
          </p>
        </label>
      </div>
      
      {selectedFile && (
  <button
    onClick={handleUpload}
    className={`mt-6 w-full flex items-center justify-center gap-2 text-white py-3 rounded-lg transition ${
      uploadMessage === "File uploaded successfully!"
        ? "bg-green-600 hover:bg-green-700"
        : uploadMessage
        ? "bg-red-600 hover:bg-red-700"
        : "bg-blue-600 hover:bg-blue-700"
    }`}
  >
    {uploadMessage === "File uploaded successfully!" ? (
      <>
        <CheckCircle size={20} /> Uploaded Successfully!
      </>
    ) : uploadMessage ? (
      <>
        <XCircle size={20} /> Upload Failed
      </>
    ) : (
      <>
        <Upload size={20} /> Upload Document
      </>
    )}
  </button>
)}
      {showFileProcess && <FileProcess />}

    </div>
  );
};

export default FileUpload;
