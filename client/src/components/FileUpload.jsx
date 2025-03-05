import React, { useState } from 'react';

const FileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handleUpload = () => {
    if (selectedFile) {
      console.log('Uploading file:', selectedFile);
      // Implement actual upload logic here
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
        
        <label 
          htmlFor="fileUpload" 
          className="cursor-pointer"
        >
          <p className="text-gray-600 mb-2">
            {selectedFile 
              ? `Selected: ${selectedFile.name}` 
              : 'Click to upload files'}
          </p>
        </label>
      </div>
      
      {selectedFile && (
        <button 
          onClick={handleUpload}
          className="mt-6 w-full bg-blue-600 text-white py-3 
          rounded-lg hover:bg-blue-700 transition"
        >
          Upload Document
        </button>
      )}
    </div>
  );
};

export default FileUpload;