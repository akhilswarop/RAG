import React from "react";
import FileUpload from "../components/FileUpload";
import Header from "../components/Header";

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-100">
      <Header />

      <div className="container mx-auto px-4 py-10">
        <h1 className="text-3xl font-bold text-blue-800 mb-6">Dashboard</h1>
        
        <FileUpload />
      </div>
    </div>
  );
};

export default Dashboard;
