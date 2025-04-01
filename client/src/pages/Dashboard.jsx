import { useState, useEffect } from "react";
import BeatLoader from "react-spinners/BeatLoader";
import FileUpload from "../components/FileUpload";
import Header from "../components/Header";



const Dashboard = () => {
  const [isLoading, setIsLoading] = useState(true);
  let [color, setColor] = useState("#ffffff");

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/healthcheck");
        if (response.ok) {
          setIsLoading(false);
        } else {
          throw new Error("Backend not ready");
        }
      } catch (error) {
        setTimeout(checkBackend, 2000); // Retry every 2 seconds
      }
    };

    checkBackend();
  }, []);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
       <BeatLoader />
       <p className = "pl-4"> Starting server. Please wait.</p>
      </div>    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />

      <div className="container mx-auto px-4 py-10">
        
        
        <FileUpload />
      </div>
    </div>
  );
};

export default Dashboard;
