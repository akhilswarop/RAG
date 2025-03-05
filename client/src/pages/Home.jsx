import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { 
  BrainIcon, 
  TrendingUpIcon,
  BookOpenIcon,
  CodeIcon,
  SearchIcon,
  ArrowRightIcon
} from "lucide-react";

const LandingPage = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");

  const features = [
    { icon: BrainIcon, title: "AI-Powered Insights", description: "Advanced analysis of your professional potential using LLMs and RAG." },
    { icon: TrendingUpIcon, title: "Career Trajectory", description: "Personalized growth and opportunity mapping with AI-driven predictions." },
    { icon: BookOpenIcon, title: "Continuous Learning", description: "Tailored skill development recommendations to keep you ahead." },
    { icon: CodeIcon, title: "AI-Augmented Research", description: "Get insights backed by Retrieval-Augmented Generation (RAG) for accurate decision-making." },
    { icon: SearchIcon, title: "Intelligent Job Matching", description: "Find the best opportunities with AI-driven job matching." }
  ];

  const handleEmailSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://127.0.0.1:5000/store-email", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      });

      const result = await response.json();
      console.log("Server Response:", result);

      if (response.ok) {
        navigate("/dashboard");  // Navigate only if request is successful
      } else {
        console.error("Failed to store email:", result);
      }
    } catch (error) {
      console.error("Error storing email:", error);
    }
  };

  return (
    <div className="w-full h-screen flex items-center justify-center bg-gradient-to-br from-blue-100 via-purple-200 to-indigo-300 p-4">
      <div className="container mx-auto max-w-7xl">
        <div className="grid lg:grid-cols-2 gap-8 h-full">
          {/* Left Content */}
          <div className="flex flex-col justify-center space-y-6">
            <h1 className="text-5xl font-bold text-gray-900 leading-tight">
              Transform Your 
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
                Professional Journey
              </span>
            </h1>
            <p className="text-xl text-gray-600 pr-8">
              Unlock personalized career insights, explore opportunities, 
              and make data-driven professional decisions with our 
              advanced AI guidance platform.
            </p>

            <form onSubmit={handleEmailSubmit} className="flex flex-col sm:flex-row gap-4 max-w-xl">
              <input 
                type="email"
                placeholder="Enter your email to get started"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="flex-1 px-6 py-4 rounded-full bg-white 
                text-gray-900 border border-gray-300 
                focus:outline-none focus:ring-2 focus:ring-blue-500 
                focus:border-transparent transition duration-300"
              />
              <button 
                type="submit"
                className="bg-blue-600 text-white px-8 py-4 rounded-full 
                font-bold hover:bg-blue-700 transition duration-300 
                flex items-center justify-center gap-2 group"
              >
                Get Started
                <ArrowRightIcon className="transform group-hover:translate-x-1 transition" />
              </button>
            </form>
          </div>

          {/* Right Side - Features */}
          <div className="grid grid-cols-1 gap-4">
            {features.map((feature, index) => (
              <div key={index} className="bg-white rounded-2xl p-5 flex items-center space-x-4 shadow-md hover:shadow-lg transition duration-300 group">
                <div className="bg-blue-100 p-3 rounded-xl">
                  <feature.icon className="w-8 h-8 text-blue-600 group-hover:rotate-6 transition" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-1">{feature.title}</h3>
                  <p className="text-gray-600 text-sm">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
