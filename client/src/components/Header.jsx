import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { Info, LogOut } from "lucide-react"; // Icons for Info & Logout

const Header = () => {
  const navigate = useNavigate(); // Hook for navigation

  // Logout Function
  const handleLogout = () => {
    // Perform logout logic here (e.g., clear authentication, local storage)
    navigate("/"); // Redirect to Home Page
  };

  return (
    <header className="bg-[#0A2540] text-white shadow-md">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        {/* Logo */}
        <Link to="/" className="text-2xl font-bold">
          Career AI
        </Link>

        <div className="flex items-center space-x-4">
          {/* Info Button */}
          <button className="p-2 rounded-full bg-blue-500 hover:bg-blue-600 transition">
            <Info className="text-white" />
          </button>

          {/* Logout Button */}
          <button
            onClick={handleLogout} // Call Logout Function
            className="p-2 rounded-full bg-red-500 hover:bg-red-600 transition"
          >
            <LogOut className="text-white" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
