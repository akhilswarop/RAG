import GuidanceGenerateComponent from "../components/GuidanceGenerator";
import { useLocation } from "react-router-dom";

const GuidanceGenerate = () => {
  const location = useLocation();
  const parsedResume = location.state?.parsedResume;

  return (
    <div className="container mx-auto p-8 bg-white shadow-lg rounded-xl">
      <h1 className="text-3xl font-bold text-center text-blue-800 mb-6">Guidance Generator</h1>
      <div className="p-6 bg-gray-100 rounded-lg shadow-inner">
        <GuidanceGenerateComponent parsedResume={parsedResume} />
      </div>
    </div>
  );
};

export default GuidanceGenerate;
