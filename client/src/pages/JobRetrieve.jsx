import JobRetriever from "../components/JobRetriever";
import { useLocation } from "react-router-dom";
import { useNavigate } from "react-router-dom";


const JobRetrieve = () => {
  const location = useLocation();
  const jobs = location.state?.jobs;

  return (
    <div className="container mx-auto p-8 bg-white shadow-lg rounded-xl">
      <div className="p-6 bg-gray-100 rounded-lg shadow-inner">
        <JobRetriever jobs={jobs}/>
      </div>
    </div>
  );
};

export default JobRetrieve;
