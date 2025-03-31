import FileProcessComponent from "../components/FileProcess";

const FileProcessor = () => {
  return (
    <div className="container mx-auto p-8 bg-white shadow-lg rounded-xl">
      <h1 className="text-3xl font-bold text-center text-blue-800 mb-6">File Processor</h1>
      <div className="p-6 bg-gray-100 rounded-lg shadow-inner">
        <FileProcessComponent />
      </div>
    </div>
  );
};

export default FileProcessor;