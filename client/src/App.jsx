import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import FileProcessor from './pages/FileProcessor';
import GuidanceGenerate from './pages/GuidanceGenerate';
import JobRetrieve from './pages/JobRetrieve';
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/file-process" element={<FileProcessor />} />
        <Route path="/guidance-generate" element={<GuidanceGenerate />} />
        <Route path="/retrieve-jobs" element={<JobRetrieve />} />        
      </Routes>
    </Router>
  );
}

export default App;