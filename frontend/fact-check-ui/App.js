import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import FactCheck from './FactCheck';
import './App.css';

// Tabbed Layout Component
function TabbedLayout() {
  const location = useLocation();
  const [activeTab, setActiveTab] = useState('allie');

  useEffect(() => {
    // Set active tab based on current route
    if (location.pathname === '/fact-check') {
      setActiveTab('fact-check');
    } else {
      setActiveTab('allie');
    }
  }, [location]);

  return (
    <div className="app">
      {/* Top Navigation Tabs */}
      <div className="tab-navigation">
        <Link
          to="/ui"
          className={`tab-link ${activeTab === 'allie' ? 'active' : ''}`}
          onClick={() => setActiveTab('allie')}
        >
          Allie UI
        </Link>
        <Link
          to="/fact-check"
          className={`tab-link ${activeTab === 'fact-check' ? 'active' : ''}`}
          onClick={() => setActiveTab('fact-check')}
        >
          Fact Checking
        </Link>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        <Routes>
          <Route path="/fact-check" element={<FactCheck />} />
          <Route path="/ui" element={<AllieUI />} />
          <Route path="/" element={<AllieUI />} />
        </Routes>
      </div>
    </div>
  );
}

// Allie UI Component (iframe wrapper)
function AllieUI() {
  return (
    <div className="allie-ui-container">
      <iframe
        src="/chat"
        title="Allie Chat Interface"
        className="allie-iframe"
        frameBorder="0"
      />
    </div>
  );
}

export default TabbedLayout;