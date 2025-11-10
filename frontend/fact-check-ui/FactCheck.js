import React, { useState, useEffect } from 'react';
import './FactCheck.css';

function FactCheck() {
  const [facts, setFacts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedFact, setSelectedFact] = useState(null);
  const [filters, setFilters] = useState({
    status: '',
    category: '',
    minConfidence: ''
  });
  const [featureFlags, setFeatureFlags] = useState({
    READ_ONLY_MEMORY: false,
    AUTO_APPLY_UPDATES: false
  });

  // Load facts on component mount
  useEffect(() => {
    loadFacts();
    loadFeatureFlags();
  }, [filters]);

  const loadFacts = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filters.status) params.append('status', filters.status);
      if (filters.category) params.append('category', filters.category);
      if (filters.minConfidence) params.append('min_confidence', filters.minConfidence);

      const response = await fetch(`/api/facts?${params}`);
      const data = await response.json();

      if (data.status === 'success') {
        setFacts(data.facts);
      }
    } catch (error) {
      console.error('Error loading facts:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadFeatureFlags = async () => {
    try {
      const response = await fetch('/api/feature_flags');
      const data = await response.json();
      if (data.status === 'success') {
        setFeatureFlags(data.flags);
      }
    } catch (error) {
      console.error('Error loading feature flags:', error);
    }
  };

  const updateFactStatus = async (factId, status, confidenceScore, reviewer, reason) => {
    try {
      const response = await fetch(`/api/facts/${factId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          status,
          confidence_score: confidenceScore,
          reviewer,
          reason
        })
      });

      const data = await response.json();
      if (data.status === 'success') {
        // Refresh facts list
        loadFacts();
        // Clear selection if this fact was selected
        if (selectedFact && selectedFact.id === factId) {
          setSelectedFact(null);
        }
      }
    } catch (error) {
      console.error('Error updating fact:', error);
    }
  };

  const bulkUpdateFacts = async (updates) => {
    try {
      const response = await fetch('/api/facts/bulk-update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ updates })
      });

      const data = await response.json();
      if (data.status === 'success') {
        loadFacts();
      }
    } catch (error) {
      console.error('Error bulk updating facts:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'true': return 'status-true';
      case 'false': return 'status-false';
      case 'not_verified': return 'status-not-verified';
      case 'needs_review': return 'status-needs-review';
      case 'experimental': return 'status-experimental';
      default: return 'status-unknown';
    }
  };

  const getConfidenceColor = (score) => {
    if (score >= 80) return 'confidence-high';
    if (score >= 60) return 'confidence-medium';
    if (score >= 40) return 'confidence-low';
    return 'confidence-very-low';
  };

  return (
    <div className="fact-check-container">
      {/* Header */}
      <div className="fact-check-header">
        <h1>Fact Checking Dashboard</h1>
        <div className="system-status">
          {featureFlags.READ_ONLY_MEMORY && (
            <span className="status-badge read-only">READ ONLY MODE</span>
          )}
          {featureFlags.AUTO_APPLY_UPDATES && (
            <span className="status-badge auto-apply">AUTO APPLY ENABLED</span>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="filters-section">
        <div className="filter-group">
          <label>Status:</label>
          <select
            value={filters.status}
            onChange={(e) => setFilters({...filters, status: e.target.value})}
          >
            <option value="">All</option>
            <option value="true">True</option>
            <option value="false">False</option>
            <option value="not_verified">Not Verified</option>
            <option value="needs_review">Needs Review</option>
            <option value="experimental">Experimental</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Min Confidence:</label>
          <input
            type="number"
            min="0"
            max="100"
            value={filters.minConfidence}
            onChange={(e) => setFilters({...filters, minConfidence: e.target.value})}
            placeholder="0-100"
          />
        </div>

        <button onClick={loadFacts} className="refresh-btn">Refresh</button>
      </div>

      {/* Main Content */}
      <div className="fact-check-main">
        {/* Facts Table */}
        <div className="facts-table-container">
          {loading ? (
            <div className="loading">Loading facts...</div>
          ) : (
            <table className="facts-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Keyword</th>
                  <th>Fact Preview</th>
                  <th>Source</th>
                  <th>Status</th>
                  <th>Confidence</th>
                  <th>Updated</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {facts.map(fact => (
                  <tr
                    key={fact.id}
                    className={selectedFact && selectedFact.id === fact.id ? 'selected' : ''}
                    onClick={() => setSelectedFact(fact)}
                  >
                    <td>{fact.id}</td>
                    <td>{fact.keyword}</td>
                    <td>{fact.fact.substring(0, 100)}...</td>
                    <td>{fact.source}</td>
                    <td>
                      <span className={`status-badge ${getStatusColor(fact.status)}`}>
                        {fact.status}
                      </span>
                    </td>
                    <td>
                      <span className={`confidence-score ${getConfidenceColor(fact.confidence_score)}`}>
                        {fact.confidence_score}%
                      </span>
                    </td>
                    <td>{new Date(fact.updated_at).toLocaleDateString()}</td>
                    <td>
                      <div className="action-buttons">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            updateFactStatus(fact.id, 'true', 85, 'reviewer', 'Manual verification');
                          }}
                          disabled={featureFlags.READ_ONLY_MEMORY}
                          className="action-btn approve"
                        >
                          ✓ True
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            updateFactStatus(fact.id, 'false', 5, 'reviewer', 'Manual verification');
                          }}
                          disabled={featureFlags.READ_ONLY_MEMORY}
                          className="action-btn reject"
                        >
                          ✗ False
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            updateFactStatus(fact.id, 'needs_review', fact.confidence_score, 'reviewer', 'Needs further review');
                          }}
                          disabled={featureFlags.READ_ONLY_MEMORY}
                          className="action-btn review"
                        >
                          ? Review
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Side Panel */}
        {selectedFact && (
          <div className="fact-details-panel">
            <div className="panel-header">
              <h3>Fact Details</h3>
              <button onClick={() => setSelectedFact(null)} className="close-btn">×</button>
            </div>

            <div className="panel-content">
              <div className="fact-info">
                <h4>Fact #{selectedFact.id}</h4>
                <p><strong>Keyword:</strong> {selectedFact.keyword}</p>
                <p><strong>Fact:</strong> {selectedFact.fact}</p>
                <p><strong>Source:</strong> {selectedFact.source}</p>
                <p><strong>Category:</strong> {selectedFact.category}</p>
                <p><strong>Status:</strong>
                  <span className={`status-badge ${getStatusColor(selectedFact.status)}`}>
                    {selectedFact.status}
                  </span>
                </p>
                <p><strong>Confidence:</strong> {selectedFact.confidence_score}%</p>
                <p><strong>Created:</strong> {new Date(selectedFact.created_at).toLocaleString()}</p>
                <p><strong>Updated:</strong> {new Date(selectedFact.updated_at).toLocaleString()}</p>
              </div>

              {/* Provenance */}
              {selectedFact.provenance && (
                <div className="provenance-section">
                  <h4>Provenance</h4>
                  <pre className="provenance-json">
                    {JSON.stringify(selectedFact.provenance, null, 2)}
                  </pre>
                </div>
              )}

              {/* Learning Log Placeholder */}
              <div className="learning-log-section">
                <h4>Change History</h4>
                <p>Learning log integration coming soon...</p>
              </div>

              {/* External Sources Placeholder */}
              <div className="external-sources-section">
                <h4>External Source Comparison</h4>
                <p>External source comparison coming soon...</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default FactCheck;