-- Migration: Create knowledge_base table
-- Up: create table
CREATE TABLE IF NOT EXISTS knowledge_base (
  id INT AUTO_INCREMENT PRIMARY KEY,
  keyword VARCHAR(255) NOT NULL,
  fact TEXT NOT NULL,
  source VARCHAR(255) DEFAULT 'internal',
  status ENUM('true','false','pending','needs_review') DEFAULT 'pending',
  confidence_score INT DEFAULT 90,
  provenance TEXT DEFAULT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_kb_keyword (keyword),
  INDEX idx_kb_status (status),
  INDEX idx_kb_confidence (confidence_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
