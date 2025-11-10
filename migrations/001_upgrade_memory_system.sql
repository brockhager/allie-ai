-- Migration: Upgrade Allie Memory System for Safe Learning
-- Date: November 9, 2025
-- Description: Adds status, confidence_score, provenance to facts table,
--              creates learning_queue and learning_log tables for auditable learning pipeline

-- =========================================
-- MIGRATION UP (Apply these changes)
-- =========================================

-- Step 1: Add new columns to facts table
ALTER TABLE facts
ADD COLUMN status ENUM('true','false','not_verified','needs_review','experimental') DEFAULT 'not_verified',
ADD COLUMN confidence_score INT DEFAULT 30,
ADD COLUMN provenance JSON DEFAULT NULL;

-- Step 2: Create learning_queue table
CREATE TABLE learning_queue (
    id INT AUTO_INCREMENT PRIMARY KEY,
    keyword VARCHAR(255) NOT NULL,
    fact TEXT NOT NULL,
    source VARCHAR(255) NOT NULL,
    provenance JSON DEFAULT NULL,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    suggested_action JSON DEFAULT NULL,
    INDEX idx_keyword (keyword),
    INDEX idx_processed (processed),
    INDEX idx_received_at (received_at)
);

-- Step 3: Create learning_log table
CREATE TABLE learning_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fact_id INT NULL,
    old_fact TEXT,
    new_fact TEXT,
    action VARCHAR(50) NOT NULL,
    reviewer VARCHAR(255),
    reason TEXT,
    meta JSON DEFAULT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_fact_id (fact_id),
    INDEX idx_action (action),
    INDEX idx_changed_at (changed_at),
    FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE SET NULL
);

-- Step 4: Create feature_flags table
CREATE TABLE feature_flags (
    id INT AUTO_INCREMENT PRIMARY KEY,
    flag_name VARCHAR(100) UNIQUE NOT NULL,
    enabled BOOLEAN DEFAULT FALSE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_flag_name (flag_name)
);

-- Step 5: Insert default feature flags
INSERT INTO feature_flags (flag_name, enabled, description) VALUES
('AUTO_APPLY_UPDATES', FALSE, 'Automatically apply suggested updates from reconciliation worker'),
('READ_ONLY_MEMORY', FALSE, 'Disable all memory modifications'),
('WRITE_DIRECT', FALSE, 'Allow direct writes to facts table bypassing queue');

-- Step 6: Create roles table
CREATE TABLE roles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    role_name VARCHAR(50) UNIQUE NOT NULL,
    permissions JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_role_name (role_name)
);

-- Step 7: Insert default roles
INSERT INTO roles (role_name, permissions) VALUES
('admin', JSON_OBJECT('facts', JSON_ARRAY('read', 'write', 'delete'), 'queue', JSON_ARRAY('read', 'write'), 'reconcile', JSON_ARRAY('approve', 'reject'), 'flags', JSON_ARRAY('read', 'write'))),
('reviewer', JSON_OBJECT('facts', JSON_ARRAY('read', 'write'), 'queue', JSON_ARRAY('read'), 'reconcile', JSON_ARRAY('approve', 'reject'))),
('viewer', JSON_OBJECT('facts', JSON_ARRAY('read'), 'queue', JSON_ARRAY('read')));

-- Step 8: Create users table (basic auth stub)
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    role_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles(id),
    INDEX idx_username (username)
);

-- Step 9: Insert default admin user
INSERT INTO users (username, role_id) VALUES ('admin', 1);

-- =========================================
-- MIGRATION DOWN (Rollback script)
-- =========================================

-- Uncomment and run these commands to rollback the migration:

-- DROP TABLE IF EXISTS users;
-- DROP TABLE IF EXISTS roles;
-- DROP TABLE IF EXISTS feature_flags;
-- DROP TABLE IF EXISTS learning_log;
-- DROP TABLE IF EXISTS learning_queue;

-- ALTER TABLE facts
-- DROP COLUMN status,
-- DROP COLUMN confidence_score,
-- DROP COLUMN provenance;

-- =========================================
-- VERIFICATION QUERIES
-- =========================================

-- Check migration success:
-- SELECT 'facts table columns' as check_type, COUNT(*) as count FROM information_schema.COLUMNS WHERE TABLE_NAME = 'facts' AND COLUMN_NAME IN ('status', 'confidence_score', 'provenance');
-- SELECT 'learning_queue table' as check_type, COUNT(*) as count FROM information_schema.TABLES WHERE TABLE_NAME = 'learning_queue';
-- SELECT 'learning_log table' as check_type, COUNT(*) as count FROM information_schema.TABLES WHERE TABLE_NAME = 'learning_log';
-- SELECT 'feature_flags table' as check_type, COUNT(*) as count FROM information_schema.TABLES WHERE TABLE_NAME = 'feature_flags';
-- SELECT 'roles table' as check_type, COUNT(*) as count FROM information_schema.TABLES WHERE TABLE_NAME = 'roles';
-- SELECT 'users table' as check_type, COUNT(*) as count FROM information_schema.TABLES WHERE TABLE_NAME = 'users';