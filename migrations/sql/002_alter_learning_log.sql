-- Migration: Add KB audit columns to learning_log
-- Up: add fact_id, reviewer, reason columns
ALTER TABLE learning_log
  ADD COLUMN fact_id INT DEFAULT NULL AFTER id,
  ADD COLUMN reviewer VARCHAR(255) DEFAULT NULL,
  ADD COLUMN reason TEXT DEFAULT NULL;
