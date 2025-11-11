-- Migration rollback: Remove KB audit columns from learning_log
-- Down: drop fact_id, reviewer, reason columns
ALTER TABLE learning_log
  DROP COLUMN fact_id,
  DROP COLUMN reviewer,
  DROP COLUMN reason;
