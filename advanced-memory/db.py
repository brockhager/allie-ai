#!/usr/bin/env python3
"""
Advanced MySQL Database Connector for Allie's Memory System

Provides persistent storage with self-correcting capabilities and learning history tracking.
"""

import mysql.connector
import json
from mysql.connector import Error
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AllieMemoryDB:
    """MySQL-based memory system with self-correction and learning tracking"""
    
    def __init__(self, host='localhost', database='allie_memory', user='allie', password='StrongPassword123!'):
        """
        Initialize database connection
        
        Args:
            host: MySQL server host
            database: Database name
            user: MySQL user
            password: MySQL password
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self._connect()
        self._initialize_tables()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                autocommit=True
            )
            if self.connection.is_connected():
                logger.info(f"Successfully connected to MySQL database: {self.database}")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise
    
    def _initialize_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.connection or not self.connection.is_connected():
            self._connect()
        
        cursor = self.connection.cursor()
        
        # Check if facts table needs updating
        cursor.execute("SHOW COLUMNS FROM facts")
        columns = {row[0] for row in cursor.fetchall()}
        
        # Add new columns if they don't exist
        if 'confidence' not in columns:
            cursor.execute("ALTER TABLE facts ADD COLUMN confidence FLOAT DEFAULT 0.8")
            logger.info("Added confidence column to facts table")
        
        if 'category' not in columns:
            cursor.execute("ALTER TABLE facts ADD COLUMN category VARCHAR(100) DEFAULT 'general'")
            logger.info("Added category column to facts table")
        
        if 'created_at' not in columns:
            cursor.execute("ALTER TABLE facts ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            logger.info("Added created_at column to facts table")
        
        if 'status' not in columns:
            cursor.execute("ALTER TABLE facts ADD COLUMN status ENUM('true', 'false', 'not_verified', 'needs_review', 'experimental') DEFAULT 'not_verified'")
            logger.info("Added status column to facts table")
        
        if 'confidence_score' not in columns:
            cursor.execute("ALTER TABLE facts ADD COLUMN confidence_score INT DEFAULT 50")
            logger.info("Added confidence_score column to facts table")
        
        # Add indexes
        try:
            cursor.execute("CREATE INDEX idx_keyword ON facts(keyword)")
        except:
            pass  # Index might already exist
        
        try:
            cursor.execute("CREATE INDEX idx_category ON facts(category)")
        except:
            pass
        
        try:
            cursor.execute("CREATE INDEX idx_updated ON facts(updated_at)")
        except:
            pass
        
        try:
            cursor.execute("CREATE INDEX idx_status ON facts(status)")
        except:
            pass
        
        try:
            cursor.execute("CREATE INDEX idx_confidence_score ON facts(confidence_score)")
        except:
            pass
        
        # Learning log table for tracking changes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                fact_id INT DEFAULT NULL,
                keyword VARCHAR(255) NOT NULL,
                old_fact TEXT,
                new_fact TEXT NOT NULL,
                source VARCHAR(255) NOT NULL,
                confidence FLOAT,
                change_type ENUM('add', 'update', 'delete', 'validate') DEFAULT 'add',
                reviewer VARCHAR(255) DEFAULT NULL,
                reason TEXT DEFAULT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_keyword (keyword),
                INDEX idx_changed_at (changed_at)
            )
        """)
        
        # Learning queue for batch processing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_queue (
                id INT AUTO_INCREMENT PRIMARY KEY,
                keyword VARCHAR(255) NOT NULL,
                fact TEXT NOT NULL,
                source VARCHAR(255) NOT NULL,
                confidence FLOAT DEFAULT 0.5,
                category VARCHAR(100) DEFAULT 'general',
                status ENUM('pending', 'validated', 'rejected', 'processed') DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP NULL,
                INDEX idx_status (status),
                INDEX idx_keyword (keyword)
            )
        """)
        
        # Fact clusters for grouping related facts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_clusters (
                id INT AUTO_INCREMENT PRIMARY KEY,
                cluster_name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_cluster_name (cluster_name)
            )
        """)
        
        # Cluster memberships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_memberships (
                id INT AUTO_INCREMENT PRIMARY KEY,
                cluster_id INT NOT NULL,
                fact_id INT NOT NULL,
                relevance_score FLOAT DEFAULT 1.0,
                FOREIGN KEY (cluster_id) REFERENCES fact_clusters(id) ON DELETE CASCADE,
                FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE,
                UNIQUE KEY unique_membership (cluster_id, fact_id)
            )
        """)
        
        cursor.close()
        logger.info("Database tables initialized successfully")

        # Create knowledge_base table if it doesn't exist
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INT AUTO_INCREMENT PRIMARY KEY,
                keyword VARCHAR(255) NOT NULL,
                fact TEXT NOT NULL,
                source VARCHAR(255) DEFAULT 'internal',
                status ENUM('true','false','not_verified','needs_review','experimental') DEFAULT 'true',
                confidence_score INT DEFAULT 90,
                provenance JSON DEFAULT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_kb_keyword (keyword),
                INDEX idx_kb_status (status),
                INDEX idx_kb_confidence (confidence_score)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.close()
    
    def add_fact(self, keyword: str, fact: str, source: str, 
                 confidence: float = 0.8, category: str = 'general',
                 status: str = 'not_verified', confidence_score: int = 50) -> Dict:
        """
        Add a new fact to memory
        
        Args:
            keyword: Main keyword/topic for the fact
            fact: The factual information
            source: Where the fact came from
            confidence: Confidence score (0.0-1.0)
            category: Category/domain of the fact
            status: Verification status ('true', 'false', 'not_verified', 'needs_review', 'experimental')
            confidence_score: Confidence score (0-100)
            
        Returns:
            Dict with status and fact_id
        """
        try:
            cursor = self.connection.cursor()
            
            # Check if fact already exists
            cursor.execute("SELECT id, fact FROM facts WHERE keyword = %s", (keyword,))
            existing = cursor.fetchone()
            
            if existing:
                # Fact exists, decide whether to update
                existing_id, existing_fact = existing
                if existing_fact != fact:
                    # Update if different
                    cursor.close()
                    return self.update_fact(keyword, fact, source, confidence)
                else:
                    cursor.close()
                    return {"status": "exists", "fact_id": existing_id, "message": "Fact already exists"}
            
            # Insert new fact
            cursor.execute("""
                INSERT INTO facts (keyword, fact, source, confidence, category, status, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (keyword, fact, source, confidence, category, status, confidence_score))
            
            fact_id = cursor.lastrowid
            
            # Log to learning history
            cursor.execute("""
                INSERT INTO learning_log (keyword, old_fact, new_fact, source, confidence, change_type)
                VALUES (%s, NULL, %s, %s, %s, 'add')
            """, (keyword, fact, source, confidence))
            
            cursor.close()
            logger.info(f"Added fact for keyword '{keyword}' from {source}")
            
            return {"status": "added", "fact_id": fact_id, "keyword": keyword}
            
        except Error as e:
            logger.error(f"Error adding fact: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_fact(self, keyword: str) -> Optional[Dict]:
        """
        Retrieve a fact by keyword
        
        Args:
            keyword: The keyword to search for
            
        Returns:
            Dict with fact details or None if not found
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, keyword, fact, source, confidence, category, status, confidence_score,
                       created_at, updated_at
                FROM facts
                WHERE keyword = %s
                ORDER BY confidence DESC, updated_at DESC
                LIMIT 1
            """, (keyword,))
            
            result = cursor.fetchone()
            cursor.close()
            
            return result
            
        except Error as e:
            logger.error(f"Error retrieving fact: {e}")
            return None
    
    def search_facts(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for facts matching a query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching facts
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Search in both keyword and fact content
            search_term = f"%{query}%"
            cursor.execute("""
                SELECT id, keyword, fact, source, confidence, category, status, confidence_score,
                       created_at, updated_at
                FROM facts
                WHERE keyword LIKE %s OR fact LIKE %s
                ORDER BY confidence DESC, updated_at DESC
                LIMIT %s
            """, (search_term, search_term, limit))
            
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error searching facts: {e}")
            return []
    
    def update_fact(self, keyword: str, new_fact: str, source: str,
                    confidence: float = 0.8, status: str = None, confidence_score: int = None) -> Dict:
        """
        Update an existing fact
        
        Args:
            keyword: Keyword to update
            new_fact: New fact content
            source: Source of the update
            confidence: Confidence in the new fact
            
        Returns:
            Dict with status
        """
        try:
            cursor = self.connection.cursor()
            
            # Get old fact for logging
            cursor.execute("SELECT fact FROM facts WHERE keyword = %s", (keyword,))
            result = cursor.fetchone()
            old_fact = result[0] if result else None
            
            if old_fact:
                # Update existing fact
                cursor.execute("""
                    UPDATE facts
                    SET fact = %s, source = %s, confidence = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE keyword = %s
                """, (new_fact, source, confidence, keyword))
                
                # Update status and confidence_score if provided
                if status is not None:
                    cursor.execute("UPDATE facts SET status = %s WHERE keyword = %s", (status, keyword))
                if confidence_score is not None:
                    cursor.execute("UPDATE facts SET confidence_score = %s WHERE keyword = %s", (confidence_score, keyword))
                
                # Log the update
                cursor.execute("""
                    INSERT INTO learning_log (keyword, old_fact, new_fact, source, confidence, change_type)
                    VALUES (%s, %s, %s, %s, %s, 'update')
                """, (keyword, old_fact, new_fact, source, confidence))
                
                cursor.close()
                logger.info(f"Updated fact for keyword '{keyword}' from {source}")
                
                return {"status": "updated", "keyword": keyword, "old_fact": old_fact, "new_fact": new_fact}
            else:
                cursor.close()
                # Fact doesn't exist, add it
                return self.add_fact(keyword, new_fact, source, confidence)
                
        except Error as e:
            logger.error(f"Error updating fact: {e}")
            return {"status": "error", "message": str(e)}
    
    def delete_fact(self, keyword: str) -> Dict:
        """
        Delete a fact
        
        Args:
            keyword: Keyword to delete
            
        Returns:
            Dict with status
        """
        try:
            cursor = self.connection.cursor()
            
            # Get fact for logging
            cursor.execute("SELECT fact, source FROM facts WHERE keyword = %s", (keyword,))
            result = cursor.fetchone()
            
            if result:
                old_fact, old_source = result
                
                # Delete the fact
                cursor.execute("DELETE FROM facts WHERE keyword = %s", (keyword,))
                
                # Log the deletion (use empty string instead of NULL for new_fact)
                cursor.execute("""
                    INSERT INTO learning_log (keyword, old_fact, new_fact, source, change_type)
                    VALUES (%s, %s, '', %s, 'delete')
                """, (keyword, old_fact, old_source))
                
                cursor.close()
                logger.info(f"Deleted fact for keyword '{keyword}'")
                
                return {"status": "deleted", "keyword": keyword}
            else:
                cursor.close()
                return {"status": "not_found", "keyword": keyword}
                
        except Error as e:
            logger.error(f"Error deleting fact: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------ Knowledge Base Methods ------------------
    def add_kb_fact(self, keyword: str, fact: str, source: str = 'internal', confidence_score: int = 90, provenance: Optional[Dict] = None, status: str = 'true') -> Dict:
        """Add a fact to the curated Knowledge Base and log the change"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT id, fact FROM knowledge_base WHERE keyword = %s", (keyword,))
            existing = cursor.fetchone()
            if existing:
                # If different, update
                    existing_id, existing_fact = existing
                    if existing_fact != fact:
                        cursor.execute("""
                            UPDATE knowledge_base SET fact=%s, source=%s, confidence_score=%s, provenance=%s, status=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s
                        """, (fact, source, confidence_score, json.dumps(provenance) if provenance else None, status, existing_id))
                        # Log
                        cursor.execute("""
                            INSERT INTO learning_log (fact_id, keyword, old_fact, new_fact, source, confidence, change_type)
                            VALUES (%s, %s, %s, %s, %s, %s, 'update')
                        """, (existing_id, keyword, existing_fact, fact, source, confidence_score))
                        cursor.close()
                        return {"status": "updated", "fact_id": existing_id}
                    else:
                        cursor.close()
                        return {"status": "exists", "fact_id": existing[0]}

            cursor.execute("""
                INSERT INTO knowledge_base (keyword, fact, source, status, confidence_score, provenance)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (keyword, fact, source, status, confidence_score, json.dumps(provenance) if provenance else None))
            kb_id = cursor.lastrowid
            cursor.execute("""
                INSERT INTO learning_log (fact_id, keyword, old_fact, new_fact, source, confidence, change_type)
                VALUES (%s, %s, %s, %s, %s, %s, 'add')
            """, (kb_id, keyword, None, fact, source, confidence_score))
            cursor.close()
            return {"status": "added", "fact_id": kb_id}
        except Error as e:
            logger.error(f"Error adding KB fact: {e}")
            return {"status": "error", "message": str(e)}

    def get_kb_fact(self, keyword: str) -> Optional[Dict]:
        """Retrieve a KB fact by keyword (exact match)"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, keyword, fact, source, status, confidence_score, provenance, updated_at
                FROM knowledge_base
                WHERE keyword = %s
                ORDER BY updated_at DESC
                LIMIT 1
            """, (keyword,))
            res = cursor.fetchone()
            cursor.close()
            return res
        except Error as e:
            logger.error(f"Error retrieving KB fact: {e}")
            return None

    def get_all_kb_facts(self, status: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Return all KB facts, optionally filtered by status, with pagination."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            if status:
                cursor.execute("""
                    SELECT id, keyword, fact, source, status, confidence_score, provenance, updated_at
                    FROM knowledge_base
                    WHERE status = %s
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                """, (status, limit, offset))
            else:
                cursor.execute("""
                    SELECT id, keyword, fact, source, status, confidence_score, provenance, updated_at
                    FROM knowledge_base
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except Error as e:
            logger.error(f"Error listing KB facts: {e}")
            return []

    def update_kb_fact(self, kb_id: int, new_fact: str = None, status: str = None, confidence_score: int = None, reviewer: str = None, reason: str = None) -> Dict:
        """Update a KB fact by id and log the change"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT fact, keyword FROM knowledge_base WHERE id = %s", (kb_id,))
            existing = cursor.fetchone()
            if not existing:
                cursor.close()
                return {"status": "not_found", "fact_id": kb_id}
            old_fact, keyword = existing
            updates = []
            params = []
            if new_fact is not None:
                updates.append("fact = %s")
                params.append(new_fact)
            if status is not None:
                updates.append("status = %s")
                params.append(status)
            if confidence_score is not None:
                updates.append("confidence_score = %s")
                params.append(confidence_score)
            if updates:
                params.append(kb_id)
                sql = f"UPDATE knowledge_base SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
                cursor.execute(sql, tuple(params))

            # Log
            cursor.execute("""
                INSERT INTO learning_log (fact_id, keyword, old_fact, new_fact, source, confidence, change_type, reviewer, reason)
                VALUES (%s, %s, %s, %s, %s, %s, 'update', %s, %s)
            """, (kb_id, keyword, old_fact, new_fact or old_fact, 'kb_admin', confidence_score if confidence_score is not None else None, reviewer, reason))

            cursor.close()
            return {"status": "updated", "fact_id": kb_id}
        except Error as e:
            logger.error(f"Error updating KB fact: {e}")
            return {"status": "error", "message": str(e)}

    def delete_kb_fact(self, kb_id: int, reviewer: str = None, reason: str = None) -> Dict:
        """Delete a KB fact and log the deletion"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT fact, keyword FROM knowledge_base WHERE id = %s", (kb_id,))
            existing = cursor.fetchone()
            if not existing:
                cursor.close()
                return {"status": "not_found", "fact_id": kb_id}
            old_fact, keyword = existing
            cursor.execute("DELETE FROM knowledge_base WHERE id = %s", (kb_id,))
            cursor.execute("""
                INSERT INTO learning_log (fact_id, keyword, old_fact, new_fact, source, change_type, reviewer, reason)
                VALUES (%s, %s, %s, '', %s, 'delete', %s, %s)
            """, (kb_id, keyword, old_fact, 'kb_admin', reviewer, reason))
            cursor.close()
            return {"status": "deleted", "fact_id": kb_id}
        except Error as e:
            logger.error(f"Error deleting KB fact: {e}")
            return {"status": "error", "message": str(e)}

    def add_learning_queue(self, keyword: str, fact: str, source: str, confidence: float = 0.3, category: str = 'general', provenance: Optional[Dict] = None) -> Dict:
        """Insert an item into learning_queue table"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO learning_queue (keyword, fact, source, confidence, category)
                VALUES (%s, %s, %s, %s, %s)
            """, (keyword, fact, source, confidence, category))
            qid = cursor.lastrowid
            cursor.execute("""
                INSERT INTO learning_log (keyword, old_fact, new_fact, source, confidence, change_type)
                VALUES (%s, %s, %s, %s, %s, 'add')
            """, (keyword, '', fact, source, confidence))
            cursor.close()
            return {"status": "queued", "queue_id": qid}
        except Error as e:
            logger.error(f"Error adding to learning queue: {e}")
            return {"status": "error", "message": str(e)}
    
    def timeline(self, limit: int = 100, include_deleted: bool = False) -> List[Dict]:
        """
        Get timeline of facts ordered by update time
        
        Args:
            limit: Maximum number of facts to return
            include_deleted: Whether to include deleted facts from log
            
        Returns:
            List of facts ordered by updated_at
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if include_deleted:
                # Include all changes from learning log
                cursor.execute("""
                    SELECT keyword, new_fact as fact, source, confidence, 
                           change_type, changed_at as updated_at
                    FROM learning_log
                    ORDER BY changed_at DESC
                    LIMIT %s
                """, (limit,))
            else:
                # Only current facts
                cursor.execute("""
                    SELECT id, keyword, fact, source, confidence, category, status, confidence_score,
                           created_at, updated_at
                    FROM facts
                    ORDER BY updated_at DESC
                    LIMIT %s
                """, (limit,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error getting timeline: {e}")
            return []
    
    def get_all_facts(self, status_filter: str = None, category_filter: str = None, 
                     limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Get all facts with optional filtering and pagination
        
        Args:
            status_filter: Filter by status ('true', 'false', 'not_verified', etc.)
            category_filter: Filter by category
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of facts
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
                SELECT id, keyword, fact, source, confidence, category, status, confidence_score,
                       created_at, updated_at
                FROM facts
                WHERE 1=1
            """
            params = []
            
            if status_filter:
                query += " AND status = %s"
                params.append(status_filter)
            
            if category_filter:
                query += " AND category = %s"
                params.append(category_filter)
            
            query += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error getting all facts: {e}")
            return []
    
    def update_fact_status(self, fact_id: int, status: str, confidence_score: int = None) -> Dict:
        """
        Update a fact's verification status and confidence score
        
        Args:
            fact_id: ID of the fact to update
            status: New status ('true', 'false', 'not_verified', 'needs_review', 'experimental')
            confidence_score: New confidence score (0-100)
            
        Returns:
            Dict with status
        """
        try:
            cursor = self.connection.cursor()
            
            # Get current fact for logging
            cursor.execute("SELECT keyword, fact, status, confidence_score FROM facts WHERE id = %s", (fact_id,))
            result = cursor.fetchone()
            
            if not result:
                return {"status": "not_found", "fact_id": fact_id}
            
            old_keyword, old_fact, old_status, old_confidence_score = result
            
            # Update the fact
            update_fields = ["status = %s"]
            params = [status]
            
            if confidence_score is not None:
                update_fields.append("confidence_score = %s")
                params.append(confidence_score)
            
            params.append(fact_id)
            update_query = f"UPDATE facts SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
            
            cursor.execute(update_query, params)
            
            # Log the change
            cursor.execute("""
                INSERT INTO learning_log (keyword, old_fact, new_fact, source, confidence, change_type)
                VALUES (%s, %s, %s, %s, %s, 'validate')
            """, (old_keyword, f"Status: {old_status} -> {status}, Confidence: {old_confidence_score} -> {confidence_score or old_confidence_score}", 
                  old_fact, "manual_verification", confidence_score/100.0 if confidence_score else old_confidence_score/100.0))
            
            cursor.close()
            logger.info(f"Updated fact {fact_id} status to {status}")
            
            return {"status": "updated", "fact_id": fact_id, "new_status": status, "new_confidence_score": confidence_score}
            
        except Error as e:
            logger.error(f"Error updating fact status: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_fact_by_id(self, fact_id: int) -> Optional[Dict]:
        """
        Get a fact by its ID
        
        Args:
            fact_id: The fact ID
            
        Returns:
            Dict with fact details or None if not found
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, keyword, fact, source, confidence, category, status, confidence_score,
                       created_at, updated_at
                FROM facts
                WHERE id = %s
            """, (fact_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            return result
            
        except Error as e:
            logger.error(f"Error retrieving fact by ID: {e}")
            return None
    
    def add_to_learning_queue(self, keyword: str, fact: str, source: str,
                              confidence: float = 0.5, category: str = 'general') -> Dict:
        """
        Add a fact to the learning queue for validation
        
        Args:
            keyword: Keyword for the fact
            fact: The fact content
            source: Source of the fact
            confidence: Initial confidence score
            category: Fact category
            
        Returns:
            Dict with status and queue_id
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO learning_queue (keyword, fact, source, confidence, category, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
            """, (keyword, fact, source, confidence, category))
            
            queue_id = cursor.lastrowid
            cursor.close()
            
            logger.info(f"Added fact to learning queue: {keyword}")
            return {"status": "queued", "queue_id": queue_id}
            
        except Error as e:
            logger.error(f"Error adding to learning queue: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_learning_queue(self, status: str = 'pending', limit: int = 100) -> List[Dict]:
        """
        Get items from learning queue
        
        Args:
            status: Queue status to filter by
            limit: Maximum items to return
            
        Returns:
            List of queued facts
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, keyword, fact, source, confidence, category, status, created_at
                FROM learning_queue
                WHERE status = %s
                ORDER BY created_at ASC
                LIMIT %s
            """, (status, limit))
            
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error getting learning queue: {e}")
            return []
    
    def process_queue_item(self, queue_id: int, action: str, confidence: float = None) -> Dict:
        """
        Process a queued fact (validate, reject, or add to memory)
        
        Args:
            queue_id: ID of the queued item
            action: 'validate', 'reject', or 'process'
            confidence: Updated confidence score
            
        Returns:
            Dict with status
        """
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Get the queued item
            cursor.execute("SELECT * FROM learning_queue WHERE id = %s", (queue_id,))
            item = cursor.fetchone()
            
            if not item:
                return {"status": "not_found"}
            
            if action == 'validate':
                # Mark as validated
                cursor.execute("""
                    UPDATE learning_queue
                    SET status = 'validated', confidence = COALESCE(%s, confidence)
                    WHERE id = %s
                """, (confidence, queue_id))
                
            elif action == 'reject':
                # Mark as rejected
                cursor.execute("""
                    UPDATE learning_queue
                    SET status = 'rejected', processed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (queue_id,))
                
            elif action == 'process':
                # Add to facts table
                final_confidence = confidence if confidence else item['confidence']
                self.add_fact(
                    item['keyword'],
                    item['fact'],
                    item['source'],
                    final_confidence,
                    item['category']
                )
                
                # Mark as processed
                cursor.execute("""
                    UPDATE learning_queue
                    SET status = 'processed', processed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (queue_id,))
            
            cursor.close()
            logger.info(f"Processed queue item {queue_id}: {action}")
            
            return {"status": "success", "action": action, "queue_id": queue_id}
            
        except Error as e:
            logger.error(f"Error processing queue item: {e}")
            return {"status": "error", "message": str(e)}
    
    def create_cluster(self, cluster_name: str, description: str = "") -> Dict:
        """Create a fact cluster for grouping related facts"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO fact_clusters (cluster_name, description)
                VALUES (%s, %s)
            """, (cluster_name, description))
            
            cluster_id = cursor.lastrowid
            cursor.close()
            
            return {"status": "created", "cluster_id": cluster_id, "cluster_name": cluster_name}
            
        except Error as e:
            if e.errno == 1062:  # Duplicate entry
                return {"status": "exists", "cluster_name": cluster_name}
            logger.error(f"Error creating cluster: {e}")
            return {"status": "error", "message": str(e)}
    
    def add_to_cluster(self, cluster_name: str, fact_id: int, relevance_score: float = 1.0) -> Dict:
        """Add a fact to a cluster"""
        try:
            cursor = self.connection.cursor()
            
            # Get cluster ID
            cursor.execute("SELECT id FROM fact_clusters WHERE cluster_name = %s", (cluster_name,))
            result = cursor.fetchone()
            
            if not result:
                # Create cluster if it doesn't exist
                cluster_result = self.create_cluster(cluster_name)
                if cluster_result['status'] == 'error':
                    return cluster_result
                cluster_id = cluster_result['cluster_id']
            else:
                cluster_id = result[0]
            
            # Add to cluster
            cursor.execute("""
                INSERT INTO cluster_memberships (cluster_id, fact_id, relevance_score)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE relevance_score = %s
            """, (cluster_id, fact_id, relevance_score, relevance_score))
            
            cursor.close()
            return {"status": "added", "cluster_name": cluster_name, "fact_id": fact_id}
            
        except Error as e:
            logger.error(f"Error adding to cluster: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_cluster_facts(self, cluster_name: str) -> List[Dict]:
        """Get all facts in a cluster"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT f.id, f.keyword, f.fact, f.source, f.confidence, f.category,
                       cm.relevance_score, f.updated_at
                FROM facts f
                JOIN cluster_memberships cm ON f.id = cm.fact_id
                JOIN fact_clusters fc ON cm.cluster_id = fc.id
                WHERE fc.cluster_name = %s
                ORDER BY cm.relevance_score DESC, f.confidence DESC
            """, (cluster_name,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error getting cluster facts: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        try:
            cursor = self.connection.cursor()
            
            stats = {}
            
            # Total facts
            cursor.execute("SELECT COUNT(*) FROM facts")
            stats['total_facts'] = cursor.fetchone()[0]
            
            # Facts by category
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM facts
                GROUP BY category
            """)
            stats['by_category'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Facts by source
            cursor.execute("""
                SELECT source, COUNT(*) as count
                FROM facts
                GROUP BY source
            """)
            stats['by_source'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Queue status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM learning_queue
                GROUP BY status
            """)
            stats['queue_status'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Learning log entries
            cursor.execute("SELECT COUNT(*) FROM learning_log")
            stats['learning_log_entries'] = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM facts")
            result = cursor.fetchone()[0]
            stats['average_confidence'] = round(result, 2) if result else 0
            
            cursor.close()
            return stats
            
        except Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
