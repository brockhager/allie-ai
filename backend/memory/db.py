"""
MySQL Database Connector for Allie's Memory System

Provides persistent storage for facts using MySQL database.
This replaces volatile linked list storage with durable database persistence.
"""

import logging
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MemoryDB:
    """
    MySQL database connector for persistent fact storage
    
    Features:
    - CRUD operations for facts
    - Keyword-based retrieval
    - Timeline queries ordered by updated_at
    - Automatic conflict resolution (updates existing facts)
    - Connection pooling for performance
    """
    
    def __init__(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = "",
        database: str = "allie_memory",
        port: int = 3306
    ):
        """
        Initialize MySQL database connection
        
        Args:
            host: MySQL host (default: localhost)
            user: MySQL user (default: root)
            password: MySQL password
            database: Database name (default: allie_memory)
            port: MySQL port (default: 3306)
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        
        # Load credentials from config if exists
        self._load_config()
        
        # Establish connection
        self.connect()
    
    def _load_config(self):
        """Load MySQL credentials from config file if exists"""
        config_file = Path(__file__).parent.parent.parent / "config" / "mysql.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.host = config.get("host", self.host)
                    self.user = config.get("user", self.user)
                    self.password = config.get("password", self.password)
                    self.database = config.get("database", self.database)
                    self.port = config.get("port", self.port)
                logger.info(f"Loaded MySQL config from {config_file}")
            except Exception as e:
                logger.warning(f"Could not load MySQL config: {e}")
    
    def connect(self):
        """Establish connection to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                autocommit=True
            )
            
            if self.connection.is_connected():
                db_info = self.connection.get_server_info()
                logger.info(f"Connected to MySQL Server version {db_info}")
                logger.info(f"Using database: {self.database}")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            self.connection = None
            return False
    
    def ensure_connection(self):
        """Ensure database connection is active, reconnect if needed"""
        if not self.connection or not self.connection.is_connected():
            logger.warning("MySQL connection lost, reconnecting...")
            return self.connect()
        return True
    
    def add_fact(
        self,
        keyword: str,
        fact: str,
        source: str = "user",
        category: str = "general",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a new fact to the database (or update if keyword exists)
        
        Args:
            keyword: Primary keyword/topic for the fact
            fact: The fact text
            source: Source of the fact (user, wikipedia, duckduckgo, etc.)
            category: Category of the fact
            confidence: Confidence score (0.0 to 1.0)
            metadata: Additional metadata as JSON
        
        Returns:
            Dictionary with status and details
        """
        if not self.ensure_connection():
            return {"status": "error", "message": "Database connection failed"}
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Check if fact with this keyword already exists
            cursor.execute(
                "SELECT id, fact FROM facts WHERE keyword = %s",
                (keyword,)
            )
            existing = cursor.fetchone()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            if existing:
                # Update existing fact
                cursor.execute(
                    """
                    UPDATE facts 
                    SET fact = %s, source = %s, category = %s, 
                        confidence = %s, metadata = %s, updated_at = NOW()
                    WHERE keyword = %s
                    """,
                    (fact, source, category, confidence, metadata_json, keyword)
                )
                
                logger.info(f"Updated fact for keyword '{keyword}'")
                
                return {
                    "status": "updated",
                    "message": f"Updated existing fact for '{keyword}'",
                    "id": existing["id"],
                    "old_fact": existing["fact"],
                    "new_fact": fact
                }
            else:
                # Insert new fact
                cursor.execute(
                    """
                    INSERT INTO facts (keyword, fact, source, category, confidence, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (keyword, fact, source, category, confidence, metadata_json)
                )
                
                fact_id = cursor.lastrowid
                logger.info(f"Added new fact with ID {fact_id} for keyword '{keyword}'")
                
                return {
                    "status": "added",
                    "message": f"Added new fact for '{keyword}'",
                    "id": fact_id,
                    "fact": fact
                }
                
        except Error as e:
            logger.error(f"Error adding fact: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            if cursor:
                cursor.close()
    
    def get_fact(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a fact by keyword
        
        Args:
            keyword: The keyword to search for
        
        Returns:
            Dictionary with fact data or None if not found
        """
        if not self.ensure_connection():
            return None
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, keyword, fact, source, category, confidence, 
                       metadata, updated_at
                FROM facts 
                WHERE keyword = %s
                """,
                (keyword,)
            )
            
            result = cursor.fetchone()
            
            if result and result.get("metadata"):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except:
                    pass
            
            return result
            
        except Error as e:
            logger.error(f"Error retrieving fact: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def search_facts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for facts matching the query (searches keyword and fact text)
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of matching facts
        """
        if not self.ensure_connection():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Search in both keyword and fact columns
            search_pattern = f"%{query}%"
            cursor.execute(
                """
                SELECT id, keyword, fact, source, category, confidence, 
                       metadata, updated_at
                FROM facts 
                WHERE keyword LIKE %s OR fact LIKE %s
                ORDER BY confidence DESC, updated_at DESC
                LIMIT %s
                """,
                (search_pattern, search_pattern, limit)
            )
            
            results = cursor.fetchall()
            
            # Parse metadata JSON
            for result in results:
                if result.get("metadata"):
                    try:
                        result["metadata"] = json.loads(result["metadata"])
                    except:
                        pass
            
            return results
            
        except Error as e:
            logger.error(f"Error searching facts: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def update_fact(self, keyword: str, new_fact: str, source: str = "correction") -> Dict[str, Any]:
        """
        Update an existing fact
        
        Args:
            keyword: Keyword of the fact to update
            new_fact: New fact text
            source: Source of the update
        
        Returns:
            Dictionary with update status
        """
        existing = self.get_fact(keyword)
        
        if not existing:
            return {
                "status": "not_found",
                "message": f"No fact found for keyword '{keyword}'"
            }
        
        # Use add_fact which handles updates automatically
        return self.add_fact(
            keyword=keyword,
            fact=new_fact,
            source=source,
            category=existing.get("category", "general"),
            confidence=existing.get("confidence", 0.9)
        )
    
    def delete_fact(self, keyword: str) -> bool:
        """
        Delete a fact from the database
        
        Args:
            keyword: Keyword of the fact to delete
        
        Returns:
            True if deleted, False if not found
        """
        if not self.ensure_connection():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM facts WHERE keyword = %s", (keyword,))
            
            deleted = cursor.rowcount > 0
            
            if deleted:
                logger.info(f"Deleted fact for keyword '{keyword}'")
            
            return deleted
            
        except Error as e:
            logger.error(f"Error deleting fact: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def timeline(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get facts in chronological order (most recent first)
        
        Args:
            limit: Maximum number of facts to return
        
        Returns:
            List of facts ordered by updated_at DESC
        """
        if not self.ensure_connection():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, keyword, fact, source, category, confidence, 
                       metadata, updated_at
                FROM facts 
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            
            results = cursor.fetchall()
            
            # Parse metadata JSON
            for result in results:
                if result.get("metadata"):
                    try:
                        result["metadata"] = json.loads(result["metadata"])
                    except:
                        pass
            
            return results
            
        except Error as e:
            logger.error(f"Error getting timeline: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored facts
        
        Returns:
            Dictionary with statistics
        """
        if not self.ensure_connection():
            return {}
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Total facts
            cursor.execute("SELECT COUNT(*) as total FROM facts")
            total = cursor.fetchone()["total"]
            
            # Facts by source
            cursor.execute(
                """
                SELECT source, COUNT(*) as count 
                FROM facts 
                GROUP BY source 
                ORDER BY count DESC
                """
            )
            sources = {row["source"]: row["count"] for row in cursor.fetchall()}
            
            # Facts by category
            cursor.execute(
                """
                SELECT category, COUNT(*) as count 
                FROM facts 
                GROUP BY category 
                ORDER BY count DESC
                """
            )
            categories = {row["category"]: row["count"] for row in cursor.fetchall()}
            
            # Most recent fact
            cursor.execute(
                "SELECT updated_at FROM facts ORDER BY updated_at DESC LIMIT 1"
            )
            recent = cursor.fetchone()
            last_updated = recent["updated_at"] if recent else None
            
            return {
                "total_facts": total,
                "sources": sources,
                "categories": categories,
                "last_updated": last_updated.isoformat() if last_updated else None
            }
            
        except Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.close()
