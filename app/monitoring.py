"""
Monitoring and metrics tracking for RAG pipeline.
Tracks query performance, quality metrics, and costs.
"""
import json
import sqlite3
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config import MONITORING_DB_PATH, MONITORING_ENABLED, LOG_QUERIES


class MetricsTracker:
    """Track and store RAG pipeline metrics."""
    
    def __init__(self):
        self.db_path = MONITORING_DB_PATH
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                response_time REAL NOT NULL,
                token_count INTEGER,
                confidence_score REAL,
                confidence_level TEXT,
                retrieval_score REAL,
                num_docs_retrieved INTEGER,
                faithfulness_score REAL,
                success BOOLEAN NOT NULL,
                error_message TEXT
            )
        """)
        
        # Aggregated metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT PRIMARY KEY,
                total_queries INTEGER,
                avg_response_time REAL,
                avg_confidence REAL,
                success_rate REAL,
                total_tokens INTEGER,
                estimated_cost REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_query(
        self,
        question: str,
        endpoint: str,
        response_time: float,
        token_count: Optional[int] = None,
        confidence_score: Optional[float] = None,
        confidence_level: Optional[str] = None,
        retrieval_score: Optional[float] = None,
        num_docs: Optional[int] = None,
        faithfulness_score: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log a query with its metrics."""
        if not MONITORING_ENABLED or not LOG_QUERIES:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO query_logs (
                    timestamp, question, endpoint, response_time, token_count,
                    confidence_score, confidence_level, retrieval_score,
                    num_docs_retrieved, faithfulness_score, success, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                question,
                endpoint,
                response_time,
                token_count,
                confidence_score,
                confidence_level,
                retrieval_score,
                num_docs,
                faithfulness_score,
                success,
                error_message
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging query: {e}")
    
    def get_recent_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent queries."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM query_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get aggregated metrics for the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT
                COUNT(*) as total_queries,
                AVG(response_time) as avg_response_time,
                AVG(confidence_score) as avg_confidence,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate,
                SUM(token_count) as total_tokens,
                AVG(retrieval_score) as avg_retrieval_score,
                AVG(faithfulness_score) as avg_faithfulness
            FROM query_logs
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
        """, (days,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            total_tokens = row[4] or 0
            # GPT-4o-mini pricing: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
            # Estimate: 70% input, 30% output
            estimated_cost = (total_tokens * 0.7 * 0.15 / 1_000_000) + (total_tokens * 0.3 * 0.60 / 1_000_000)
            
            return {
                "total_queries": row[0] or 0,
                "avg_response_time": round(row[1] or 0, 3),
                "avg_confidence": round(row[2] or 0, 3),
                "success_rate": round(row[3] or 0, 3),
                "total_tokens": total_tokens,
                "estimated_cost": round(estimated_cost, 4),
                "avg_retrieval_score": round(row[5] or 0, 3),
                "avg_faithfulness": round(row[6] or 0, 3)
            }
        
        return {}
    
    def get_endpoint_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics per endpoint."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT
                endpoint,
                COUNT(*) as count,
                AVG(response_time) as avg_time,
                AVG(confidence_score) as avg_confidence
            FROM query_logs
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY endpoint
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = {}
        for row in rows:
            stats[row[0]] = {
                "count": row[1],
                "avg_response_time": round(row[2], 3),
                "avg_confidence": round(row[3] or 0, 3)
            }
        
        return stats
    
    def get_low_confidence_queries(self, threshold: float = 0.6, limit: int = 20) -> List[Dict[str, Any]]:
        """Get queries with low confidence scores for review."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, question, confidence_score, confidence_level, endpoint
            FROM query_logs
            WHERE confidence_score < ? AND success = 1
            ORDER BY timestamp DESC
            LIMIT ?
        """, (threshold, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


# Global metrics tracker instance
_metrics_tracker = None


def get_metrics_tracker() -> MetricsTracker:
    """Get or create global metrics tracker instance."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker
