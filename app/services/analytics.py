"""
Public Health Analytics Service
Provides outbreak detection, trend analysis, and population health insights.
Anonymized data only - compliant with DPDP Act.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import Counter
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)


class HealthAnalytics:
    """
    Analytics service for public health insights.
    Uses anonymized/pseudonymized data only.
    """
    
    # Alert thresholds for outbreak detection
    OUTBREAK_THRESHOLDS = {
        "fever": {"daily_threshold": 10, "weekly_increase_pct": 50},
        "diarrhea": {"daily_threshold": 8, "weekly_increase_pct": 40},
        "malaria": {"daily_threshold": 5, "weekly_increase_pct": 30},
        "dengue": {"daily_threshold": 3, "weekly_increase_pct": 25},
        "respiratory": {"daily_threshold": 15, "weekly_increase_pct": 60},
    }
    
    def __init__(self, db_path: str = "ghia.db"):
        self.db_path = db_path
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_symptom_trends(
        self,
        phc_id: str = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get symptom trends for outbreak detection.
        Returns anonymized aggregate data only.
        """
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT DATE(created_at) as date, chief_complaint, symptoms_json, risk_level
        FROM intake_records
        WHERE created_at > ?
        """
        params = [since]
        
        if phc_id:
            query += " AND phc_id = ?"
            params.append(phc_id)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Aggregate by date and symptom category
        daily_counts = {}
        symptom_counts = Counter()
        risk_distribution = Counter()
        
        for row in rows:
            date = row["date"]
            if date not in daily_counts:
                daily_counts[date] = {"total": 0, "categories": Counter()}
            
            daily_counts[date]["total"] += 1
            
            # Categorize chief complaint
            complaint = (row["chief_complaint"] or "").lower()
            category = self._categorize_symptom(complaint)
            daily_counts[date]["categories"][category] += 1
            symptom_counts[category] += 1
            
            risk_distribution[row["risk_level"]] += 1
        
        return {
            "period_days": days,
            "total_cases": len(rows),
            "daily_counts": daily_counts,
            "symptom_distribution": dict(symptom_counts.most_common(10)),
            "risk_distribution": dict(risk_distribution),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _categorize_symptom(self, complaint: str) -> str:
        """Categorize symptom for aggregation"""
        categories = {
            "fever": ["fever", "bukhar", "बुखार", "temperature"],
            "respiratory": ["cough", "khansi", "खांसी", "cold", "breathing", "saans"],
            "diarrhea": ["diarrhea", "dast", "दस्त", "loose", "vomit"],
            "pain": ["pain", "dard", "दर्द", "ache"],
            "skin": ["rash", "itching", "khujli", "skin"],
            "maternal": ["pregnant", "delivery", "bleeding", "anc"],
        }
        
        for category, keywords in categories.items():
            if any(kw in complaint for kw in keywords):
                return category
        return "other"
    
    def check_outbreak_alerts(self, phc_id: str = None) -> List[Dict]:
        """
        Check for potential disease outbreaks.
        Compares current week with previous week.
        """
        alerts = []
        
        current_week_start = datetime.utcnow() - timedelta(days=7)
        previous_week_start = current_week_start - timedelta(days=7)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get current week data
        query = """
        SELECT chief_complaint FROM intake_records
        WHERE created_at > ? AND created_at <= ?
        """
        params = [current_week_start.isoformat(), datetime.utcnow().isoformat()]
        
        if phc_id:
            query += " AND phc_id = ?"
            params.append(phc_id)
        
        cursor.execute(query, params)
        current_week = [self._categorize_symptom(row["chief_complaint"] or "") for row in cursor.fetchall()]
        
        # Get previous week data
        params_prev = [previous_week_start.isoformat(), current_week_start.isoformat()]
        if phc_id:
            params_prev.append(phc_id)
        
        cursor.execute(query, params_prev)
        previous_week = [self._categorize_symptom(row["chief_complaint"] or "") for row in cursor.fetchall()]
        conn.close()
        
        current_counts = Counter(current_week)
        previous_counts = Counter(previous_week)
        
        for category, threshold in self.OUTBREAK_THRESHOLDS.items():
            current_count = current_counts.get(category, 0)
            previous_count = previous_counts.get(category, 0)
            
            # Check absolute threshold
            if current_count >= threshold["daily_threshold"] * 7:
                alerts.append({
                    "type": "high_volume",
                    "category": category,
                    "current_count": current_count,
                    "threshold": threshold["daily_threshold"] * 7,
                    "severity": "warning",
                    "message_hindi": f"{category} के मामले सामान्य से अधिक हैं",
                    "message_english": f"Higher than usual {category} cases",
                    "recommendation": "Investigate potential outbreak"
                })
            
            # Check week-over-week increase
            if previous_count > 0:
                increase_pct = ((current_count - previous_count) / previous_count) * 100
                if increase_pct >= threshold["weekly_increase_pct"]:
                    alerts.append({
                        "type": "rapid_increase",
                        "category": category,
                        "current_count": current_count,
                        "previous_count": previous_count,
                        "increase_percent": round(increase_pct, 1),
                        "severity": "alert",
                        "message_hindi": f"{category} में {round(increase_pct)}% की वृद्धि",
                        "message_english": f"{round(increase_pct)}% increase in {category}",
                        "recommendation": "Consider public health intervention"
                    })
        
        return alerts
    
    def get_phc_dashboard_stats(self, phc_id: str) -> Dict[str, Any]:
        """
        Get dashboard statistics for a specific PHC.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = datetime.utcnow().date().isoformat()
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        month_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        
        stats = {
            "phc_id": phc_id,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Today's cases
        cursor.execute("""
        SELECT COUNT(*) as count, risk_level 
        FROM intake_records 
        WHERE DATE(created_at) = ? AND (phc_id = ? OR phc_id IS NULL)
        GROUP BY risk_level
        """, (today, phc_id))
        
        today_stats = {"total": 0, "urgent": 0, "moderate": 0, "routine": 0}
        for row in cursor.fetchall():
            today_stats[row["risk_level"]] = row["count"]
            today_stats["total"] += row["count"]
        stats["today"] = today_stats
        
        # This week
        cursor.execute("""
        SELECT COUNT(*) as count FROM intake_records 
        WHERE created_at > ? AND (phc_id = ? OR phc_id IS NULL)
        """, (week_ago, phc_id))
        stats["this_week_total"] = cursor.fetchone()["count"]
        
        # This month
        cursor.execute("""
        SELECT COUNT(*) as count FROM intake_records 
        WHERE created_at > ? AND (phc_id = ? OR phc_id IS NULL)
        """, (month_ago, phc_id))
        stats["this_month_total"] = cursor.fetchone()["count"]
        
        # Top complaints this week
        cursor.execute("""
        SELECT chief_complaint, COUNT(*) as count 
        FROM intake_records 
        WHERE created_at > ? AND (phc_id = ? OR phc_id IS NULL) AND chief_complaint IS NOT NULL
        GROUP BY chief_complaint
        ORDER BY count DESC
        LIMIT 5
        """, (week_ago, phc_id))
        stats["top_complaints"] = [{"complaint": row["chief_complaint"], "count": row["count"]} for row in cursor.fetchall()]
        
        conn.close()
        
        # Check for alerts
        stats["alerts"] = self.check_outbreak_alerts(phc_id)
        
        return stats


# Singleton
_analytics: Optional[HealthAnalytics] = None

def get_analytics() -> HealthAnalytics:
    global _analytics
    if _analytics is None:
        _analytics = HealthAnalytics()
    return _analytics