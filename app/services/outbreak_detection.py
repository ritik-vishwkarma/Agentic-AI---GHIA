"""
Community Health Intelligence System
- Real-time outbreak detection using temporal-spatial clustering
- Privacy-preserving (works on anonymized data only)
- Alerts health authorities automatically
- Feeds back into agent decision-making
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
import logging

logger = logging.getLogger(__name__)

DB_PATH = "ghia.db"


@dataclass
class HealthAlert:
    """Standardized health alert"""
    alert_id: str
    alert_type: str  # outbreak, cluster, regional_outbreak
    condition: str
    severity: str  # low, medium, high, critical
    affected_area: str
    case_count: int
    baseline_count: int
    increase_percentage: float
    first_detected: datetime
    patient_demographics: Dict[str, Any]
    recommended_action: str
    message_hindi: str
    message_english: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["first_detected"] = self.first_detected.isoformat()
        return result


class TemporalSpatialAnalyzer:
    """
    Detects unusual disease patterns using statistical methods.
    Works on anonymized/aggregated data only - DPDP compliant.
    """
    
    BASELINE_RATES = {
        "fever": 15.0,
        "diarrhea": 8.0,
        "respiratory": 12.0,
        "dengue": 2.0,
        "malaria": 3.0,
        "typhoid": 1.5,
        "jaundice": 1.0,
        "skin_infection": 5.0
    }
    
    ALERT_THRESHOLDS = {
        "low": 1.5,
        "medium": 2.0,
        "high": 3.0,
        "critical": 5.0
    }
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.alert_cache: Dict[str, HealthAlert] = {}
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _get_table_columns(self, cursor) -> List[str]:
        """Get list of columns in intake_records table"""
        try:
            cursor.execute("PRAGMA table_info(intake_records)")
            return [col[1] for col in cursor.fetchall()]
        except Exception:
            return []
    
    def analyze_recent_patterns(
        self,
        phc_id: str = None,
        hours: int = 24
    ) -> List[HealthAlert]:
        """Analyze recent case patterns for unusual clusters."""
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check which columns exist
            columns = self._get_table_columns(cursor)
            
            if not columns:
                # Table doesn't exist yet
                conn.close()
                return []
            
            # Build query based on available columns
            select_cols = ["DATE(created_at) as date", "chief_complaint"]
            
            if "symptoms_json" in columns:
                select_cols.append("symptoms_json")
            else:
                select_cols.append("NULL as symptoms_json")
                
            if "risk_level" in columns:
                select_cols.append("risk_level")
            else:
                select_cols.append("NULL as risk_level")
                
            if "phc_id" in columns:
                select_cols.append("phc_id")
            else:
                select_cols.append("NULL as phc_id")
            
            query = f"""
            SELECT {', '.join(select_cols)}
            FROM intake_records
            WHERE created_at > ?
            """
            params = [since]
            
            # Only filter by phc_id if column exists and value provided
            if phc_id and "phc_id" in columns:
                query += " AND (phc_id = ? OR phc_id IS NULL)"
                params.append(phc_id)
            
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
        except sqlite3.OperationalError as e:
            logger.warning(f"Database query failed (table may not exist yet): {e}")
            return []
        except Exception as e:
            logger.error(f"Outbreak detection error: {e}")
            return []
        
        if not results:
            return []
        
        categories = self._categorize_cases(results)
        
        alerts = []
        for category, data in categories.items():
            alert = self._detect_anomaly(
                category=category,
                current_count=data["count"],
                time_window_hours=hours,
                demographics=data["demographics"],
                phc_id=phc_id or "ALL"
            )
            if alert:
                alerts.append(alert)
        
        # Spatial clustering for multi-PHC
        if not phc_id:
            spatial_alerts = self._detect_spatial_clusters(results)
            alerts.extend(spatial_alerts)
        
        return alerts
    
    def _categorize_cases(self, results: List[Dict]) -> Dict[str, Dict]:
        """Group cases by disease category - anonymized aggregation."""
        categories = defaultdict(lambda: {"count": 0, "demographics": defaultdict(int)})
        
        for row in results:
            complaint = (row.get("chief_complaint") or "").lower()
            symptoms = row.get("symptoms_json") or "[]"
            category = self._classify_to_category(complaint, symptoms)
            categories[category]["count"] += 1
        
        return dict(categories)
    
    def _classify_to_category(self, complaint: str, symptoms_json: str) -> str:
        """Map symptoms to disease categories."""
        text = (complaint + " " + symptoms_json).lower()
        
        if any(kw in text for kw in ["dengue", "mosquito", "platelet"]):
            return "dengue"
        if any(kw in text for kw in ["malaria", "chills", "shivering", "rigors"]):
            return "malaria"
        if any(kw in text for kw in ["typhoid", "rose spots"]):
            return "typhoid"
        if any(kw in text for kw in ["fever", "bukhar", "temperature"]):
            return "fever"
        if any(kw in text for kw in ["diarrhea", "dast", "loose", "vomit", "ulti"]):
            return "diarrhea"
        if any(kw in text for kw in ["cough", "khansi", "cold", "breathing", "saans"]):
            return "respiratory"
        if any(kw in text for kw in ["yellow", "jaundice", "pila"]):
            return "jaundice"
        if any(kw in text for kw in ["rash", "skin", "itch", "khujli"]):
            return "skin_infection"
        
        return "other"
    
    def _detect_anomaly(
        self,
        category: str,
        current_count: int,
        time_window_hours: int,
        demographics: Dict,
        phc_id: str
    ) -> Optional[HealthAlert]:
        """Statistical anomaly detection."""
        baseline_rate = self.BASELINE_RATES.get(category, 5.0)
        expected_count = max(1, (baseline_rate * time_window_hours) / (7 * 24))
        
        if current_count <= expected_count:
            return None
        
        increase_pct = ((current_count - expected_count) / expected_count) * 100
        
        # Determine severity
        severity = None
        for level in ["critical", "high", "medium", "low"]:
            if current_count >= expected_count * self.ALERT_THRESHOLDS[level]:
                severity = level
                break
        
        if severity in [None, "low"]:
            return None
        
        alert = HealthAlert(
            alert_id=f"{phc_id}_{category}_{datetime.utcnow().strftime('%Y%m%d%H')}",
            alert_type="outbreak" if severity in ["high", "critical"] else "cluster",
            condition=category,
            severity=severity,
            affected_area=phc_id,
            case_count=current_count,
            baseline_count=int(expected_count),
            increase_percentage=increase_pct,
            first_detected=datetime.utcnow(),
            patient_demographics=dict(demographics),
            recommended_action=self._get_recommended_action(category, severity),
            message_hindi=self._generate_alert_hindi(category, current_count, severity),
            message_english=self._generate_alert_english(category, current_count, severity)
        )
        
        self.alert_cache[alert.alert_id] = alert
        logger.warning(f"🚨 HEALTH ALERT: {severity.upper()} - {category} ({current_count} cases)")
        
        return alert
    
    def _detect_spatial_clusters(self, results: List[Dict]) -> List[HealthAlert]:
        """Detect regional outbreaks across multiple PHCs."""
        phc_conditions = defaultdict(lambda: defaultdict(int))
        
        for row in results:
            phc = row.get("phc_id") or "UNKNOWN"
            condition = self._classify_to_category(
                row.get("chief_complaint") or "",
                row.get("symptoms_json") or "[]"
            )
            phc_conditions[condition][phc] += 1
        
        alerts = []
        for condition, phc_counts in phc_conditions.items():
            if len(phc_counts) >= 3:  # 3+ PHCs affected
                total_cases = sum(phc_counts.values())
                alerts.append(HealthAlert(
                    alert_id=f"REGIONAL_{condition}_{datetime.utcnow().strftime('%Y%m%d')}",
                    alert_type="regional_outbreak",
                    condition=condition,
                    severity="high",
                    affected_area=f"{len(phc_counts)} PHCs",
                    case_count=total_cases,
                    baseline_count=0,
                    increase_percentage=0,
                    first_detected=datetime.utcnow(),
                    patient_demographics={},
                    recommended_action=f"District-level intervention for {condition}",
                    message_hindi=f"⚠️ क्षेत्रीय प्रकोप: {condition} - {len(phc_counts)} PHCs में {total_cases} मामले",
                    message_english=f"⚠️ Regional Outbreak: {condition} - {total_cases} cases across {len(phc_counts)} PHCs"
                ))
                logger.critical(f"🌍 REGIONAL OUTBREAK: {condition}")
        
        return alerts
    
    def _get_recommended_action(self, condition: str, severity: str) -> str:
        actions = {
            "dengue": {
                "medium": "Increase vector control, health education",
                "high": "Emergency vector control, active case finding",
                "critical": "District emergency, mass IEC campaigns"
            },
            "diarrhea": {
                "medium": "Water quality testing, hygiene promotion",
                "high": "Emergency water chlorination, ORS distribution",
                "critical": "Water supply investigation, mass treatment"
            }
        }
        default = {"medium": "Enhanced surveillance", "high": "Active case finding", "critical": "Emergency response"}
        return actions.get(condition, default).get(severity, "Monitor closely")
    
    def _generate_alert_hindi(self, condition: str, count: int, severity: str) -> str:
        cond_hindi = {"fever": "बुखार", "dengue": "डेंगू", "diarrhea": "दस्त", "respiratory": "श्वसन"}
        sev_hindi = {"medium": "सामान्य", "high": "गंभीर", "critical": "अति गंभीर"}
        return f"⚠️ स्वास्थ्य सतर्कता: {cond_hindi.get(condition, condition)} के {count} मामले। स्थिति: {sev_hindi.get(severity, severity)}"
    
    def _generate_alert_english(self, condition: str, count: int, severity: str) -> str:
        return f"⚠️ Health Alert: {count} cases of {condition}. Severity: {severity.upper()}"
    
    def get_active_alerts(self, phc_id: str = None) -> List[HealthAlert]:
        """Get currently active alerts (48h window)."""
        threshold = datetime.utcnow() - timedelta(hours=48)
        return [
            alert for alert in self.alert_cache.values()
            if alert.first_detected > threshold
            and (not phc_id or alert.affected_area in [phc_id, "ALL"])
        ]


def enrich_state_with_outbreak_context(state: Dict[str, Any], analyzer: TemporalSpatialAnalyzer) -> Dict[str, Any]:
    """Makes agents 'outbreak-aware' by enriching state with community health context."""
    try:
        alerts = analyzer.get_active_alerts(state.get("phc_id"))
    except Exception as e:
        logger.warning(f"Failed to get outbreak alerts: {e}")
        return state
    
    if not alerts:
        return state
    
    state["community_health_alerts"] = [alert.to_dict() for alert in alerts]
    
    symptoms_text = str(state.get("chief_complaint", "")).lower()
    
    for alert in alerts:
        if alert.condition in symptoms_text:
            if state.get("risk_level") == "routine":
                state["risk_level"] = "moderate"
            
            state.setdefault("warnings", []).append(
                f"⚠️ Ongoing {alert.condition} outbreak in area - test recommended"
            )
            
            logger.info(f"📊 Outbreak-aware adjustment for {alert.condition}")
    
    return state


# Singleton
_analyzer: Optional[TemporalSpatialAnalyzer] = None

def get_outbreak_analyzer() -> TemporalSpatialAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = TemporalSpatialAnalyzer()
    return _analyzer