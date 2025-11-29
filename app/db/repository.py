"""Database repository functions - bridges agent pipeline with database"""
from typing import Dict, Optional, List
from app.db.database import save_intake_record, get_all_intakes, get_intake_by_id
import logging

logger = logging.getLogger(__name__)


def save_intake(state: Dict) -> int:
    """
    Save intake from agent pipeline state to database.
    
    Args:
        state: Final state from agent pipeline
        
    Returns:
        Record ID
    """
    try:
        # Extract data from state
        intake_data = {
            "chief_complaint": state.get("chief_complaint"),
            "symptoms": state.get("symptoms", []),
            "duration": state.get("duration"),
            "severity": state.get("severity"),
            "associated_symptoms": state.get("associated_symptoms", []),
        }
        
        record_id = save_intake_record(
            raw_transcript=state.get("raw_transcript", ""),
            language_detected=state.get("language_detected", "hi"),
            intake_data=intake_data,
            risk_level=state.get("risk_level", "routine"),
            agent_decisions=state.get("agent_decisions", []),
            summary_english=state.get("summary_english", ""),
            summary_hindi=state.get("summary_hindi", ""),
            follow_up_questions=state.get("followup_questions", []),
            recommended_action=state.get("recommended_action", "")
        )
        
        logger.info(f"✅ Saved intake record: {record_id}")
        return record_id
        
    except Exception as e:
        logger.error(f"❌ Failed to save intake: {e}")
        raise


def get_recent_intakes(limit: int = 50) -> List[Dict]:
    """Get recent intake records for dashboard"""
    return get_all_intakes()[:limit]


def get_intake_details(record_id: int) -> Optional[Dict]:
    """Get full details of a single intake"""
    return get_intake_by_id(record_id)