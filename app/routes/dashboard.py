# """Dashboard API routes for doctor interface"""

# from fastapi import APIRouter, HTTPException, Depends
# from typing import List
# import logging
# from datetime import datetime

# from app.db.database import get_all_intakes, get_intake_by_id, get_db

# logger = logging.getLogger(__name__)
# router = APIRouter()


# @router.get("/intakes")
# async def list_all_intakes():
#     """
#     Get all intake records for the doctor dashboard.
#     Returns most recent first.
#     """
#     try:
#         intakes = get_all_intakes()
#         return {"count": len(intakes), "intakes": intakes}
#     except Exception as e:
#         logger.error(f"Failed to fetch intakes: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/intakes/{record_id}")
# async def get_intake(record_id: int):
#     """
#     Get a single intake record by ID.
#     """
#     try:
#         intake = get_intake_by_id(record_id)
#         if not intake:
#             raise HTTPException(status_code=404, detail="Intake not found")
#         return intake
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Failed to fetch intake {record_id}: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/stats")
# async def get_dashboard_stats():
#     """
#     Get summary statistics for the dashboard.
#     """
#     try:
#         intakes = get_all_intakes()

#         urgent_count = sum(1 for i in intakes if i.get("risk_level") == "urgent")
#         moderate_count = sum(1 for i in intakes if i.get("risk_level") == "moderate")
#         routine_count = sum(1 for i in intakes if i.get("risk_level") == "routine")

#         return {
#             "total_intakes": len(intakes),
#             "urgent": urgent_count,
#             "moderate": moderate_count,
#             "routine": routine_count,
#             "today_count": len(intakes),  # TODO: Filter by today
#         }
#     except Exception as e:
#         logger.error(f"Failed to fetch stats: {e}")
#         raise HTTPException(status_code=500, detail=str(e))



# @router.get("/alerts")
# async def get_dashboard_alerts(limit: int = 10, db=Depends(get_db)):
#     """
#     Get active alerts for the dashboard.
#     Includes outbreak alerts and urgent cases requiring attention.
#     """
#     try:
#         alerts = []

#         # Get outbreak alerts
#         try:
#             from app.services.outbreak_detection import OutbreakDetector

#             detector = OutbreakDetector(db)
#             outbreak_alerts = await detector.get_active_alerts()
#             alerts.extend(
#                 [
#                     {
#                         "id": f"outbreak_{i}",
#                         "type": "outbreak",
#                         "severity": "high",
#                         "title": alert.get("disease", "Unknown") + " Outbreak Alert",
#                         "message": f"{alert.get('case_count', 0)} cases detected in {alert.get('location', 'area')}",
#                         "created_at": alert.get(
#                             "detected_at", datetime.utcnow().isoformat()
#                         ),
#                         "acknowledged": False,
#                     }
#                     for i, alert in enumerate(outbreak_alerts)
#                 ]
#             )
#         except Exception as e:
#             logger.warning(f"Could not fetch outbreak alerts: {e}")

#         # Get urgent unreviewed cases - use correct table name
#         try:
#             urgent_cases = db.fetch_all(
#                 """
#                 SELECT id, chief_complaint, created_at 
#                 FROM intake_records 
#                 WHERE risk_level = 'urgent' 
#                 ORDER BY created_at DESC
#                 LIMIT ?
#             """,
#                 (limit,),
#             )

#             alerts.extend(
#                 [
#                     {
#                         "id": f"urgent_{case['id']}",
#                         "type": "urgent_case",
#                         "severity": "urgent",
#                         "title": "Urgent Case Requires Review",
#                         "message": f"Case #{case['id']}: {case['chief_complaint']}",
#                         "created_at": case["created_at"],
#                         "acknowledged": False,
#                         "intake_id": case["id"],
#                     }
#                     for case in urgent_cases
#                 ]
#             )
#         except Exception as e:
#             logger.warning(f"Could not fetch urgent cases: {e}")

#         # Sort by severity and date
#         severity_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
#         alerts.sort(
#             key=lambda x: (severity_order.get(x["severity"], 4), x["created_at"]),
#             reverse=True,
#         )

#         return {
#             "alerts": alerts[:limit],
#             "count": len(alerts),
#             "unacknowledged": sum(
#                 1 for a in alerts if not a.get("acknowledged", False)
#             ),
#         }

#     except Exception as e:
#         logger.error(f"Failed to get alerts: {e}")
#         return {"alerts": [], "count": 0, "unacknowledged": 0}


# @router.post("/alerts/{alert_id}/acknowledge")
# async def acknowledge_alert(alert_id: str):
#     """Mark an alert as acknowledged"""
#     return {"status": "acknowledged", "alert_id": alert_id}

"""Dashboard API routes for doctor interface"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from datetime import datetime

from app.db.database import get_all_intakes, get_intake_by_id, get_connection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/intakes")
async def list_all_intakes():
    """
    Get all intake records for the doctor dashboard.
    Returns most recent first.
    """
    try:
        intakes = get_all_intakes()
        return {"count": len(intakes), "intakes": intakes}
    except Exception as e:
        logger.error(f"Failed to fetch intakes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intakes/{record_id}")
async def get_intake(record_id: int):
    """
    Get a single intake record by ID.
    """
    try:
        intake = get_intake_by_id(record_id)
        if not intake:
            raise HTTPException(status_code=404, detail="Intake not found")
        return intake
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch intake {record_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_dashboard_stats():
    """
    Get summary statistics for the dashboard.
    """
    try:
        intakes = get_all_intakes()

        urgent_count = sum(1 for i in intakes if i.get("risk_level") == "urgent")
        moderate_count = sum(1 for i in intakes if i.get("risk_level") == "moderate")
        routine_count = sum(1 for i in intakes if i.get("risk_level") == "routine")

        return {
            "total_intakes": len(intakes),
            "urgent": urgent_count,
            "moderate": moderate_count,
            "routine": routine_count,
            "today_count": len(intakes),  # TODO: Filter by today
        }
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_dashboard_alerts(limit: int = 10):
    """
    Get active alerts for the dashboard.
    Includes outbreak alerts and urgent cases requiring attention.
    """
    try:
        alerts = []

        # Get outbreak alerts using correct class name
        try:
            from app.services.outbreak_detection import TemporalSpatialAnalyzer
            
            analyzer = TemporalSpatialAnalyzer()
            outbreak_alerts = analyzer.get_active_alerts()  # Not async, returns List[HealthAlert]
            
            alerts.extend(
                [
                    {
                        "id": f"outbreak_{i}",
                        "type": "outbreak",
                        "severity": alert.severity,
                        "title": f"{alert.condition} Outbreak Alert",
                        "message": f"{alert.case_count} cases detected in {alert.affected_area}",
                        "created_at": alert.first_detected.isoformat(),
                        "acknowledged": False,
                    }
                    for i, alert in enumerate(outbreak_alerts)
                ]
            )
        except Exception as e:
            logger.warning(f"Could not fetch outbreak alerts: {e}")

        # Get urgent unreviewed cases - create fresh connection to avoid threading issues
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, chief_complaint, created_at 
                FROM intake_records 
                WHERE risk_level = 'urgent' 
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            urgent_cases = [dict(row) for row in cursor.fetchall()]
            conn.close()

            alerts.extend(
                [
                    {
                        "id": f"urgent_{case['id']}",
                        "type": "urgent_case",
                        "severity": "urgent",
                        "title": "Urgent Case Requires Review",
                        "message": f"Case #{case['id']}: {case['chief_complaint'] or 'No complaint recorded'}",
                        "created_at": case["created_at"],
                        "acknowledged": False,
                        "intake_id": case["id"],
                    }
                    for case in urgent_cases
                ]
            )
        except Exception as e:
            logger.warning(f"Could not fetch urgent cases: {e}")

        # Sort by severity and date
        severity_order = {"urgent": 0, "critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(
            key=lambda x: (severity_order.get(x["severity"], 4), x.get("created_at", "")),
            reverse=True,
        )

        return {
            "alerts": alerts[:limit],
            "count": len(alerts),
            "unacknowledged": sum(
                1 for a in alerts if not a.get("acknowledged", False)
            ),
        }

    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return {"alerts": [], "count": 0, "unacknowledged": 0}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Mark an alert as acknowledged"""
    return {"status": "acknowledged", "alert_id": alert_id}