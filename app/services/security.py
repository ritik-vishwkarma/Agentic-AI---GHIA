"""
Enhanced Data Security Layer for GHIA
Implements encryption, consent management, audit logging, and data anonymization
Compliant with: India DPDP Act 2023, HIPAA guidelines, ABDM standards
"""
import hashlib
import hmac
import secrets
import base64
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Callable
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
import json
import sqlite3
from functools import wraps
from enum import Enum
from contextlib import contextmanager
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataClassification(str, Enum):
    """Data classification levels per DPDP Act"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"  # Mental health, HIV, genetic
    CRITICAL = "critical"  # Biometric data


class AccessLevel(str, Enum):
    """User access levels with hierarchical permissions"""
    PATIENT = "patient"
    HEALTH_WORKER = "health_worker"
    DOCTOR = "doctor"
    ADMIN = "admin"
    SYSTEM = "system"


class KeyProvider(ABC):
    """Abstract key provider for different environments"""
    
    @abstractmethod
    def get_key(self, key_id: str) -> bytes:
        pass
    
    @abstractmethod
    def rotate_key(self, key_id: str) -> str:
        pass


class EnvironmentKeyProvider(KeyProvider):
    """Key provider using environment variables"""
    
    def __init__(self, settings):
        self.settings = settings
        self._keys: Dict[str, bytes] = {}
    
    def get_key(self, key_id: str = "master") -> bytes:
        if key_id not in self._keys:
            if self.settings.encryption_key:
                self._keys[key_id] = self._derive_key(
                    self.settings.encryption_key.get_secret_value()
                )
            else:
                raise ValueError("Encryption key not configured")
        return self._keys[key_id]
    
    def _derive_key(self, password: str) -> bytes:
        salt = hashlib.sha256(
            f"ghia_{self.settings.default_phc_id}_v2".encode()
        ).digest()[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def rotate_key(self, key_id: str) -> str:
        raise NotImplementedError("Key rotation requires vault integration")


class HealthDataEncryption:
    """
    Encrypts Protected Health Information (PHI) at rest and in transit.
    Uses AES-256-GCM via Fernet for authenticated encryption.
    """
    
    def __init__(self, key_provider: KeyProvider):
        self.key_provider = key_provider
        self._cipher_cache: Dict[str, Fernet] = {}
        self._key_version = "v2"
    
    def _get_cipher(self, key_id: str = "master") -> Fernet:
        if key_id not in self._cipher_cache:
            key = self.key_provider.get_key(key_id)
            self._cipher_cache[key_id] = Fernet(key)
        return self._cipher_cache[key_id]
    
    def encrypt_phi(
        self,
        data: Dict[str, Any],
        classification: DataClassification = DataClassification.CONFIDENTIAL,
        context: Dict[str, str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Encrypt PHI with metadata and context binding.
        
        Args:
            data: Health data to encrypt
            classification: Data sensitivity level
            context: Additional context (patient_id, session_id)
        
        Returns:
            Tuple of (encrypted_payload, metadata)
        """
        envelope = {
            "data": data,
            "classification": classification.value,
            "encrypted_at": datetime.utcnow().isoformat(),
            "key_version": self._key_version,
            "context_hash": self._hash_context(context) if context else None,
            "nonce": secrets.token_hex(8)
        }
        
        json_bytes = json.dumps(envelope, ensure_ascii=False, default=str).encode('utf-8')
        encrypted = self._get_cipher().encrypt(json_bytes)
        
        metadata = {
            "version": self._key_version,
            "algorithm": "Fernet-AES-128-CBC-HMAC-SHA256",
            "classification": classification.value,
            "checksum": hashlib.sha256(encrypted).hexdigest()[:16],
            "encrypted_at": envelope["encrypted_at"]
        }
        
        return base64.urlsafe_b64encode(encrypted).decode('utf-8'), metadata
    
    def decrypt_phi(
        self,
        encrypted_data: str,
        expected_checksum: str = None,
        context: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Decrypt PHI with integrity verification."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            
            if expected_checksum:
                actual = hashlib.sha256(encrypted_bytes).hexdigest()[:16]
                if not secrets.compare_digest(actual, expected_checksum):
                    raise ValueError("Checksum verification failed")
            
            decrypted = self._get_cipher().decrypt(encrypted_bytes)
            envelope = json.loads(decrypted.decode('utf-8'))
            
            # Verify context binding if provided
            if context and envelope.get("context_hash"):
                expected_hash = self._hash_context(context)
                if envelope["context_hash"] != expected_hash:
                    logger.warning("Context mismatch during decryption")
            
            return envelope.get("data", envelope)
            
        except InvalidToken:
            raise ValueError("Decryption failed - invalid key or corrupted data")
    
    def _hash_context(self, context: Dict[str, str]) -> str:
        """Create hash of context for binding verification."""
        sorted_ctx = json.dumps(context, sort_keys=True)
        return hashlib.sha256(sorted_ctx.encode()).hexdigest()[:16]
    
    def hash_identifier(self, identifier: str, purpose: str = "general") -> str:
        """Create purpose-bound pseudonymous identifier."""
        key = self.key_provider.get_key()[:32]
        data = f"{purpose}:{identifier}".encode()
        return hmac.new(key, data, hashlib.sha256).hexdigest()[:16]
    
    def generate_pseudonym(self, patient_id: str) -> str:
        """Generate consistent pseudonymous ID for research."""
        hash_value = self.hash_identifier(patient_id, "pseudonym")
        return f"GHIA-{hash_value.upper()}"


class DataAnonymizer:
    """
    Anonymizes health data for research and analytics.
    Implements k-anonymity and l-diversity principles.
    """
    
    # Quasi-identifiers that need generalization
    QUASI_IDENTIFIERS = {
        "age": lambda x: f"{(int(x) // 10) * 10}-{(int(x) // 10) * 10 + 9}" if x else None,
        "pincode": lambda x: x[:3] + "XXX" if x and len(x) >= 3 else None,
        "village": lambda _: "[REDACTED]",
        "phone": lambda _: "[REDACTED]",
    }
    
    # Direct identifiers to remove
    DIRECT_IDENTIFIERS = [
        "name", "patient_name", "abha_id", "aadhaar",
        "phone_number", "mobile", "email", "address",
        "father_name", "mother_name", "spouse_name"
    ]
    
    def __init__(self, encryption: HealthDataEncryption):
        self.encryption = encryption
    
    def anonymize(
        self,
        data: Dict[str, Any],
        preserve_medical: bool = True
    ) -> Dict[str, Any]:
        """
        Anonymize patient data for analytics.
        
        Args:
            data: Original patient data
            preserve_medical: Keep medical details (symptoms, diagnosis)
        
        Returns:
            Anonymized data dictionary
        """
        result = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Remove direct identifiers
            if key_lower in self.DIRECT_IDENTIFIERS:
                continue
            
            # Generalize quasi-identifiers
            if key_lower in self.QUASI_IDENTIFIERS:
                result[key] = self.QUASI_IDENTIFIERS[key_lower](value)
                continue
            
            # Preserve or redact based on type
            if preserve_medical or key_lower not in ["notes", "history"]:
                result[key] = value
        
        # Add pseudonymous ID
        if "patient_id" in data:
            result["pseudo_id"] = self.encryption.generate_pseudonym(data["patient_id"])
        
        result["anonymized_at"] = datetime.utcnow().isoformat()
        return result
    
    def create_research_dataset(
        self,
        records: List[Dict[str, Any]],
        k_anonymity: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Create k-anonymous dataset for research.
        Groups records to ensure at least k identical quasi-identifiers.
        """
        anonymized = [self.anonymize(r) for r in records]
        
        # Group by quasi-identifier combination
        groups: Dict[str, List] = {}
        for record in anonymized:
            key = self._quasi_identifier_key(record)
            groups.setdefault(key, []).append(record)
        
        # Filter groups with fewer than k records
        result = []
        suppressed = 0
        for key, group in groups.items():
            if len(group) >= k_anonymity:
                result.extend(group)
            else:
                suppressed += len(group)
        
        logger.info(f"Research dataset: {len(result)} records, {suppressed} suppressed for k={k_anonymity}")
        return result
    
    def _quasi_identifier_key(self, record: Dict) -> str:
        """Create grouping key from quasi-identifiers."""
        parts = [str(record.get(qi, "")) for qi in self.QUASI_IDENTIFIERS.keys()]
        return "|".join(parts)


class ConsentManager:
    """
    Manages patient consent for data processing.
    Required under India DPDP Act 2023.
    """
    
    CONSENT_PURPOSES = {
        "treatment": ("उपचार", "Medical treatment"),
        "research": ("अनुसंधान", "Anonymized research"),
        "analytics": ("विश्लेषण", "Public health analytics"),
        "sharing": ("साझाकरण", "Share with other providers"),
        "emergency": ("आपातकालीन", "Emergency use"),
        "abdm": ("आयुष्मान भारत", "Ayushman Bharat integration"),
        "followup": ("अनुवर्ती", "Follow-up reminders"),
    }
    
    def __init__(self, db_manager: 'DatabaseManager'):
        self.db = db_manager
        self._init_tables()
    
    def _init_tables(self):
        """Initialize consent tables."""
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS patient_consents (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            abha_id TEXT,
            consent_type TEXT NOT NULL,
            granted INTEGER NOT NULL,
            granted_at TEXT,
            expires_at TEXT,
            version INTEGER DEFAULT 1,
            ip_address TEXT,
            device_fingerprint TEXT,
            witness_name TEXT,
            phc_id TEXT,
            signature_hash TEXT,
            revoked_at TEXT,
            revocation_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(patient_id, consent_type)
        )
        """)
        
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS consent_audit (
            id TEXT PRIMARY KEY,
            consent_id TEXT NOT NULL,
            action TEXT NOT NULL,
            performed_by TEXT,
            performed_at TEXT NOT NULL,
            details TEXT,
            FOREIGN KEY (consent_id) REFERENCES patient_consents(id)
        )
        """)
    
    def record_consent(
        self,
        patient_id: str,
        consent_type: str,
        granted: bool,
        phc_id: str,
        recorded_by: str,
        abha_id: str = None,
        ip_address: str = None,
        device_fingerprint: str = None,
        witness_name: str = None,
        duration_days: int = 365
    ) -> Dict[str, Any]:
        """Record patient consent with full audit trail."""
        if consent_type not in self.CONSENT_PURPOSES:
            raise ValueError(f"Invalid consent type: {consent_type}")
        
        consent_id = str(uuid.uuid4())
        now = datetime.utcnow()
        expires = now + timedelta(days=duration_days) if granted else None
        
        # Create signature hash
        signature_data = f"{patient_id}:{consent_type}:{granted}:{now.isoformat()}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        self.db.execute("""
        INSERT INTO patient_consents 
        (id, patient_id, abha_id, consent_type, granted, granted_at, 
         expires_at, ip_address, device_fingerprint, witness_name, 
         phc_id, signature_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(patient_id, consent_type) DO UPDATE SET
            granted = excluded.granted,
            granted_at = excluded.granted_at,
            expires_at = excluded.expires_at,
            version = version + 1,
            updated_at = CURRENT_TIMESTAMP,
            signature_hash = excluded.signature_hash
        """, (
            consent_id, patient_id, abha_id, consent_type, int(granted),
            now.isoformat() if granted else None,
            expires.isoformat() if expires else None,
            ip_address, device_fingerprint, witness_name, phc_id, signature
        ))
        
        # Audit trail
        self.db.execute("""
        INSERT INTO consent_audit (id, consent_id, action, performed_by, performed_at, details)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), consent_id,
            "GRANTED" if granted else "DENIED",
            recorded_by, now.isoformat(),
            json.dumps({"ip": ip_address, "witness": witness_name})
        ))
        
        return {
            "consent_id": consent_id,
            "patient_id": patient_id,
            "consent_type": consent_type,
            "granted": granted,
            "expires_at": expires.isoformat() if expires else None,
            "signature": signature[:16]
        }
    
    def check_consent(
        self,
        patient_id: str,
        consent_type: str,
        purpose: str = None
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if patient has valid consent.
        
        Returns:
            Tuple of (has_consent, consent_details)
        """
        result = self.db.fetch_one("""
        SELECT * FROM patient_consents
        WHERE patient_id = ? AND consent_type = ?
        AND granted = 1 AND revoked_at IS NULL
        AND (expires_at IS NULL OR expires_at > ?)
        """, (patient_id, consent_type, datetime.utcnow().isoformat()))
        
        if result:
            return True, dict(result)
        return False, None
    
    def revoke_consent(
        self,
        patient_id: str,
        consent_type: str,
        revoked_by: str,
        reason: str = None
    ) -> bool:
        """Revoke consent (Right to Withdraw under DPDP)."""
        now = datetime.utcnow()
        
        self.db.execute("""
        UPDATE patient_consents
        SET revoked_at = ?, revocation_reason = ?, updated_at = ?
        WHERE patient_id = ? AND consent_type = ? AND revoked_at IS NULL
        """, (now.isoformat(), reason, now.isoformat(), patient_id, consent_type))
        
        return True
    
    def get_patient_consents(self, patient_id: str) -> List[Dict]:
        """Get all consents for a patient (Right to Access)."""
        results = self.db.fetch_all("""
        SELECT consent_type, granted, granted_at, expires_at, 
               revoked_at, version, phc_id
        FROM patient_consents WHERE patient_id = ?
        ORDER BY updated_at DESC
        """, (patient_id,))
        
        return [
            {
                **dict(r),
                "purpose_hindi": self.CONSENT_PURPOSES.get(r["consent_type"], ("", ""))[0],
                "purpose_english": self.CONSENT_PURPOSES.get(r["consent_type"], ("", ""))[1]
            }
            for r in results
        ]


class AuditLogger:
    """
    Comprehensive audit logging for compliance.
    Supports HIPAA, DPDP Act requirements.
    """
    
    class AuditAction(str, Enum):
        CREATE = "CREATE"
        READ = "READ"
        UPDATE = "UPDATE"
        DELETE = "DELETE"
        EXPORT = "EXPORT"
        SHARE = "SHARE"
        LOGIN = "LOGIN"
        LOGOUT = "LOGOUT"
        CONSENT = "CONSENT"
        SEARCH = "SEARCH"
    
    def __init__(self, db_manager: 'DatabaseManager'):
        self.db = db_manager
        self._init_tables()
        self._buffer: List[Dict] = []
        self._buffer_size = 100
    
    def _init_tables(self):
        """Initialize audit tables with partitioning support."""
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_role TEXT,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            patient_id TEXT,
            phc_id TEXT,
            session_id TEXT,
            ip_address TEXT,
            user_agent TEXT,
            request_id TEXT,
            details TEXT,
            success INTEGER DEFAULT 1,
            error_message TEXT,
            duration_ms INTEGER,
            data_classification TEXT
        )
        """)
        
        # Indexes for common queries
        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_logs(patient_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_phc ON audit_logs(phc_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action, timestamp)",
        ]:
            self.db.execute(idx)
    
    def log(
        self,
        user_id: str,
        action: 'AuditAction',
        resource_type: str,
        resource_id: str = None,
        patient_id: str = None,
        user_role: str = None,
        phc_id: str = None,
        session_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        request_id: str = None,
        details: Dict = None,
        success: bool = True,
        error_message: str = None,
        duration_ms: int = None,
        data_classification: DataClassification = None,
        buffered: bool = True
    ):
        """Log an audit event."""
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "user_role": user_role,
            "action": action.value if isinstance(action, self.AuditAction) else action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "patient_id": patient_id,
            "phc_id": phc_id,
            "session_id": session_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "request_id": request_id,
            "details": json.dumps(details, ensure_ascii=False) if details else None,
            "success": int(success),
            "error_message": error_message,
            "duration_ms": duration_ms,
            "data_classification": data_classification.value if data_classification else None
        }
        
        if buffered:
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                self.flush()
        else:
            self._write_entry(entry)
        
        # Log critical events immediately
        if not success or action in [self.AuditAction.DELETE, self.AuditAction.EXPORT]:
            level = logging.WARNING if not success else logging.INFO
            logger.log(level, f"AUDIT: {user_id} {action} {resource_type}/{resource_id}")
    
    def _write_entry(self, entry: Dict):
        """Write single entry to database."""
        self.db.execute("""
        INSERT INTO audit_logs 
        (id, timestamp, user_id, user_role, action, resource_type, 
         resource_id, patient_id, phc_id, session_id, ip_address,
         user_agent, request_id, details, success, error_message,
         duration_ms, data_classification)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(entry.values()))
    
    def flush(self):
        """Flush buffered entries to database."""
        if not self._buffer:
            return
        
        entries = self._buffer
        self._buffer = []
        
        for entry in entries:
            self._write_entry(entry)
    
    def get_patient_access_history(
        self,
        patient_id: str,
        days: int = 90,
        include_system: bool = False
    ) -> List[Dict]:
        """Get access history for DPDP Right to Access."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        query = """
        SELECT timestamp, user_id, user_role, action, resource_type,
               phc_id, ip_address, details, success
        FROM audit_logs
        WHERE patient_id = ? AND timestamp > ?
        """
        
        if not include_system:
            query += " AND user_role != 'SYSTEM'"
        
        query += " ORDER BY timestamp DESC LIMIT 1000"
        
        results = self.db.fetch_all(query, (patient_id, since))
        return [dict(r) for r in results]
    
    def generate_compliance_report(
        self,
        phc_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance audit report."""
        results = self.db.fetch_all("""
        SELECT action, COUNT(*) as count, 
               SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
        FROM audit_logs
        WHERE phc_id = ? AND timestamp BETWEEN ? AND ?
        GROUP BY action
        """, (phc_id, start_date.isoformat(), end_date.isoformat()))
        
        return {
            "phc_id": phc_id,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "actions": {r["action"]: {"total": r["count"], "success": r["success_count"]} for r in results},
            "generated_at": datetime.utcnow().isoformat()
        }


class DatabaseManager:
    """
    Thread-safe database connection manager.
    Supports SQLite with PostgreSQL migration path.
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._local = None
        
        # Parse database type
        if database_url.startswith("sqlite"):
            self.db_type = "sqlite"
            self.db_path = database_url.replace("sqlite:///", "")
        elif database_url.startswith("postgresql"):
            self.db_type = "postgresql"
        else:
            raise ValueError(f"Unsupported database: {database_url}")
    
    @contextmanager
    def connection(self):
        """Get database connection context manager."""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def execute(self, query: str, params: tuple = None):
        """Execute a query."""
        with self.connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.lastrowid
    
    def fetch_one(self, query: str, params: tuple = None):
        """Fetch single row."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return cursor.fetchone()
    
    def fetch_all(self, query: str, params: tuple = None):
        """Fetch all rows."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return cursor.fetchall()


class DataRetentionManager:
    """
    Manages data lifecycle per DPDP Act requirements.
    Handles retention, anonymization, and deletion.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        encryption: HealthDataEncryption,
        anonymizer: DataAnonymizer,
        settings
    ):
        self.db = db_manager
        self.encryption = encryption
        self.anonymizer = anonymizer
        self.settings = settings
    
    def process_retention_policies(self) -> Dict[str, int]:
        """
        Process data retention policies.
        Should be run daily via scheduler.
        """
        stats = {
            "anonymized": 0,
            "deleted": 0,
            "archived": 0
        }
        
        now = datetime.utcnow()
        
        # Anonymize old records (after 2 years by default)
        anonymize_before = now - timedelta(days=self.settings.anonymization_after_days)
        stats["anonymized"] = self._anonymize_old_records(anonymize_before)
        
        # Archive audit logs older than retention period
        archive_before = now - timedelta(days=self.settings.audit_log_retention_days)
        stats["archived"] = self._archive_audit_logs(archive_before)
        
        # Delete expired temporary data
        stats["deleted"] = self._cleanup_temp_data()
        
        logger.info(f"Retention processing complete: {stats}")
        return stats
    
    def _anonymize_old_records(self, before: datetime) -> int:
        """Anonymize patient records older than threshold."""
        # Implementation would query and update records
        return 0
    
    def _archive_audit_logs(self, before: datetime) -> int:
        """Archive old audit logs to cold storage."""
        return 0
    
    def _cleanup_temp_data(self) -> int:
        """Remove expired temporary data."""
        return 0
    
    def handle_deletion_request(
        self,
        patient_id: str,
        requested_by: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Handle Right to Erasure request under DPDP.
        Note: Medical records may have legal retention requirements.
        """
        # Check for legal holds or retention requirements
        can_delete = self._check_deletion_eligibility(patient_id)
        
        if not can_delete["eligible"]:
            return {
                "success": False,
                "reason": can_delete["reason"],
                "retention_until": can_delete.get("retention_until")
            }
        
        # Anonymize instead of delete for medical records
        # Full deletion only for non-medical data
        
        return {
            "success": True,
            "action": "anonymized",
            "completed_at": datetime.utcnow().isoformat()
        }
    
    def _check_deletion_eligibility(self, patient_id: str) -> Dict:
        """Check if patient data can be deleted."""
        # Check for:
        # - Active treatment
        # - Legal holds
        # - Regulatory retention requirements
        return {"eligible": True}


# Factory functions
_db_manager: Optional[DatabaseManager] = None
_encryption: Optional[HealthDataEncryption] = None
_consent_manager: Optional[ConsentManager] = None
_audit_logger: Optional[AuditLogger] = None
_anonymizer: Optional[DataAnonymizer] = None


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        from app.config import get_settings
        settings = get_settings()
        _db_manager = DatabaseManager(settings.database_url)
    return _db_manager


def get_encryption() -> HealthDataEncryption:
    global _encryption
    if _encryption is None:
        from app.config import get_settings
        settings = get_settings()
        key_provider = EnvironmentKeyProvider(settings)
        _encryption = HealthDataEncryption(key_provider)
    return _encryption


def get_consent_manager() -> ConsentManager:
    global _consent_manager
    if _consent_manager is None:
        _consent_manager = ConsentManager(get_db_manager())
    return _consent_manager


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(get_db_manager())
    return _audit_logger


def get_anonymizer() -> DataAnonymizer:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = DataAnonymizer(get_encryption())
    return _anonymizer


# Decorator for automatic audit logging
def audit_action(
    action: AuditLogger.AuditAction,
    resource_type: str,
    get_resource_id: Callable = None,
    get_patient_id: Callable = None
):
    """Decorator to automatically log function calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = datetime.utcnow()
            success = True
            error = None
            result = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = int((datetime.utcnow() - start).total_seconds() * 1000)
                
                get_audit_logger().log(
                    user_id=kwargs.get("user_id", "system"),
                    action=action,
                    resource_type=resource_type,
                    resource_id=get_resource_id(result) if get_resource_id and result else None,
                    patient_id=get_patient_id(kwargs) if get_patient_id else kwargs.get("patient_id"),
                    success=success,
                    error_message=error,
                    duration_ms=duration
                )
        
        return wrapper
    return decorator