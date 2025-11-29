"""Enhanced Configuration with validation, secrets management, and feature flags"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, SecretStr, model_validator
from functools import lru_cache
from typing import List, Optional, Dict, Any, Set
from enum import Enum
import os
import json


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"


class DatabaseType(str, Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class Settings(BaseSettings):
    """Application settings with comprehensive validation"""

    # Application Identity
    app_name: str = "GHIA - Gramin Health Intake Assistant"
    app_version: str = "3.0.0"
    app_instance_id: str = Field(default_factory=lambda: os.urandom(8).hex())
    environment: Environment = Environment.DEVELOPMENT

    @property
    def debug(self) -> bool:
        return self.environment in [Environment.DEVELOPMENT, Environment.TESTING]

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    # LLM Configuration
    primary_llm_provider: LLMProvider = LLMProvider.GROQ
    fallback_llm_providers: List[LLMProvider] = Field(
        default_factory=lambda: [LLMProvider.GEMINI]
    )

    groq_api_key: Optional[SecretStr] = None
    gemini_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    azure_openai_api_key: Optional[SecretStr] = None
    azure_openai_endpoint: Optional[str] = None

    # LiveKit Configuration
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None

    # Model configuration per provider
    llm_models: Dict[str, str] = Field(
        default_factory=lambda: {
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-1.5-flash",
            "openai": "gpt-4o-mini",
            "azure_openai": "gpt-4o",
        }
    )

    # LLM Parameters
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=2048, ge=100, le=16384)
    llm_timeout_seconds: int = Field(default=30, ge=5, le=120)
    llm_max_retries: int = Field(default=3, ge=1, le=10)
    llm_retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)

    # ASR Configuration
    default_language: str = "hi"
    supported_languages: List[str] = Field(
        default_factory=lambda: [
            "hi",
            "bn",
            "gu",
            "mr",
            "ta",
            "te",
            "kn",
            "ml",
            "pa",
            "or",
            "as",
            "ne",
            "en",
        ]
    )
    asr_provider: str = "groq"  # groq, openai, local
    asr_model: str = "whisper-large-v3"
    asr_fallback_enabled: bool = True
    asr_max_duration_seconds: int = Field(default=300, ge=10, le=600)

    # Database Configuration
    database_type: DatabaseType = DatabaseType.SQLITE
    database_url: str = Field(default="sqlite:///./data/ghia.db")
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_max_overflow: int = Field(default=10, ge=0, le=100)
    database_pool_timeout: int = Field(default=30, ge=5, le=120)
    database_echo: bool = False

    # Redis Configuration
    redis_enabled: bool = False
    redis_url: Optional[str] = None
    redis_session_ttl_hours: int = Field(default=24, ge=1, le=168)
    redis_cache_ttl_minutes: int = Field(default=60, ge=1, le=1440)

    # Security Configuration
    encryption_key: Optional[SecretStr] = None
    jwt_secret_key: SecretStr = SecretStr("CHANGE_THIS_IN_PRODUCTION")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=480, ge=30, le=1440)
    jwt_refresh_token_expire_days: int = Field(default=7, ge=1, le=30)

    # Password Policy
    password_min_length: int = Field(default=12, ge=8, le=128)
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True

    # Session Security
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "lax"

    # Data Retention (DPDP Compliance)
    phi_retention_days: int = Field(default=365 * 7, ge=365)  # 7 years
    audit_log_retention_days: int = Field(default=365 * 10, ge=365)  # 10 years
    anonymization_after_days: int = Field(default=365 * 2, ge=30)  # 2 years
    session_data_retention_hours: int = Field(default=48, ge=1)
    temp_file_retention_hours: int = Field(default=24, ge=1)

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    rate_limit_burst: int = Field(default=10, ge=1, le=100)
    rate_limit_by_ip: bool = True
    rate_limit_by_user: bool = True

    # ABDM Integration
    abdm_enabled: bool = False
    abdm_client_id: Optional[str] = None
    abdm_client_secret: Optional[SecretStr] = None
    abdm_environment: str = "sandbox"  # sandbox, production
    abdm_callback_url: Optional[str] = None

    @property
    def abdm_base_url(self) -> str:
        return {
            "sandbox": "https://dev.abdm.gov.in",
            "production": "https://abdm.gov.in",
        }.get(self.abdm_environment, "https://dev.abdm.gov.in")

    # PHC Configuration
    default_phc_id: str = "PHC-DEFAULT"
    phc_state: str = "MP"
    phc_district: str = "Bhopal"
    phc_block: Optional[str] = None
    phc_name: Optional[str] = None

    # Analytics & Monitoring
    enable_analytics: bool = True
    enable_telemetry: bool = True
    enable_performance_monitoring: bool = True
    outbreak_alert_threshold: int = Field(default=5, ge=1)
    alert_webhook_url: Optional[str] = None
    alert_email_recipients: List[str] = Field(default_factory=list)

    # Sentry/Error Tracking
    sentry_dsn: Optional[str] = None
    sentry_traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # Feature Flags
    features: Dict[str, bool] = Field(
        default_factory=lambda: {
            "follow_up_agent": True,
            "drug_interaction_check": True,
            "differential_diagnosis": True,
            "voice_feedback": False,
            "multi_turn_conversation": True,
            "symptom_checker": True,
            "emergency_detection": True,
            "appointment_scheduling": False,
            "telemedicine": False,
            "lab_integration": False,
            "pharmacy_integration": False,
        }
    )

    @property
    def enable_differential_diagnosis(self) -> bool:
        """Check if differential diagnosis feature is enabled"""
        return self.features.get("differential_diagnosis", True)

    @property
    def enable_drug_interaction_check(self) -> bool:
        """Check if drug interaction check feature is enabled"""
        return self.features.get("drug_interaction_check", True)

    @property
    def enable_follow_up_agent(self) -> bool:
        """Check if follow-up agent feature is enabled"""
        return self.features.get("follow_up_agent", True)

    @property
    def enable_emergency_detection(self) -> bool:
        """Check if emergency detection feature is enabled"""
        return self.features.get("emergency_detection", True)

    # Default PHC ID
    default_phc_id: str = "PHC001"

    def is_feature_enabled(self, feature: str) -> bool:
        return self.features.get(feature, False)

    # Agent Configuration
    agent_memory_type: str = "buffer"  # buffer, summary, vector
    agent_memory_window: int = Field(default=10, ge=1, le=50)
    agent_max_iterations: int = Field(default=10, ge=1, le=50)
    agent_verbose: bool = False

    # Clinical Decision Support
    clinical_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_differential_diagnoses: int = Field(default=5, ge=1, le=10)
    enable_clinical_guidelines: bool = True

    # File Upload
    max_upload_size_mb: int = Field(default=10, ge=1, le=100)
    allowed_audio_formats: List[str] = Field(
        default_factory=lambda: ["wav", "mp3", "m4a", "ogg", "webm"]
    )
    allowed_image_formats: List[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"]
    )
    temp_upload_dir: str = "./data/temp"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json, text
    log_file_path: Optional[str] = "./logs/ghia.log"
    log_rotation_size_mb: int = Field(default=10, ge=1, le=100)
    log_retention_days: int = Field(default=30, ge=1, le=365)

    # Validators
    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key(cls, v, info):
        if info.data.get("environment") == Environment.PRODUCTION:
            if not v:
                raise ValueError("Encryption key required in production")
            if len(v.get_secret_value()) < 32:
                raise ValueError("Encryption key must be at least 32 characters")
        return v

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v, info):
        if info.data.get("environment") == Environment.PRODUCTION:
            secret = v.get_secret_value()
            if secret == "CHANGE_THIS_IN_PRODUCTION":
                raise ValueError("JWT secret must be changed in production")
            if len(secret) < 32:
                raise ValueError("JWT secret must be at least 32 characters")
        return v

    @model_validator(mode="after")
    def validate_llm_keys(self):
        """Ensure API key exists for configured providers"""
        provider = self.primary_llm_provider
        key_map = {
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.AZURE_OPENAI: self.azure_openai_api_key,
        }

        if provider in key_map and not key_map.get(provider):
            if self.environment != Environment.TESTING:
                raise ValueError(f"API key required for {provider.value}")

        return self

    # Helper methods
    def get_llm_model(self, provider: LLMProvider = None) -> str:
        provider = provider or self.primary_llm_provider
        return self.llm_models.get(provider.value, self.llm_models["groq"])

    def get_api_key(self, provider: LLMProvider) -> Optional[str]:
        key_map = {
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.AZURE_OPENAI: self.azure_openai_api_key,
        }
        secret = key_map.get(provider)
        return secret.get_secret_value() if secret else None

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for SQLAlchemy."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "pool_timeout": self.database_pool_timeout,
            "echo": self.database_echo,
        }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
        "env_nested_delimiter": "__",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache():
    """Clear cached settings (for testing)."""
    get_settings.cache_clear()


def get_feature_flags() -> Dict[str, bool]:
    """Get all feature flags."""
    return get_settings().features.copy()
