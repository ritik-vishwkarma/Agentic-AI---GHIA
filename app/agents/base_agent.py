"""Base agent classes for GHIA multi-agent system"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")


class AgentStatus(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentContext:
    """Context passed between agents"""
    session_id: str
    patient_id: Optional[str] = None
    user_id: str = "system"
    user_role: str = "health_worker"
    phc_id: str = ""
    language: str = "hi"
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_message(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        return self.conversation_history[-n:]


@dataclass
class AgentResult(Generic[T]):
    """Standardized agent result"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.settings = get_settings()
        self.status = AgentStatus.IDLE
    
    @abstractmethod
    async def execute(self, context: AgentContext, **kwargs) -> AgentResult:
        pass
    
    async def __call__(self, context: AgentContext, **kwargs) -> AgentResult:
        self.status = AgentStatus.PROCESSING
        try:
            result = await self.execute(context, **kwargs)
            self.status = AgentStatus.COMPLETED
            result.metadata["agent"] = self.name
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.exception(f"Agent {self.name} failed: {e}")
            return AgentResult(success=False, error=str(e))


class LLMAgent(BaseAgent):
    """Agent that uses LLM for reasoning"""
    
    def __init__(self, name: str, system_prompt: str = None):
        super().__init__(name)
        self.system_prompt = system_prompt
        self._llm = None
    
    async def _get_llm(self):
        if self._llm is None:
            from app.services.llm import get_llm_service
            self._llm = get_llm_service()
        return self._llm
    
    async def generate(self, prompt: str, context: AgentContext, temperature: float = None) -> str:
        llm = await self._get_llm()
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in context.get_recent_messages():
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        return await llm.generate(
            messages=messages,
            temperature=temperature or self.settings.llm_temperature
        )