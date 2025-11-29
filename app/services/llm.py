"""LLM Service with provider abstraction and fallback support"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings, LLMProvider

logger = logging.getLogger(__name__)


class LLMService(ABC):
    """Abstract LLM service interface"""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        pass
    
    @abstractmethod
    async def invoke(
        self,
        messages: List[Any],
        system_prompt: str = None,
        temperature: float = 0.3
    ) -> str:
        """LangChain-compatible invoke method"""
        pass


class GroqLLMService(LLMService):
    """Groq LLM implementation"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        client = self._get_client()
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def invoke(
        self,
        messages: List[Any],
        system_prompt: str = None,
        temperature: float = 0.3
    ) -> str:
        """LangChain-compatible invoke method"""
        formatted_messages = []
        
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})
        
        return await self.generate(formatted_messages, temperature=temperature)


class GeminiLLMService(LLMService):
    """Google Gemini implementation"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        client = self._get_client()
        
        # Convert messages to Gemini format
        prompt = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
        ])
        
        response = await asyncio.to_thread(
            client.generate_content,
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        
        return response.text
    
    async def invoke(
        self,
        messages: List[Any],
        system_prompt: str = None,
        temperature: float = 0.3
    ) -> str:
        """LangChain-compatible invoke method"""
        formatted_messages = []
        
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})
        
        return await self.generate(formatted_messages, temperature=temperature)


class FallbackLLMService(LLMService):
    """LLM service with fallback support"""
    
    def __init__(self, primary: LLMService, fallbacks: List[LLMService] = None):
        self.primary = primary
        self.fallbacks = fallbacks or []
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        services = [self.primary] + self.fallbacks
        last_error = None
        
        for i, service in enumerate(services):
            try:
                return await service.generate(messages, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"LLM service {i} failed: {e}")
                last_error = e
                continue
        
        raise Exception(f"All LLM services failed. Last error: {last_error}")
    
    async def invoke(
        self,
        messages: List[Any],
        system_prompt: str = None,
        temperature: float = 0.3
    ) -> str:
        """LangChain-compatible invoke method with fallback"""
        services = [self.primary] + self.fallbacks
        last_error = None
        
        for i, service in enumerate(services):
            try:
                return await service.invoke(messages, system_prompt, temperature)
            except Exception as e:
                logger.warning(f"LLM service {i} invoke failed: {e}")
                last_error = e
                continue
        
        raise Exception(f"All LLM services failed. Last error: {last_error}")


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get configured LLM service singleton."""
    global _llm_service
    
    if _llm_service is None:
        settings = get_settings()
        
        services = []
        
        # Primary provider
        if settings.groq_api_key:
            services.append(GroqLLMService(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.llm_models.get("groq", "llama-3.1-70b-versatile")
            ))
        
        if settings.gemini_api_key:
            services.append(GeminiLLMService(
                api_key=settings.gemini_api_key.get_secret_value(),
                model=settings.llm_models.get("gemini", "gemini-1.5-flash")
            ))
        
        if not services:
            raise ValueError("No LLM API keys configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env")
        
        if len(services) == 1:
            _llm_service = services[0]
        else:
            _llm_service = FallbackLLMService(services[0], services[1:])
    
    return _llm_service