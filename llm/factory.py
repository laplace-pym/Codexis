"""
LLM Factory - Create LLM instances based on configuration.
"""

from typing import Optional

from .base import BaseLLM
from .deepseek import DeepSeekLLM
from .openai_adapter import OpenAILLM
from .anthropic_adapter import AnthropicLLM
from utils.config import Config, LLMConfig


class LLMFactory:
    """
    Factory class for creating LLM instances.
    
    Usage:
        # Using config
        llm = LLMFactory.create("deepseek")
        
        # Using explicit configuration
        llm = LLMFactory.create_from_config(llm_config)
    """
    
    _providers = {
        "deepseek": DeepSeekLLM,
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a new LLM provider.
        
        Args:
            name: Provider name (e.g., "custom_llm")
            provider_class: Class that extends BaseLLM
        """
        if not issubclass(provider_class, BaseLLM):
            raise TypeError(f"Provider class must extend BaseLLM")
        cls._providers[name] = provider_class
    
    @classmethod
    def create(cls, provider: Optional[str] = None, config: Optional[Config] = None) -> BaseLLM:
        """
        Create an LLM instance for the specified provider.
        
        Args:
            provider: Provider name. If not specified, uses default from config.
            config: Configuration instance. If not provided, loads from environment.
            
        Returns:
            Configured LLM instance
        """
        if config is None:
            config = Config.get_instance()
        
        provider = provider or config.default_provider
        llm_config = config.get_llm_config(provider)
        
        return cls.create_from_config(provider, llm_config)
    
    @classmethod
    def create_from_config(cls, provider: str, llm_config: LLMConfig) -> BaseLLM:
        """
        Create an LLM instance from explicit configuration.
        
        Args:
            provider: Provider name
            llm_config: LLM configuration object
            
        Returns:
            Configured LLM instance
        """
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown provider: {provider}. Available: {available}")
        
        provider_class = cls._providers[provider]
        return provider_class(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            model=llm_config.model,
        )
    
    @classmethod
    def available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())
