"""
Configuration management - Load settings from environment variables.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider."""
    api_key: str
    base_url: str
    model: str
    
    def is_valid(self) -> bool:
        """Check if the configuration has required fields."""
        return bool(self.api_key and self.base_url and self.model)


@dataclass
class AgentConfig:
    """Configuration for the Agent."""
    max_iterations: int = 10
    execution_timeout: int = 30
    sandbox_enabled: bool = True
    workspace_dir: str = "./workspace"
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class that loads all settings from environment."""
    
    deepseek: LLMConfig = field(default_factory=lambda: LLMConfig("", "", ""))
    openai: LLMConfig = field(default_factory=lambda: LLMConfig("", "", ""))
    anthropic: LLMConfig = field(default_factory=lambda: LLMConfig("", "", ""))
    default_provider: str = "deepseek"
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    _instance: Optional["Config"] = None
    
    @classmethod
    def load(cls, env_path: Optional[str] = None) -> "Config":
        """
        Load configuration from environment variables.
        
        Args:
            env_path: Optional path to .env file. If not provided,
                     looks for .env in current directory and parent directories.
        """
        # Load .env file
        if env_path:
            load_dotenv(env_path)
        else:
            # Try to find .env file
            current = Path.cwd()
            while current != current.parent:
                env_file = current / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    break
                current = current.parent
            else:
                # Also try to load from default location
                load_dotenv()
        
        config = cls(
            deepseek=LLMConfig(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            ),
            openai=LLMConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            ),
            anthropic=LLMConfig(
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            ),
            default_provider=os.getenv("DEFAULT_LLM_PROVIDER", "deepseek"),
            agent=AgentConfig(
                max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
                execution_timeout=int(os.getenv("EXECUTION_TIMEOUT", "30")),
                sandbox_enabled=os.getenv("SANDBOX_ENABLED", "true").lower() == "true",
                workspace_dir=os.getenv("WORKSPACE_DIR", "./workspace"),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            ),
        )
        
        cls._instance = config
        return config
    
    @classmethod
    def get_instance(cls) -> "Config":
        """Get the singleton config instance, loading if necessary."""
        if cls._instance is None:
            cls.load()
        return cls._instance
    
    def get_llm_config(self, provider: Optional[str] = None) -> LLMConfig:
        """
        Get LLM configuration for a specific provider.
        
        Args:
            provider: Provider name (deepseek, openai, anthropic).
                     If not provided, uses default_provider.
        """
        provider = provider or self.default_provider
        
        provider_map = {
            "deepseek": self.deepseek,
            "openai": self.openai,
            "anthropic": self.anthropic,
        }
        
        if provider not in provider_map:
            raise ValueError(f"Unknown provider: {provider}. "
                           f"Available: {list(provider_map.keys())}")
        
        config = provider_map[provider]
        
        if not config.is_valid():
            raise ValueError(f"Invalid configuration for provider '{provider}'. "
                           f"Please check your .env file.")
        
        return config
    
    def ensure_workspace(self) -> Path:
        """Ensure workspace directory exists and return its path."""
        workspace = Path(self.agent.workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace


# Convenience function
def get_config() -> Config:
    """Get the global configuration instance."""
    return Config.get_instance()
