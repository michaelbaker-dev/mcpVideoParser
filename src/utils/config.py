"""Configuration management for the MCP video server."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import logging


class StorageConfig(BaseModel):
    """Storage configuration."""
    base_path: str = "./video_data"
    max_cache_size_gb: int = 50
    cleanup_after_days: int = 30
    
    @validator('base_path')
    def validate_base_path(cls, v):
        # First check for environment variable
        env_path = os.environ.get('VIDEO_DATA_PATH')
        if env_path:
            path = Path(env_path)
        else:
            # If the path is relative, make it relative to the project root
            path = Path(v)
            if not path.is_absolute():
                # Find the project root (where mcp_video_server.py is located)
                current_file = Path(__file__)  # This is config.py
                project_root = current_file.parent.parent.parent  # Go up to project root
                path = project_root / path
        
        # Ensure the path exists
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class AudioConfig(BaseModel):
    """Audio processing configuration."""
    model: str = "base"
    language: str = "en"
    enable_timestamps: bool = True


class ProcessingConfig(BaseModel):
    """Video processing configuration."""
    frame_sample_rate: int = 30
    enable_scene_detection: bool = True
    scene_threshold: float = 0.3
    max_frames_per_video: int = 1000
    frame_quality: int = 85
    frame_format: str = "jpg"
    audio: AudioConfig = Field(default_factory=AudioConfig)


class LLMConfig(BaseModel):
    """LLM configuration."""
    ollama_host: str = "http://localhost:11434"
    vision_model: str = "llava:latest"
    text_model: str = "llama2:latest"
    embedding_model: str = "nomic-embed-text"
    max_context_length: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 60


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    max_concurrent_videos: int = 2
    frame_batch_size: int = 10
    enable_gpu: bool = True
    thread_count: int = 4


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file: Optional[str] = "logs/mcp-video-server.log"


class Config(BaseModel):
    """Main configuration model."""
    storage: StorageConfig = Field(default_factory=StorageConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[Config] = None
        self.logger = logging.getLogger(__name__)
        
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations."""
        # Get the project root directory
        current_file = Path(__file__)  # This is config.py
        project_root = current_file.parent.parent.parent  # Go up to project root
        
        search_paths = [
            project_root / "config.json",
            project_root / "config" / "config.json",
            project_root / "config" / "default_config.json",
            Path(os.path.expanduser("~/.mcp-video/config.json")),
            Path("/etc/mcp-video/config.json")
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
                
        # Return default config path relative to project root
        return str(project_root / "config" / "default_config.json")
    
    def load(self) -> Config:
        """Load configuration from file."""
        if self._config is not None:
            return self._config
            
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                self._config = Config(**data)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            else:
                self._config = Config()
                self.logger.info("Using default configuration")
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self._config = Config()
            
        # Create necessary directories
        self._create_directories()
        
        return self._config
    
    def _create_directories(self):
        """Create necessary directories based on configuration."""
        if self._config:
            # Storage directories
            base_path = Path(self._config.storage.base_path)
            for subdir in ['originals', 'processed', 'index']:
                (base_path / subdir).mkdir(parents=True, exist_ok=True)
            
            # Log directory
            if self._config.logging.file:
                log_dir = Path(self._config.logging.file).parent
                log_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self) -> Config:
        """Get configuration, loading if necessary."""
        if self._config is None:
            self.load()
        return self._config
    
    def reload(self):
        """Reload configuration from file."""
        self._config = None
        return self.load()
    
    def save(self, config: Config, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            json.dump(config.dict(), f, indent=2)
        self.logger.info(f"Saved configuration to {save_path}")


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    return _config_manager.get()


def reload_config():
    """Reload the global configuration."""
    return _config_manager.reload()