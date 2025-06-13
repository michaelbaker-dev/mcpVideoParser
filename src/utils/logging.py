"""Logging configuration and utilities."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from pythonjsonlogger import jsonlogger
from rich.logging import RichHandler
from rich.console import Console


class VideoServerLogger:
    """Custom logger for the video server."""
    
    def __init__(self, name: str, config: Optional[dict] = None):
        self.logger = logging.getLogger(name)
        self.config = config or {}
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger with appropriate handlers."""
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(self.config.get('level', 'INFO'))
        
        # Console handler with rich formatting for development
        if self.config.get('format', 'json') == 'rich':
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=True
            )
            console_handler.setLevel(self.logger.level)
            self.logger.addHandler(console_handler)
        else:
            # JSON formatter for production
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s',
                rename_fields={'timestamp': '@timestamp'}
            )
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.logger.level)
            self.logger.addHandler(console_handler)
        
        # File handler if configured
        log_file = self.config.get('file')
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_formatter = jsonlogger.JsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(self.logger.level)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


def setup_logging(config: dict) -> None:
    """Set up logging for the entire application."""
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.get('level', 'INFO'))
    
    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add custom handler
    if config.get('format') == 'rich':
        handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            rich_tracebacks=True
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            rename_fields={'timestamp': '@timestamp'}
        )
        handler.setFormatter(formatter)
    
    root_logger.addHandler(handler)
    
    # File logging
    log_file = config.get('file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = jsonlogger.JsonFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)