# Development Guide

## Setting Up Development Environment

### Prerequisites

1. Python 3.10+
2. Git
3. Ollama
4. ffmpeg

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/michaelbaker-dev/mcpVideoParser.git
cd mcpVideoParser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Project Structure

```
mcp-video-server/
├── src/                    # Main source code
│   ├── llm/               # LLM client implementations
│   │   └── ollama_client.py
│   ├── processors/        # Video processing logic
│   │   └── video.py
│   ├── storage/           # Storage and database
│   │   ├── manager.py
│   │   └── schemas.py
│   ├── tools/             # MCP tool definitions
│   │   └── mcp_tools.py
│   └── utils/             # Utilities
│       ├── config.py
│       ├── date_parser.py
│       └── logging.py
├── standalone_client/      # Client implementations
│   ├── mcp_http_client.py
│   └── video_client.py
├── tests/                  # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── config/                 # Configuration files
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_storage_manager.py -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Integration Tests

Integration tests require Ollama to be running:

```bash
# Start Ollama
ollama serve

# Run integration tests
pytest tests/integration/ -v

# Run specific integration test
pytest tests/integration/test_video_processing_flow.py::TestVideoProcessingFlow::test_complete_video_processing -v
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html  # On macOS
# Or browse to htmlcov/index.html
```

## Code Style

### Linting

```bash
# Run all linters
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Format code
black .

# Type checking
mypy src/
```

### Pre-commit Hooks

Pre-commit hooks run automatically on commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/psf/black
    rev: 23.10.0
    hooks:
      - id: black
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.0
    hooks:
      - id: mypy
```

## Adding New Features

### 1. Adding a New MCP Tool

```python
# src/tools/mcp_tools.py

@mcp.tool()
async def my_new_tool(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """
    Description of what the tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    try:
        # Implementation
        result = await do_something(param1, param2)
        return {"status": "success", "result": result}
    except Exception as e:
        self.logger.error(f"Error in my_new_tool: {e}")
        return {"status": "error", "error": str(e)}
```

### 2. Adding Tests

```python
# tests/unit/test_my_feature.py

import pytest
from unittest.mock import Mock, AsyncMock

class TestMyFeature:
    @pytest.fixture
    def mock_storage(self):
        return Mock()
    
    @pytest.mark.asyncio
    async def test_my_new_tool(self, mock_storage):
        # Arrange
        tool = MyTool(mock_storage)
        
        # Act
        result = await tool.my_new_tool("test", 42)
        
        # Assert
        assert result["status"] == "success"
        assert "result" in result
```

### 3. Adding Configuration Options

```python
# src/utils/config.py

class MyFeatureConfig(BaseModel):
    """Configuration for my feature."""
    enabled: bool = True
    setting1: str = "default"
    setting2: int = 42
    
    class Config:
        extra = "forbid"

# Add to main config
class Config(BaseSettings):
    my_feature: MyFeatureConfig = MyFeatureConfig()
```

## Debugging

### Debug Logging

```python
# Enable debug logging
export LOG_LEVEL=DEBUG

# Or in code
from src.utils.logging import get_logger
logger = get_logger(__name__)
logger.debug("Debug message")
```

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **Async Errors**: Remember to use `await` with async functions
3. **Database Locks**: Only one process should access the database
4. **Memory Issues**: Process videos in batches

## Performance Profiling

```python
# Profile a function
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

## Release Process

1. **Update Version**:
   ```python
   # mcp_video_server.py
   SERVER_VERSION = "0.1.1"
   ```

2. **Update Documentation**:
   - README.md version
   - CHANGELOG.md

3. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Create Tag**:
   ```bash
   git tag -a v0.1.1 -m "Release version 0.1.1"
   git push origin v0.1.1
   ```

## Contributing Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Write tests** for new functionality
4. **Ensure tests pass**: `pytest`
5. **Follow code style**: `ruff check . && black .`
6. **Commit with clear messages**
7. **Push to your fork**
8. **Create a Pull Request**

## Development Tools

### Useful Commands

```bash
# Find all TODO comments
grep -r "TODO" src/

# Check for large files
find . -type f -size +1M

# Profile memory usage
python -m memory_profiler script.py

# Watch for file changes
watchmedo auto-restart --patterns="*.py" --recursive -- python mcp_video_server.py
```

### VS Code Extensions

Recommended extensions for development:
- Python
- Pylance
- Black Formatter
- Ruff
- GitLens
- Python Test Explorer

### Debug Configuration

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug MCP Server",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/mcp_video_server.py",
      "args": ["--http", "--host", "localhost", "--port", "8000"],
      "console": "integratedTerminal"
    }
  ]
}
```