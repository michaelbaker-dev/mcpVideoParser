# Contributing to MCP Video Parser

We love your input! We want to make contributing to MCP Video Parser as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/michaelbaker-dev/mcpVideoParser/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/michaelbaker-dev/mcpVideoParser/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Process

### Setting Up Your Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mcpVideoParser.git
   cd mcpVideoParser
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Making Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and add tests

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Check code style**:
   ```bash
   ruff check .
   black --check .
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

### Submitting a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub

3. **Describe your changes** in the PR description

4. **Link any relevant issues**

## Code Style

- We use [Black](https://github.com/psf/black) for Python code formatting
- We use [Ruff](https://github.com/astral-sh/ruff) for linting
- Follow PEP 8 guidelines
- Write descriptive variable and function names
- Add type hints where possible
- Document your functions with docstrings

### Example Code Style

```python
from typing import Optional, Dict, Any


async def process_video_frame(
    frame_path: str,
    model_name: str = "llava:latest",
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single video frame using the specified model.
    
    Args:
        frame_path: Path to the frame image file
        model_name: Name of the Ollama model to use
        prompt: Optional custom prompt for analysis
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        FileNotFoundError: If frame_path doesn't exist
        ProcessingError: If analysis fails
    """
    # Implementation here
    pass
```

## Testing

- Write tests for any new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Use pytest fixtures for common test setup
- Mock external dependencies (Ollama, file system, etc.)

### Test Structure

```python
import pytest
from unittest.mock import Mock, AsyncMock


class TestVideoProcessor:
    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage manager."""
        return Mock()
    
    @pytest.mark.asyncio
    async def test_process_video_success(self, mock_storage):
        """Test successful video processing."""
        # Arrange
        processor = VideoProcessor(mock_storage)
        
        # Act
        result = await processor.process_video("test_id")
        
        # Assert
        assert result.status == "completed"
```

## Documentation

- Update README.md if you change functionality
- Add docstrings to all public functions and classes
- Update API documentation for new MCP tools
- Include examples in documentation

## Community

- Be respectful and inclusive
- Help others in issues and discussions
- Share your use cases and experiences
- Suggest improvements and features

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## Questions?

Feel free to open an issue with a `question` label or start a discussion!