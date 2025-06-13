# 🔧 Development Guide - MCP Video Analysis Server

This guide is for developers working on or extending the MCP Video Analysis Server.

## 🏗️ Architecture Overview

### Core Components

1. **Storage Manager** (`src/storage/manager.py`)
   - SQLite database for metadata
   - File system for video/frame storage
   - Search indexing
   - Cleanup operations

2. **Video Processor** (`src/processors/video.py`)
   - Frame extraction using OpenCV
   - Scene detection algorithms
   - Audio extraction with ffmpeg
   - Whisper integration for transcription

3. **LLM Client** (`src/llm/ollama_client.py`)
   - Ollama API integration
   - Vision model support (LLaVA)
   - Text generation (Llama2)
   - Batch processing for efficiency

4. **MCP Tools** (`src/tools/mcp_tools.py`)
   - Tool definitions for Claude Desktop
   - Request/response handling
   - Error management

## 🧪 Testing Strategy

### Unit Tests

Located in `tests/unit/`, covering:
- Storage operations
- Video processing logic
- LLM client functionality
- Individual tool operations

Run unit tests:
```bash
pytest tests/unit/ -v
```

### Integration Tests

Located in `tests/integration/`, covering:
- Full video processing pipeline
- LLM integration (requires Ollama)
- Search functionality
- Error scenarios

Run integration tests:
```bash
# Start Ollama first
ollama serve

# Run tests
pytest tests/integration/ -v
```

### Test Coverage

Generate coverage report:
```bash
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html
```

## 🔨 Adding New Features

### 1. Adding a New MCP Tool

```python
# In src/tools/mcp_tools.py
@mcp.tool()
async def your_new_tool(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """
    Tool description for Claude.
    
    Args:
        param1: Description
        param2: Optional description
    
    Returns:
        Result dictionary
    """
    try:
        # Implementation
        result = await some_operation(param1, param2)
        return {"status": "success", "data": result}
    except Exception as e:
        self.logger.error(f"Error in tool: {e}")
        return {"error": str(e)}
```

### 2. Adding Video Processing Features

```python
# In src/processors/video.py
async def extract_motion_vectors(self, video_path: Path) -> List[MotionData]:
    """Extract motion information from video."""
    # Implementation using OpenCV or other libraries
    pass
```

### 3. Extending Storage Capabilities

```python
# In src/storage/manager.py
def store_video_embeddings(self, video_id: str, embeddings: np.ndarray):
    """Store video embeddings for similarity search."""
    # Implementation
    pass
```

## 🐞 Debugging

### Enable Debug Logging

1. In `config/default_config.json`:
```json
{
  "logging": {
    "level": "DEBUG",
    "format": "rich"  // For colored output
  }
}
```

2. Environment variable:
```bash
export LOG_LEVEL=DEBUG
python server.py
```

### Common Issues

1. **Import Errors**
   - Ensure PYTHONPATH includes project root
   - Check virtual environment activation

2. **Async Errors**
   - Use `pytest-asyncio` for async tests
   - Properly await all async operations

3. **Database Locks**
   - Use context managers for DB access
   - Implement proper connection pooling

## 🚀 Performance Optimization

### Frame Processing

1. **Batch Processing**
   ```python
   # Process frames in batches
   batch_size = 10
   for i in range(0, len(frames), batch_size):
       batch = frames[i:i+batch_size]
       await process_batch(batch)
   ```

2. **Concurrent Analysis**
   ```python
   # Use asyncio for concurrent operations
   tasks = [analyze_frame(f) for f in frames]
   results = await asyncio.gather(*tasks)
   ```

### Database Optimization

1. **Indexes**: Already created for common queries
2. **Batch Inserts**: Use executemany for bulk operations
3. **Connection Pooling**: Reuse connections

### Memory Management

1. **Stream Processing**: Process video in chunks
2. **Cleanup**: Delete temporary files immediately
3. **Frame Resizing**: Reduce resolution before analysis

## 🔌 Integration Points

### Adding New LLM Providers

```python
# Create new client in src/llm/
class NewLLMClient:
    async def analyze_image(self, image_path: Path, prompt: str) -> str:
        # Implementation
        pass
```

### Supporting New Video Formats

1. Update `_get_video_path()` in processor
2. Add format validation
3. Test with sample files

### External Services

1. Cloud storage (S3, GCS)
2. External transcription services
3. Video streaming protocols

## 📦 Dependency Management

### Adding Dependencies

1. Add to `requirements.txt`
2. Document why it's needed
3. Check license compatibility
4. Update setup instructions

### Updating Dependencies

```bash
# Check outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package_name

# Update requirements.txt
pip freeze > requirements.txt
```

## 🔐 Security Considerations

1. **Input Validation**
   - Sanitize file paths
   - Validate video formats
   - Check file sizes

2. **Resource Limits**
   - Max video duration
   - Max file size
   - Processing timeouts

3. **Data Privacy**
   - Local processing only
   - No external API calls for video data
   - Secure file permissions

## 📈 Monitoring and Metrics

### Performance Metrics

```python
# Add timing decorators
from functools import wraps
import time

def measure_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper
```

### Health Checks

```python
# Add health check endpoint
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Check system health."""
    return {
        "ollama": await check_ollama_status(),
        "storage": check_storage_space(),
        "database": check_db_connection()
    }
```

## 🎯 Future Enhancements

### Planned Features

1. **Real-time Processing**
   - Stream processing support
   - Live video analysis
   - WebRTC integration

2. **Advanced Analysis**
   - Object tracking
   - Face detection (privacy-preserving)
   - Action recognition

3. **Better Search**
   - Vector similarity search
   - Semantic search with embeddings
   - Time-based queries

### Architecture Improvements

1. **Microservices**
   - Separate processing service
   - API gateway
   - Message queue integration

2. **Scalability**
   - Distributed processing
   - Cloud deployment options
   - Kubernetes support

## 📝 Code Style Guide

1. **Type Hints**: Always use type hints
2. **Docstrings**: Google style docstrings
3. **Async/Await**: Prefer async for I/O operations
4. **Error Handling**: Explicit try/except blocks
5. **Logging**: Use structured logging

### Example:

```python
async def process_video_segment(
    video_id: str,
    start_time: float,
    end_time: float
) -> ProcessingResult:
    """
    Process a segment of video.
    
    Args:
        video_id: Unique video identifier
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        
    Returns:
        ProcessingResult with segment data
        
    Raises:
        ValueError: If timestamps are invalid
        ProcessingError: If processing fails
    """
    logger.info(f"Processing segment {start_time}-{end_time} for {video_id}")
    
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Failed to process segment: {e}")
        raise ProcessingError(f"Segment processing failed: {e}")
```

---

**Last Updated**: December 2024
**Maintainers**: Development Team