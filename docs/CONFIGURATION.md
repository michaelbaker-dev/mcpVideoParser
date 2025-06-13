# Configuration Guide

## Configuration File

The main configuration file is located at `config/default_config.json`. You can also create environment-specific configs:
- `config/config.json` - Custom configuration (overrides defaults)
- `config/production.json` - Production settings
- `config/development.json` - Development settings

## Configuration Options

### Storage Settings

```json
{
  "storage": {
    "base_path": "./video_data",      // Where to store all video data
    "max_cache_size_gb": 50,          // Maximum cache size before cleanup
    "cleanup_after_days": 30          // Delete old processed data after N days
  }
}
```

**Environment Variables:**
- `VIDEO_DATA_PATH` - Override base_path

### Processing Settings

```json
{
  "processing": {
    "frame_sample_rate": 30,          // Extract 1 frame every N frames
    "enable_scene_detection": true,   // Auto-detect scene changes
    "scene_threshold": 0.3,           // Sensitivity (0.0-1.0, lower = more sensitive)
    "max_frames_per_video": 1000,     // Maximum frames to extract per video
    "frame_quality": 85,              // JPEG quality (1-100)
    "frame_format": "jpg",            // Frame format (jpg, png)
    "audio": {
      "model": "base",                // Whisper model size
      "language": "en",               // Audio language
      "enable_timestamps": true       // Include word timestamps
    }
  }
}
```

### LLM Settings

```json
{
  "llm": {
    "ollama_host": "http://localhost:11434",  // Ollama server URL
    "vision_model": "llava:latest",           // Model for analyzing frames
    "text_model": "llama2:latest",            // Model for text generation
    "embedding_model": "nomic-embed-text",    // Model for embeddings
    "max_context_length": 4096,               // Maximum context window
    "temperature": 0.7,                       // Generation temperature
    "timeout_seconds": 60                     // Request timeout
  }
}
```

**Environment Variables:**
- `OLLAMA_HOST` - Override Ollama server URL

### Performance Settings

```json
{
  "performance": {
    "max_concurrent_videos": 2,       // Process N videos simultaneously
    "frame_batch_size": 10,           // Analyze N frames per batch
    "enable_gpu": true,               // Use GPU acceleration if available
    "thread_count": 4                 // Worker threads for processing
  }
}
```

### Logging Settings

```json
{
  "logging": {
    "level": "INFO",                  // DEBUG, INFO, WARNING, ERROR
    "format": "json",                 // json or text
    "file": "logs/mcp-video-server.log"
  }
}
```

## Configuration Examples

### High-Quality Analysis

For detailed analysis with more frames:

```json
{
  "processing": {
    "frame_sample_rate": 15,
    "max_frames_per_video": 2000,
    "frame_quality": 95,
    "scene_threshold": 0.2
  }
}
```

### Storage Optimization

For limited storage environments:

```json
{
  "storage": {
    "max_cache_size_gb": 10,
    "cleanup_after_days": 7
  },
  "processing": {
    "frame_sample_rate": 60,
    "max_frames_per_video": 500,
    "frame_quality": 70,
    "frame_format": "jpg"
  }
}
```

### Fast Processing

For real-time or near real-time processing:

```json
{
  "processing": {
    "frame_sample_rate": 120,
    "enable_scene_detection": false,
    "max_frames_per_video": 100,
    "audio": {
      "model": "tiny",
      "enable_timestamps": false
    }
  },
  "llm": {
    "timeout_seconds": 30,
    "temperature": 0.5
  }
}
```

### Multi-Camera Setup

For surveillance or multi-camera environments:

```json
{
  "storage": {
    "base_path": "/mnt/surveillance/video_data",
    "max_cache_size_gb": 500,
    "cleanup_after_days": 90
  },
  "performance": {
    "max_concurrent_videos": 4,
    "frame_batch_size": 20,
    "thread_count": 8
  }
}
```

## Location Configuration

Locations are auto-detected from video paths or can be specified during processing. Common patterns:

- `/path/to/shed/video.mp4` → Location: "shed"
- `/cameras/front_door/recording.mp4` → Location: "front_door"
- Custom: Specify with `--location` flag

## Advanced Configuration

### Custom Model Configuration

To use different models:

```bash
# Pull custom models
ollama pull llava:13b
ollama pull mistral:latest

# Update config
{
  "llm": {
    "vision_model": "llava:13b",
    "text_model": "mistral:latest"
  }
}
```

### Network Configuration

For remote Ollama servers:

```json
{
  "llm": {
    "ollama_host": "http://192.168.1.100:11434"
  }
}
```

### Resource Limits

To limit resource usage:

```json
{
  "performance": {
    "max_concurrent_videos": 1,
    "frame_batch_size": 5,
    "enable_gpu": false,
    "thread_count": 2
  },
  "processing": {
    "max_frames_per_video": 200
  }
}
```

## Monitoring Configuration

Check current configuration:

```bash
# View effective configuration
python -c "from src.utils.config import get_config; print(get_config().model_dump_json(indent=2))"

# Test configuration
python scripts/test_config.py
```

## Configuration Validation

The system validates configuration on startup:
- Path existence and permissions
- Model availability
- Ollama connectivity
- Storage space

Invalid configurations will show clear error messages with suggestions for fixes.