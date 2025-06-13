# 🚀 MCP Video Analysis Server - Setup Instructions

This document contains the complete setup and usage instructions for the MCP Video Analysis Server. Keep this document updated as the project evolves.

## 📋 Prerequisites

Before setting up the server, ensure you have:

1. **Python 3.9 or higher**
   ```bash
   python --version  # Should show 3.9+
   ```

2. **FFmpeg** (for video processing)
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html and add to PATH
   ```

3. **Ollama** (for local LLM inference)
   - Download from [ollama.ai](https://ollama.ai)
   - After installation, start Ollama:
     ```bash
     ollama serve
     ```

## 🛠️ Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mcp-video-server
cd mcp-video-server
```

### 2. Run the Setup Script

The easiest way to set up everything:

```bash
./setup.py
```

This script will:
- ✅ Check Python version
- ✅ Verify ffmpeg installation
- ✅ Install Python dependencies
- ✅ Create necessary directories
- ✅ Check Ollama status
- ✅ Optionally configure Claude Desktop

### 3. Install Ollama Models

If the setup script reports missing models:

```bash
# Pull required models
ollama pull llava    # Vision model for frame analysis
ollama pull llama2   # Text model for Q&A and summaries
```

### 4. Configure Claude Desktop (Optional)

If you want to use with Claude Desktop:

```bash
python scripts/setup_claude_desktop.py
```

This will automatically add the server to your Claude Desktop configuration.

**Note**: Restart Claude Desktop after configuration.

## 🎬 Usage Guide

### Using with Claude Desktop

Once configured, you can use natural language commands in Claude:

```
"Process the video at /path/to/my/video.mp4"
"What happens in video vid_abc123?"
"Search for videos where someone mentions 'quarterly results'"
"Give me a detailed summary of video vid_xyz789"
"What's happening at the 2 minute mark in video vid_abc123?"
"List all videos in the system"
"Show me statistics about video storage"
```

### Using the Standalone CLI

The CLI provides direct access to all features:

#### Process a Video
```bash
./standalone_client/video_client.py process /path/to/video.mp4
```

#### Ask Questions
```bash
./standalone_client/video_client.py ask vid_abc123 "What is the main topic discussed?"
```

#### Search Videos
```bash
./standalone_client/video_client.py search "meeting"
```

#### List All Videos
```bash
./standalone_client/video_client.py list
```

#### Get Video Summary
```bash
./standalone_client/video_client.py summary vid_abc123
```

#### Check System Status
```bash
./standalone_client/video_client.py status
```

## 🧪 Testing the Installation

### 1. Test with Sample Video

A sample video is included for testing:

```bash
# Process the sample video
./standalone_client/video_client.py process video_data/originals/sample_video.mp4

# Ask a question about it (replace vid_xxx with the actual ID returned)
./standalone_client/video_client.py ask vid_xxx "What is shown in this video?"
```

### 2. Run Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Run tests
pytest tests/unit/
```

### 3. Run Integration Tests

```bash
# Requires Ollama to be running
pytest tests/integration/
```

## 🔧 Configuration

### Default Configuration

The server uses `config/default_config.json`. Key settings:

```json
{
  "processing": {
    "frame_sample_rate": 30,      // Extract 1 frame every 30 frames
    "max_frames_per_video": 1000, // Limit frames per video
    "enable_scene_detection": true // Smart frame selection
  },
  "llm": {
    "vision_model": "llava:latest",
    "text_model": "llama2:latest",
    "temperature": 0.7
  }
}
```

### Custom Configuration

Create `config/config.json` to override defaults:

```json
{
  "processing": {
    "frame_sample_rate": 60  // Less frequent sampling for longer videos
  }
}
```

## 🐛 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Check logs
journalctl -u ollama  # Linux with systemd
```

### FFmpeg Not Found

```bash
# Verify installation
which ffmpeg
ffmpeg -version

# Add to PATH if needed
export PATH="/path/to/ffmpeg/bin:$PATH"
```

### Python Import Errors

```bash
# Ensure you're in the project directory
cd /path/to/mcp-video-server

# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Claude Desktop Not Detecting Server

1. Check the configuration file:
   ```bash
   # macOS
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   
   # Windows
   type %APPDATA%\Claude\claude_desktop_config.json
   
   # Linux
   cat ~/.config/Claude/claude_desktop_config.json
   ```

2. Verify the server path is correct
3. Restart Claude Desktop
4. Check server logs: `tail -f logs/mcp-video-server.log`

### Video Processing Fails

1. Check available disk space
2. Verify video file is not corrupted:
   ```bash
   ffprobe /path/to/video.mp4
   ```
3. Try with a smaller sample rate:
   ```bash
   ./standalone_client/video_client.py process video.mp4 --sample-rate 60
   ```

## 📊 Performance Tips

### For Large Videos

1. Adjust frame sampling rate in config
2. Increase max_frames_per_video limit if needed
3. Use scene detection for better frame selection
4. Process during off-hours for better performance

### For Many Videos

1. Use the search feature instead of listing all
2. Implement cleanup for old videos:
   ```python
   # In Python
   from src.storage.manager import StorageManager
   storage = StorageManager()
   storage.cleanup_old_files(days=30)
   ```

## 🔄 Updating the Project

### Pull Latest Changes

```bash
git pull origin main
pip install -r requirements.txt
```

### Update Ollama Models

```bash
ollama pull llava:latest
ollama pull llama2:latest
```

### Migrate Database (if needed)

Database migrations are handled automatically on startup.

## 📝 Development Notes

### Running the MCP Server Directly

```bash
# For development/debugging
python server.py
```

### Adding New Features

1. Create feature branch
2. Add tests in `tests/`
3. Update documentation
4. Submit pull request

### Project Structure Overview

- `src/processors/` - Video processing logic
- `src/storage/` - Database and file management
- `src/llm/` - Ollama integration
- `src/tools/` - MCP tool definitions
- `src/utils/` - Shared utilities

## 🆘 Getting Help

1. Check the logs: `logs/mcp-video-server.log`
2. Run system status: `./standalone_client/video_client.py status`
3. Check GitHub issues
4. Review test files for usage examples

## 📅 Maintenance Schedule

- **Daily**: Check disk usage for video storage
- **Weekly**: Review and clean old processed videos
- **Monthly**: Update Ollama models
- **As needed**: Update Python dependencies

---

**Last Updated**: December 2024
**Version**: 0.1.1

Remember to keep this document updated as the project evolves!