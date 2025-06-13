# HTTP MCP Setup Guide

## ✅ Current Status

The HTTP MCP video analysis server is **working** with some final integration tweaks needed. Here's what's implemented:

### Working Components:
- ✅ HTTP server starts successfully on any host/port
- ✅ Health endpoint responding at `/health`
- ✅ Client connects and authenticates
- ✅ Server creates transport sessions
- ✅ Proper JSON-RPC message format
- ✅ Process and port management
- ✅ Remote connectivity ready

### Final Issues:
- 🔧 Session ID management for tool calls (MCP protocol requirement)
- 🔧 Tool call routing (400 errors but sessions are created)

## 🚀 Quick Start

### 1. Start HTTP Server
```bash
# Local development
python mcp_video_server.py --http --host localhost --port 8000

# Remote access (from other machines/mobile)
python mcp_video_server.py --http --host 0.0.0.0 --port 8000
```

### 2. Use HTTP Client
```bash
# Local connection
python standalone_client/mcp_http_client.py --server http://localhost:8000

# Remote connection
python standalone_client/mcp_http_client.py --server http://YOUR_SERVER_IP:8000

# With custom chat LLM
python standalone_client/mcp_http_client.py --server http://localhost:8000 --chat-llm mistral:latest
```

### 3. Use via video_client.py (Recommended)
```bash
# HTTP mode (default)
python standalone_client/video_client.py chat --chat-llm mistral:latest

# Specify server
python standalone_client/video_client.py chat --server http://remote-server:8000

# Legacy stdio mode
python standalone_client/video_client.py chat --stdio
```

## 🧠 LLM Configuration

The system uses **two different LLMs** for different purposes:

### 1. **Video Analysis LLM** (Server-side)
- **Purpose**: Analyzes video frames, extracts objects, describes scenes
- **Model**: `vision_model` in `config/default_config.json` (default: `llava:latest`)
- **Usage**: Automatically used for all video processing
- **Configuration**: 
  ```json
  {
    "llm": {
      "vision_model": "llava:latest",  // For video frame analysis
      "text_model": "llama2:latest"   // For summaries and Q&A
    }
  }
  ```

### 2. **Chat LLM** (Client-side)
- **Purpose**: Formats responses, handles natural language queries, chat interface
- **Model**: Specified when starting client with `--chat-llm` flag
- **Usage**: Processes user queries and formats server responses
- **Examples**:
  ```bash
  --chat-llm llama2:latest      # Default
  --chat-llm mistral:latest     # Better conversation
  --chat-llm codellama:latest   # Code-focused
  --chat-llm dolphin-mistral    # Instruction-tuned
  ```

### Model Requirements
- **Vision Model**: Must support image analysis (LLaVA, Bakllava, etc.)
- **Chat Model**: Any Ollama text model works
- **Server**: Requires vision model for video processing
- **Client**: Can use any chat model, runs independently

## 📋 Available Commands

### HTTP Client Commands:
- `status` - System status and video count
- `tools` - List available MCP tools  
- `help` - Show command help
- `exit` - Quit application
- Natural language queries like:
  - "show latest videos"
  - "what happened at the shed today?"
  - "find videos with cars"

## 🔧 Configuration Files

### Server Configuration (`config/default_config.json`):
```json
{
  "llm": {
    "vision_model": "llava:latest",     // Video frame analysis
    "text_model": "llama2:latest",      // Summaries and Q&A
    "ollama_host": "http://localhost:11434"
  },
  "performance": {
    "frame_batch_size": 10,             // Frames processed at once
    "max_concurrent_videos": 2          // Parallel video processing
  }
}
```

### Client Usage:
- No configuration needed
- Specify models via command line
- Connects to any MCP server URL

## 🌐 Remote Access Setup

### For Mobile/Remote Clients:

1. **Start server with external access**:
   ```bash
   python mcp_video_server.py --http --host 0.0.0.0 --port 8000
   ```

2. **Find your server IP**:
   ```bash
   # macOS/Linux
   ifconfig | grep "inet " | grep -v 127.0.0.1
   
   # Or use hostname
   hostname -I
   ```

3. **Connect from remote device**:
   ```bash
   python standalone_client/mcp_http_client.py --server http://192.168.1.100:8000
   ```

### Firewall Configuration:
```bash
# Allow port 8000 through firewall
sudo ufw allow 8000
```

## 🧪 Testing

### Manual Testing:
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with full integration
python test_http_setup.py
```

### Automated Tests:
```bash
# Run HTTP client tests
pytest tests/test_http_client.py -v

# Run all tests
pytest tests/ -v
```

## 🔍 Debugging

### Server Logs:
```bash
tail -f logs/mcp-video-server.log
```

### Common Issues:

1. **Port in use**: Kill existing processes
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

2. **Ollama not running**:
   ```bash
   ollama serve
   ollama pull llava:latest
   ollama pull llama2:latest
   ```

3. **Connection refused**: Check firewall and host binding

## 📱 Mobile Usage

The HTTP client can run on any device with Python:
- **Termux** (Android)
- **Pythonista** (iOS) 
- **Linux tablets**
- **Remote SSH sessions**

Just install requirements and use the HTTP client:
```bash
pip install httpx rich
python mcp_http_client.py --server http://YOUR_SERVER_IP:8000
```

## 🔄 Next Steps

1. **Complete session management** for tool calls
2. **Add WebSocket support** for real-time updates  
3. **Mobile app wrapper** for better UX
4. **Web interface** for browser access
5. **Authentication** for secure remote access

The foundation is solid and remote connectivity is working. The final MCP protocol integration just needs session handling completion.