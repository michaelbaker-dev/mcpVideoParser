# MCP Video Parser

A powerful video analysis system that uses the Model Context Protocol (MCP) to process, analyze, and query video content using AI vision models.

## 🎬 Features

- **AI-Powered Video Analysis**: Automatically extracts and analyzes frames using vision LLMs (Llava)
- **Natural Language Queries**: Search videos using conversational queries
- **Time-Based Search**: Query videos by relative time ("last week") or specific dates
- **Location-Based Organization**: Organize videos by location (shed, garage, etc.)
- **Audio Transcription**: Extract and search through video transcripts
- **Chat Integration**: Natural conversations with Mistral/Llama while maintaining video context
- **Scene Detection**: Intelligent frame extraction based on visual changes
- **MCP Protocol**: Standards-based integration with Claude and other MCP clients

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- ffmpeg (for video processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/michaelbaker-dev/mcpVideoParser.git
cd mcpVideoParser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pull required Ollama models:
```bash
ollama pull llava:latest    # For vision analysis
ollama pull mistral:latest  # For chat interactions
```

4. Start the MCP server:
```bash
python mcp_video_server.py --http --host localhost --port 8000
```

### Basic Usage

1. **Process a video**:
```bash
python process_new_video.py /path/to/video.mp4 --location garage
```

2. **Start the chat client**:
```bash
python standalone_client/mcp_http_client.py --chat-llm mistral:latest
```

3. **Example queries**:
- "Show me the latest videos"
- "What happened at the garage yesterday?"
- "Find videos with cars"
- "Give me a summary of all videos from last week"

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Video Files   │────▶│ Video Processor │────▶│ Frame Analysis  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                         │
                                ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MCP Server    │◀────│ Storage Manager │◀────│   Ollama LLM    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   HTTP Client   │
└─────────────────┘
```

## 🛠️ Configuration

Edit `config/default_config.json` to customize:

- **Frame extraction rate**: How many frames to analyze
- **Scene detection sensitivity**: When to capture scene changes
- **Storage settings**: Where to store videos and data
- **LLM models**: Which models to use for vision and chat

See [Configuration Guide](docs/CONFIGURATION.md) for details.

## 🔧 MCP Tools

The server exposes these MCP tools:

- `process_video` - Process and analyze a video file
- `query_location_time` - Query videos by location and time
- `search_videos` - Search video content and transcripts
- `get_video_summary` - Get AI-generated summary of a video
- `ask_video` - Ask questions about specific videos
- `analyze_moment` - Analyze specific timestamp in a video
- `get_video_stats` - Get system statistics
- `get_video_guide` - Get usage instructions

## 📖 Documentation

- [API Reference](docs/API.md) - Detailed MCP tool documentation
- [Configuration Guide](docs/CONFIGURATION.md) - Customization options
- [Video Analysis Info](VIDEO_ANALYSIS_INFO.md) - How video processing works
- [Development Guide](docs/DEVELOPMENT.md) - Contributing and testing
- [Deployment Guide](docs/DEPLOYMENT.md) - Production setup

## 🚦 Development

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests (requires Ollama)
python -m pytest tests/integration/ -v
```

### Project Structure
```
mcp-video-server/
├── src/
│   ├── llm/            # LLM client implementations
│   ├── processors/     # Video processing logic
│   ├── storage/        # Database and file management
│   ├── tools/          # MCP tool definitions
│   └── utils/          # Utilities and helpers
├── standalone_client/  # HTTP client implementation
├── config/            # Configuration files
├── tests/             # Test suite
└── video_data/        # Video storage (git-ignored)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Roadmap

- ✅ Basic video processing and analysis
- ✅ MCP server implementation
- ✅ Natural language queries
- ✅ Chat integration with context
- 🚧 Enhanced time parsing (see [INTELLIGENT_QUERY_PLAN.md](INTELLIGENT_QUERY_PLAN.md))
- 🚧 Multi-camera support
- 🚧 Real-time processing
- 🚧 Web interface

## 🐛 Troubleshooting

### Common Issues

1. **Ollama not running**:
```bash
ollama serve  # Start Ollama
```

2. **Missing models**:
```bash
ollama pull llava:latest
ollama pull mistral:latest
```

3. **Port already in use**:
```bash
# Change port in command
python mcp_video_server.py --http --port 8001
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on [FastMCP](https://github.com/jlowin/fastmcp) framework
- Uses [Ollama](https://ollama.ai) for local LLM inference
- Inspired by the Model Context Protocol specification

## 💬 Support

- [Report Issues](https://github.com/michaelbaker-dev/mcpVideoParser/issues)
- [Discussions](https://github.com/michaelbaker-dev/mcpVideoParser/discussions)

---

**Version**: 0.1.0  
**Author**: Michael Baker  
**Status**: Beta - Breaking changes possible