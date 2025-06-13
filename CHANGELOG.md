# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-06-13

### Added
- Initial release of MCP Video Parser
- Video processing with frame extraction and analysis
- MCP server implementation with 8 tools
- Natural language query support
- Location-based video organization
- Time-based video queries
- Audio transcription with Whisper
- Scene detection for intelligent frame extraction
- Chat integration with video context awareness
- HTTP-based MCP transport for remote access
- Comprehensive test suite
- Documentation and examples

### Features
- `process_video` - Process and analyze video files
- `query_location_time` - Query videos by location and time
- `search_videos` - Search video content and transcripts
- `get_video_summary` - Generate AI summaries
- `ask_video` - Ask questions about specific videos
- `analyze_moment` - Analyze specific timestamps
- `get_video_stats` - System statistics
- `get_video_guide` - Usage instructions

### Technical
- FastMCP framework integration
- Ollama LLM integration (Llava for vision, Mistral/Llama for chat)
- SQLite storage with location-based file organization
- Configurable frame extraction rates
- Session management for HTTP connections
- Proper MCP protocol handshake implementation

### Known Issues
- Limited natural language time parsing (see INTELLIGENT_QUERY_PLAN.md)
- Basic pattern matching for query routing
- No real-time video processing support yet

[0.1.0]: https://github.com/michaelbaker-dev/mcpVideoParser/releases/tag/v0.1.0