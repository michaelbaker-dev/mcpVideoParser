# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-06-12

### Added
- **Video cleanup utility** (`clean_videos.py`) - Complete system reset functionality
  - Remove all videos from database and file system
  - Optional backup of database before cleaning
  - Dry-run mode for safe preview of deletions
  - Selective cleanup (keep originals vs clean everything)
  - Interactive confirmation with override options
  - Detailed statistics on cleanup results
- **Comprehensive unit tests** for video and audio processing
  - Video processing functionality tests with mocked dependencies
  - Audio transcription tests with Whisper integration
  - Integration tests for `process_new_video.py` script
  - Mock-based testing for various failure scenarios
- **Enhanced help documentation** in MCP HTTP client
  - Organized help into clear categories (transcript, visual, combined analysis)
  - Specific query examples for different use cases
  - Pro tips about system capabilities
  - Updated welcome message highlighting transcript and visual features

### Fixed
- **Video listing issue** - "list all videos" now shows all videos correctly
  - Fixed date parser to handle "recent" queries with 7-day window instead of 24 hours
  - Added `time_field` parameter to choose between `recording_timestamp` and `created_at`
  - Modified queries to use `created_at` for "recent/latest" to show recently processed videos
  - Root cause: New videos had old recording timestamps but recent creation dates
- **ProcessingResult attribute error** in `process_new_video.py`
  - Fixed AttributeError where script tried to access non-existent 'summary' attribute
  - Changed to display transcript preview instead of summary
  - Verified video and audio processing works correctly with sample videos

### Enhanced
- **Audio processing documentation** - Clarified that complete transcripts are preserved
  - System stores full transcripts (1000+ words per video) in database
  - Search functionality works across complete transcript content
  - Different query types access transcript vs visual analysis appropriately
- **Date parsing improvements** - Better handling of natural language time queries
  - "Recent/recently/latest" now properly returns last 7 days of content
  - Enhanced query routing based on user intent (recent processing vs recording time)
- **Utility scripts documentation** - Added comprehensive section to README
  - Video cleanup script usage and examples
  - Video processing script documentation
  - Clear instructions for different cleanup scenarios

### Technical Improvements
- **Database query optimization** - Smart time field selection for better results
- **Test coverage expansion** - Comprehensive test suite for core functionality  
- **Error handling improvements** - Better error messages and graceful failures
- **Code organization** - Cleaner separation of concerns in video processing pipeline

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

[0.1.1]: https://github.com/michaelbaker-dev/mcpVideoParser/releases/tag/v0.1.1
[0.1.0]: https://github.com/michaelbaker-dev/mcpVideoParser/releases/tag/v0.1.0