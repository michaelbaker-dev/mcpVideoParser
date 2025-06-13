# MCP Video Server Architecture

## Overview

The MCP Video Server uses a flexible architecture that separates concerns and allows for different LLMs to handle different tasks.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Claude Desktop  │     │  Chat Client    │     │   Direct CLI    │
│                 │     │ (Chat LLM)      │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         │ MCP Protocol          │ MCP Protocol           │ Direct API
         ▼                       ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Video Server                           │
├─────────────────────────────────────────────────────────────────┤
│ • Video Processor (frames + audio extraction)                   │
│ • Storage Manager (SQLite + hierarchical file storage)          │
│ • LLM Client for Video Analysis (Vision LLM: llava)            │
│ • MCP Tools (8 tools exposed via MCP protocol)                 │
└─────────────────────────────────────────────────────────────────┘
                    │                       │
                    ▼                       ▼
         ┌─────────────────┐     ┌─────────────────┐
         │     Ollama      │     │  Video Storage  │
         │ (Local LLMs)    │     │  (Filesystem)   │
         └─────────────────┘     └─────────────────┘
```

## LLM Separation

The system uses different LLMs for different purposes:

### 1. Video Analysis LLM (Server-side)
- **Model**: `llava:latest` (vision) + `llama2:latest` (text)
- **Purpose**: Analyze video frames, generate descriptions, answer questions about video content
- **Configured in**: `config/default_config.json` → `llm.vision_model` and `llm.text_model`

### 2. Chat Interface LLM (Client-side)
- **Model**: Configurable (default: `llama2:latest`)
- **Purpose**: 
  - Parse natural language queries
  - Determine user intent
  - Format responses in conversational style
- **Configured via**: Command line argument `--chat-llm`

### 3. Claude (When using Claude Desktop)
- **Model**: Claude (Anthropic's model)
- **Purpose**: Direct interaction with MCP tools
- **Note**: Responses go directly to Claude, no intermediate LLM needed

## Benefits of This Architecture

1. **Flexibility**: Different LLMs can be optimized for different tasks
   - Vision LLM (LLaVA) for understanding video content
   - Fast text LLM for chat interactions
   - Specialized models for specific domains

2. **Performance**: 
   - Chat responses can use a smaller, faster model
   - Video analysis can use a more powerful vision model
   - Both run locally via Ollama

3. **Consistency**: 
   - All clients (Claude, Chat, CLI) use the same MCP server
   - Same tools and capabilities across all interfaces
   - Single source of truth for video data

4. **Privacy**: 
   - All processing happens locally
   - No data sent to cloud services
   - Complete control over your video data

## Client Types

### 1. Claude Desktop
- Uses MCP protocol directly
- No intermediate LLM needed
- Claude handles natural language understanding

### 2. MCP Chat Client
- Uses separate Chat LLM for query understanding
- Communicates with MCP server via protocol
- Formats responses conversationally

### 3. Direct CLI
- Bypasses MCP for simple operations
- Direct access to storage and processor
- Useful for automation and scripts

## Data Flow Examples

### Example 1: Chat Client Query
```
User: "What happened at the shed yesterday?"
  ↓
Chat LLM: Parse intent → {intent: "query_videos", location: "shed", time: "yesterday"}
  ↓
MCP Client: Call tool "query_location_time" with parameters
  ↓
MCP Server: Query database, return results
  ↓
Chat LLM: Format response → "Found 3 videos from the shed yesterday..."
  ↓
User: Sees formatted response with table
```

### Example 2: Claude Desktop Query
```
User (in Claude): "Show me videos from the driveway"
  ↓
Claude: Understands intent, calls MCP tool directly
  ↓
MCP Server: Returns results in JSON
  ↓
Claude: Formats and displays to user
```

### Example 3: Video Processing
```
Video File → MCP Server
  ↓
Frame Extraction → Multiple frames
  ↓
Vision LLM (LLaVA): Analyze each frame → Descriptions
  ↓
Audio Extraction → Whisper → Transcript
  ↓
Database: Store metadata, descriptions, transcript
  ↓
Response: "Video processed successfully"
```

## Configuration

### Server Configuration (`config/default_config.json`)
```json
{
  "llm": {
    "vision_model": "llava:latest",  // For video frame analysis
    "text_model": "llama2:latest",   // For text generation
    "temperature": 0.7
  }
}
```

### Chat Client Configuration
```bash
# Use default chat model (llama2)
./video_client.py chat

# Use a different model for chat
./video_client.py chat --chat-llm mistral:latest
```

## Available Ollama Models

For video analysis (vision):
- `llava:latest` - Recommended for frame analysis
- `bakllava:latest` - Alternative vision model

For chat interface:
- `llama2:latest` - Good balance of speed and quality
- `mistral:latest` - Faster, good for chat
- `neural-chat:latest` - Optimized for conversations
- `phi:latest` - Very fast, smaller model

For specialized tasks:
- `codellama:latest` - If analyzing code in videos
- `medllama2:latest` - For medical content
- `nous-hermes:latest` - Good general performance