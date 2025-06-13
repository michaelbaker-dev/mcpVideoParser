# API Reference

## MCP Tools

The MCP Video Parser exposes the following tools through the Model Context Protocol:

### process_video

Process a video file for analysis.

**Parameters:**
- `video_path` (string, required): Path to the video file
- `location` (string, optional): Location name (e.g., "shed", "garage"). Auto-detected from path if not provided
- `recording_timestamp` (string, optional): When the video was recorded (ISO format or natural language). Uses file timestamp if not provided
- `sample_rate` (integer, optional): Extract 1 frame every N frames. Default: 30
- `extract_audio` (boolean, optional): Whether to transcribe audio. Default: true

**Returns:**
```json
{
  "video_id": "vid_abc123",
  "status": "completed",
  "frames_extracted": 45,
  "frames_analyzed": 45,
  "has_transcript": true,
  "processing_time": 23.5,
  "message": "Successfully processed video. ID: vid_abc123"
}
```

### query_location_time

Query videos by location and time using natural language.

**Parameters:**
- `location` (string, optional): Location name (e.g., "shed", "driveway", "front door")
- `time_query` (string, optional): Natural language time query (e.g., "today", "yesterday", "last week", "December 6th", "last 3 days")
- `content_query` (string, optional): Content to search for (e.g., "car", "person")
- `limit` (integer, optional): Maximum number of results. Default: 20

**Returns:**
```json
{
  "query": {
    "location": "shed",
    "time": "yesterday",
    "content": null,
    "time_range": "2024-01-14 00:00:00 to 2024-01-14 23:59:59"
  },
  "total_results": 3,
  "results": [
    {
      "video_id": "vid_abc123",
      "location": "shed",
      "timestamp": "2024-01-14T15:30:00",
      "time_ago": "1 day ago",
      "duration": "120.5s",
      "summary": "[0.0s] Person entering shed\n[30.0s] Working on equipment",
      "has_transcript": true
    }
  ]
}
```

### search_videos

Search across all processed videos.

**Parameters:**
- `query` (string, required): Search query (searches transcripts and descriptions)
- `limit` (integer, optional): Maximum number of results. Default: 10

**Returns:**
```json
{
  "query": "delivery",
  "total_results": 2,
  "results": [
    {
      "video_id": "vid_def456",
      "location": "front_door",
      "timestamp": "2024-01-14T10:15:00",
      "relevance_score": 0.85,
      "matched_in": ["transcript", "description"],
      "preview": "...package delivery at front door...",
      "summary": "Delivery person drops off package"
    }
  ]
}
```

### get_video_summary

Get AI-generated summary of a video.

**Parameters:**
- `video_id` (string, required): Video identifier
- `detail_level` (string, optional): One of "brief", "medium", or "detailed". Default: "medium"

**Returns:**
```json
{
  "video_id": "vid_abc123",
  "location": "garage",
  "duration": "180.0s",
  "summary": "The video shows a person working on a car in the garage. Key events include:\n1. Opening garage door (0:00-0:15)\n2. Inspecting engine (0:30-1:45)\n3. Changing oil (2:00-2:45)",
  "key_moments": [
    {"time": "0:30", "event": "Started engine inspection"},
    {"time": "2:00", "event": "Began oil change"}
  ],
  "objects_detected": ["person", "car", "tools"],
  "has_audio": true,
  "transcript_summary": "Discussion about car maintenance"
}
```

### ask_video

Ask a question about a processed video.

**Parameters:**
- `video_id` (string, required): Video identifier
- `question` (string, required): Natural language question about the video
- `context_window` (integer, optional): Seconds of context around relevant moments. Default: 60

**Returns:**
```json
{
  "video_id": "vid_abc123",
  "question": "What tools were used?",
  "answer": "Based on the video analysis, the following tools were used: a wrench set for removing bolts, an oil filter wrench, and a funnel for adding new oil. These tools were primarily used between the 2:00 and 2:45 marks.",
  "relevant_moments": [
    {"timestamp": "2:05", "description": "Using wrench on oil pan"},
    {"timestamp": "2:20", "description": "Removing oil filter"}
  ],
  "confidence": 0.92
}
```

### analyze_moment

Analyze a specific timestamp in a video.

**Parameters:**
- `video_id` (string, required): Video identifier
- `timestamp` (number, required): Timestamp in seconds
- `duration` (number, optional): Duration to analyze in seconds. Default: 10

**Returns:**
```json
{
  "video_id": "vid_abc123",
  "timestamp": 120.0,
  "duration": 10.0,
  "analysis": {
    "description": "Person standing in front of car with hood open, appears to be explaining something while pointing at the engine",
    "objects": ["person", "car", "engine", "tools"],
    "activities": ["explaining", "pointing", "demonstrating"],
    "audio_transcript": "...and you can see here where the oil filter is located..."
  }
}
```

### get_video_stats

Get system statistics.

**Parameters:** None

**Returns:**
```json
{
  "storage": {
    "total_videos": 45,
    "processed_videos": 45,
    "total_size_gb": 12.3,
    "locations": ["garage", "shed", "driveway", "front_door"]
  },
  "system": {
    "ollama_available": true,
    "vision_model": "llava:latest",
    "text_model": "llama2:latest",
    "processing_queue": 0
  },
  "recent_activity": {
    "last_processed": "2024-01-15T10:30:00",
    "videos_today": 3,
    "videos_this_week": 15
  }
}
```

### get_video_guide

Get usage instructions and examples.

**Parameters:** None

**Returns:** Markdown-formatted guide with examples of how to use the video analysis system.

## Error Responses

All tools return consistent error responses:

```json
{
  "error": "Error message",
  "status": "failed",
  "details": "Additional error context"
}
```

## Rate Limits

- Video processing: Limited by `max_concurrent_videos` config (default: 2)
- Queries: No hard limit, but complex queries may take longer
- LLM calls: Limited by Ollama's capacity

## Best Practices

1. **Video Processing**:
   - Process videos during off-peak hours
   - Use appropriate `sample_rate` for your needs
   - Consider disabling audio transcription for silent videos

2. **Querying**:
   - Be specific with location and time queries
   - Use content queries to narrow results
   - Combine tools for complex questions

3. **Performance**:
   - Limit result counts for faster responses
   - Cache frequently accessed summaries
   - Monitor storage usage regularly