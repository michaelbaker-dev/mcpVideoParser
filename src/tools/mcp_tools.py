"""MCP tool definitions for video analysis."""
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from mcp.server.fastmcp import Context
from ..storage.manager import StorageManager
from ..storage.schemas import (
    ProcessingStatus, QueryRequest, QueryResponse,
    SearchResult, VideoSummary, MomentAnalysis
)
from ..processors.video import VideoProcessor
from ..llm.ollama_client import OllamaClient
from ..utils.logging import get_logger


class VideoTools:
    """MCP tools for video analysis."""
    
    def __init__(self, processor: VideoProcessor, storage: StorageManager):
        self.processor = processor
        self.storage = storage
        self.logger = get_logger(__name__)
        self._llm_client = None
    
    async def _get_llm_client(self) -> OllamaClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = OllamaClient()
        return self._llm_client
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as relative time."""
        from datetime import datetime as dt
        now = dt.now()
        diff = now - timestamp
        
        if diff.days == 0:
            if diff.seconds < 3600:
                return f"{diff.seconds // 60} minutes ago"
            else:
                return f"{diff.seconds // 3600} hours ago"
        elif diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 30:
            return f"{diff.days // 7} weeks ago"
        else:
            return timestamp.strftime("%B %d, %Y")
    
    def register(self, mcp):
        """Register all tools with the MCP server."""
        
        @mcp.tool()
        async def process_video(
            video_path: str,
            location: Optional[str] = None,
            recording_timestamp: Optional[str] = None,
            sample_rate: Optional[int] = None,
            extract_audio: bool = True
        ) -> Dict[str, Any]:
            """
            Process a video file for analysis.
            
            Args:
                video_path: Path to the video file
                location: Location name (e.g., "shed", "driveway"). Auto-detected if not provided.
                recording_timestamp: When the video was recorded (ISO format or natural language).
                                   Uses file timestamp if not provided.
                sample_rate: Extract 1 frame every N frames (default: 30)
                extract_audio: Whether to transcribe audio (default: True)
                
            Returns:
                Processing result with video_id and statistics
            """
            try:
                # Parse timestamp if provided
                parsed_timestamp = None
                if recording_timestamp:
                    from datetime import datetime
                    from ..utils.date_parser import DateParser
                    
                    # Try ISO format first
                    try:
                        parsed_timestamp = datetime.fromisoformat(recording_timestamp)
                    except:
                        # Try natural language parsing
                        start_time, _ = DateParser.parse_date_query(recording_timestamp)
                        parsed_timestamp = start_time
                
                # Store video
                self.logger.info(f"Processing video: {video_path}")
                metadata = await self.storage.store_video(
                    video_path, 
                    location=location,
                    recording_timestamp=parsed_timestamp
                )
                
                # Update processor config if sample_rate provided
                if sample_rate:
                    self.processor.config.processing.frame_sample_rate = sample_rate
                
                # Process video
                result = await self.processor.process_video(metadata.video_id)
                
                return {
                    "video_id": result.video_id,
                    "status": result.status.value,
                    "frames_extracted": result.frames_extracted,
                    "frames_analyzed": result.frames_analyzed,
                    "has_transcript": result.transcript is not None,
                    "processing_time": result.processing_time,
                    "message": f"Successfully processed video. ID: {result.video_id}"
                }
                
            except Exception as e:
                self.logger.error(f"Error processing video: {e}")
                return {
                    "error": str(e),
                    "status": "failed"
                }
        
        @mcp.tool()
        async def ask_video(
            video_id: str,
            question: str,
            context_window: int = 60
        ) -> Dict[str, Any]:
            """
            Ask a question about a processed video.
            
            Args:
                video_id: ID of the processed video
                question: Natural language question about the video
                context_window: Seconds of context around relevant moments (default: 60)
                
            Returns:
                Answer based on video content analysis
            """
            try:
                # Get video data
                processing_result = self.storage.get_processing_result(video_id)
                if not processing_result:
                    return {"error": f"Video {video_id} not found or not processed"}
                
                # Build context from frame descriptions and transcript
                context_parts = []
                
                # Add frame descriptions
                if processing_result.timeline:
                    context_parts.append("Visual timeline:")
                    for frame in processing_result.timeline[:50]:  # Limit frames
                        context_parts.append(
                            f"[{frame.timestamp:.1f}s] {frame.description}"
                        )
                
                # Add transcript
                if processing_result.transcript:
                    context_parts.append("\nTranscript:")
                    context_parts.append(processing_result.transcript[:2000])
                
                context = "\n".join(context_parts)
                
                # Get LLM to answer
                llm = await self._get_llm_client()
                answer = await llm.answer_video_question(question, context)
                
                if answer:
                    return {
                        "video_id": video_id,
                        "question": question,
                        "answer": answer,
                        "confidence": 0.8,
                        "sources": {
                            "frames_used": len(processing_result.timeline),
                            "has_transcript": bool(processing_result.transcript)
                        }
                    }
                else:
                    return {
                        "error": "Failed to generate answer",
                        "video_id": video_id
                    }
                    
            except Exception as e:
                self.logger.error(f"Error answering question: {e}")
                return {"error": str(e)}
        
        @mcp.tool()
        async def query_location_time(
            location: Optional[str] = None,
            time_query: Optional[str] = None,
            content_query: Optional[str] = None,
            limit: int = 20
        ) -> Dict[str, Any]:
            """
            Query videos by location and time using natural language.
            
            Args:
                location: Location name (e.g., "shed", "driveway", "front door")
                time_query: Natural language time query (e.g., "today", "yesterday", 
                           "last week", "December 6th", "last 3 days")
                content_query: Optional content to search for (e.g., "car", "person")
                limit: Maximum number of results
                
            Returns:
                List of videos matching the criteria with summaries
                
            Examples:
                - location="shed", time_query="today" 
                - location="driveway", time_query="yesterday", content_query="car"
                - time_query="last week" (all locations)
            """
            try:
                from ..utils.date_parser import DateParser
                
                # Parse time query
                start_time = None
                end_time = None
                if time_query:
                    start_time, end_time = DateParser.parse_date_query(time_query)
                    self.logger.info(f"Parsed time query '{time_query}' to {start_time} - {end_time}")
                
                # Query videos
                videos = self.storage.query_videos_by_location_and_time(
                    location=location,
                    start_time=start_time,
                    end_time=end_time,
                    content_query=content_query,
                    limit=limit
                )
                
                if not videos:
                    return {
                        "query": {
                            "location": location,
                            "time": time_query,
                            "content": content_query
                        },
                        "message": "No videos found matching the criteria",
                        "results": []
                    }
                
                # Build results with summaries
                results = []
                for video in videos:
                    # Get processing result for summary
                    processing_result = self.storage.get_processing_result(video.video_id)
                    
                    # Create summary of what happened
                    summary_parts = []
                    if processing_result and processing_result.timeline:
                        # Get first few frame descriptions
                        for frame in processing_result.timeline[:3]:
                            if frame.description:
                                summary_parts.append(f"[{frame.timestamp:.1f}s] {frame.description}")
                    
                    results.append({
                        "video_id": video.video_id,
                        "location": video.location,
                        "timestamp": video.recording_timestamp.isoformat(),
                        "time_ago": self._format_time_ago(video.recording_timestamp),
                        "duration": f"{video.duration:.1f}s" if video.duration else "Unknown",
                        "summary": "\n".join(summary_parts) if summary_parts else "No analysis available",
                        "has_transcript": bool(processing_result and processing_result.transcript)
                    })
                
                return {
                    "query": {
                        "location": location,
                        "time": time_query,
                        "content": content_query,
                        "time_range": DateParser.format_datetime_range(start_time, end_time) if start_time else None
                    },
                    "total_results": len(results),
                    "results": results
                }
                
            except Exception as e:
                self.logger.error(f"Error querying videos: {e}")
                return {"error": str(e)}
        
        @mcp.tool()
        async def search_videos(
            query: str,
            limit: int = 10
        ) -> Dict[str, Any]:
            """
            Search across all processed videos.
            
            Args:
                query: Search query (searches transcripts and descriptions)
                limit: Maximum number of results (default: 10)
                
            Returns:
                List of matching videos with relevance scores
            """
            try:
                results = self.storage.search_videos(query, limit)
                
                search_results = []
                for video_id, filename, match_count in results:
                    search_results.append({
                        "video_id": video_id,
                        "filename": filename,
                        "relevance_score": match_count,
                        "matches": match_count
                    })
                
                return {
                    "query": query,
                    "total_results": len(search_results),
                    "results": search_results
                }
                
            except Exception as e:
                self.logger.error(f"Error searching videos: {e}")
                return {"error": str(e)}
        
        @mcp.tool()
        async def list_videos(
            status: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            List all videos in the system.
            
            Args:
                status: Filter by status (pending, processing, completed, failed)
                
            Returns:
                List of videos with their metadata
            """
            try:
                with self.storage._get_db() as conn:
                    cursor = conn.cursor()
                    
                    if status:
                        cursor.execute(
                            "SELECT * FROM videos WHERE status = ? ORDER BY created_at DESC",
                            (status,)
                        )
                    else:
                        cursor.execute("SELECT * FROM videos ORDER BY created_at DESC")
                    
                    videos = []
                    for row in cursor.fetchall():
                        videos.append({
                            "video_id": row['video_id'],
                            "filename": row['filename'],
                            "duration": row['duration'],
                            "status": row['status'],
                            "created_at": row['created_at'],
                            "processed_at": row['processed_at']
                        })
                
                return {
                    "total_videos": len(videos),
                    "videos": videos
                }
                
            except Exception as e:
                self.logger.error(f"Error listing videos: {e}")
                return {"error": str(e)}
        
        @mcp.tool()
        async def get_video_summary(
            video_id: str,
            detail_level: str = "medium"
        ) -> Dict[str, Any]:
            """
            Get an AI-generated summary of a video.
            
            Args:
                video_id: ID of the processed video
                detail_level: Level of detail (brief, medium, detailed)
                
            Returns:
                AI-generated video summary
            """
            try:
                # Get video data
                processing_result = self.storage.get_processing_result(video_id)
                if not processing_result:
                    return {"error": f"Video {video_id} not found or not processed"}
                
                metadata = self.storage.get_video_metadata(video_id)
                
                # Get frame descriptions
                frame_descriptions = [
                    f.description for f in processing_result.timeline
                    if f.description
                ]
                
                # Generate summary
                llm = await self._get_llm_client()
                summary_text = await llm.generate_video_summary(
                    frame_descriptions,
                    processing_result.transcript
                )
                
                if summary_text:
                    # Extract key moments (simplified)
                    key_moments = []
                    for i, frame in enumerate(processing_result.timeline[::10]):  # Every 10th frame
                        if frame.description and len(frame.description) > 20:
                            key_moments.append({
                                "timestamp": frame.timestamp,
                                "description": frame.description[:100]
                            })
                            if len(key_moments) >= 5:
                                break
                    
                    return {
                        "video_id": video_id,
                        "filename": metadata.filename,
                        "duration": metadata.duration,
                        "summary": summary_text,
                        "key_moments": key_moments,
                        "frame_count": len(processing_result.timeline),
                        "has_transcript": bool(processing_result.transcript)
                    }
                else:
                    return {
                        "error": "Failed to generate summary",
                        "video_id": video_id
                    }
                    
            except Exception as e:
                self.logger.error(f"Error generating summary: {e}")
                return {"error": str(e)}
        
        @mcp.tool()
        async def analyze_moment(
            video_id: str,
            timestamp: float,
            duration: float = 10.0
        ) -> Dict[str, Any]:
            """
            Analyze a specific moment in a video.
            
            Args:
                video_id: ID of the processed video
                timestamp: Start time in seconds
                duration: How many seconds to analyze (default: 10)
                
            Returns:
                Detailed analysis of the specified moment
            """
            try:
                # Get video data
                processing_result = self.storage.get_processing_result(video_id)
                if not processing_result:
                    return {"error": f"Video {video_id} not found or not processed"}
                
                # Find relevant frames
                end_time = timestamp + duration
                relevant_frames = [
                    f for f in processing_result.timeline
                    if timestamp <= f.timestamp <= end_time
                ]
                
                if not relevant_frames:
                    return {
                        "error": f"No frames found in the range {timestamp:.1f}s - {end_time:.1f}s"
                    }
                
                # Collect descriptions and objects
                descriptions = [f.description for f in relevant_frames if f.description]
                all_objects = set()
                for frame in relevant_frames:
                    all_objects.update(frame.objects_detected)
                
                # Extract transcript segment if available
                transcript_segment = ""
                if processing_result.transcript:
                    # This is simplified - in production, use proper timestamp alignment
                    words = processing_result.transcript.split()
                    words_per_second = len(words) / (processing_result.timeline[-1].timestamp or 1)
                    start_word = int(timestamp * words_per_second)
                    end_word = int(end_time * words_per_second)
                    transcript_segment = " ".join(words[start_word:end_word])
                
                return {
                    "video_id": video_id,
                    "timestamp_start": timestamp,
                    "timestamp_end": end_time,
                    "duration": duration,
                    "frames_analyzed": len(relevant_frames),
                    "descriptions": descriptions,
                    "objects_detected": list(all_objects),
                    "transcript_segment": transcript_segment,
                    "summary": f"Analyzed {len(relevant_frames)} frames from {timestamp:.1f}s to {end_time:.1f}s"
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing moment: {e}")
                return {"error": str(e)}
        
        @mcp.tool()
        async def get_video_stats() -> Dict[str, Any]:
            """
            Get storage and processing statistics.
            
            Returns:
                System statistics including storage usage and video counts
            """
            try:
                stats = self.storage.get_storage_stats()
                
                # Check Ollama status
                llm = await self._get_llm_client()
                ollama_available = await llm.is_available()
                
                return {
                    "storage": {
                        "total_videos": stats["total_videos"],
                        "processed_videos": stats["processed_videos"],
                        "total_size_gb": round(stats["total_size_gb"], 2),
                        "processed_size_gb": round(stats["processed_size_gb"], 2)
                    },
                    "system": {
                        "ollama_available": ollama_available,
                        "base_path": str(self.storage.base_path)
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error getting stats: {e}")
                return {"error": str(e)}
        
        @mcp.resource("video://guide")
        async def get_video_guide() -> str:
            """Provide a guide for using the video analysis tools."""
            return """# Video Analysis MCP Server Guide

## Overview
This MCP server provides video analysis capabilities using local LLMs through Ollama.

## Available Tools

### 1. process_video
Process a video file for analysis.
- **video_path**: Path to the video file
- **location**: Location name (e.g., "shed", "driveway")
- **recording_timestamp**: When recorded (ISO format or natural language)
- **sample_rate**: Frame sampling rate (default: 30)
- **extract_audio**: Whether to transcribe audio (default: true)

Example: Process a video at /path/to/video.mp4 from the shed yesterday

### 2. ask_video
Ask questions about a processed video.
- **video_id**: The video ID returned from process_video
- **question**: Your question about the video
- **context_window**: Seconds of context to consider (default: 60)

Example: "What activities are shown in video vid_abc123?"

### 3. query_location_time
Query videos by location and time using natural language.
- **location**: Location name (e.g., "shed", "driveway", "front door")
- **time_query**: Natural language time (e.g., "today", "yesterday", "last week")
- **content_query**: Optional content search (e.g., "car", "person")
- **limit**: Maximum results (default: 20)

Examples:
- "What happened at the shed today?"
- "Were there any cars in the driveway yesterday?"
- "Show me all videos from last week"

### 4. search_videos
Search across all processed videos.
- **query**: Search term
- **limit**: Maximum results (default: 10)

Example: Search for "meeting" across all videos

### 5. list_videos
List all videos in the system.
- **status**: Filter by status (optional)

### 5. get_video_summary
Get an AI-generated summary of a video.
- **video_id**: The video ID
- **detail_level**: "brief", "medium", or "detailed"

### 6. analyze_moment
Analyze a specific moment in a video.
- **video_id**: The video ID
- **timestamp**: Start time in seconds
- **duration**: Duration to analyze (default: 10)

### 7. get_video_stats
Get system statistics and storage usage.

## Workflow Example

1. Process a video:
   "Process the video at /home/user/videos/meeting.mp4"

2. Ask questions:
   "What topics were discussed in video vid_abc123?"
   "Were there any action items mentioned?"

3. Search for specific content:
   "Search for videos mentioning 'product launch'"

4. Get summaries:
   "Get a detailed summary of video vid_abc123"

## Requirements

- Ollama must be running locally
- Required models: llava (for vision) and llama2 (for text)
- Sufficient disk space for video processing

## Tips

- Process videos once, then query multiple times
- Use search to find videos by content
- Analyze specific moments for detailed information
"""
    
    async def cleanup(self):
        """Clean up resources."""
        if self._llm_client:
            await self._llm_client.close()