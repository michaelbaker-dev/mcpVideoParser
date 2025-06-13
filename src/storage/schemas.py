"""Data models and schemas for video processing and storage."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path


class ProcessingStatus(str, Enum):
    """Video processing status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class FrameAnalysis(BaseModel):
    """Analysis result for a single frame."""
    frame_number: int
    timestamp: float
    frame_path: str
    description: str
    objects_detected: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class VideoMetadata(BaseModel):
    """Metadata for a video file."""
    video_id: str
    original_path: str
    filename: str
    location: str  # e.g., "shed", "driveway", "front_door"
    recording_timestamp: datetime  # When the video was recorded
    duration: float
    fps: float
    width: int
    height: int
    codec: str
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    
    @validator('video_id')
    def validate_video_id(cls, v):
        if not v or len(v) < 6:
            raise ValueError('video_id must be at least 6 characters')
        return v


class ProcessingResult(BaseModel):
    """Result of video processing operation."""
    video_id: str
    status: ProcessingStatus
    frames_extracted: int
    frames_analyzed: int
    transcript: Optional[str] = None
    timeline: List[FrameAnalysis] = Field(default_factory=list)
    processing_time: float
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class VideoSummary(BaseModel):
    """AI-generated summary of video content."""
    video_id: str
    brief: str
    key_moments: List[Dict[str, Any]]
    topics: List[str]
    people_count: Optional[int] = None
    duration_summary: str
    generated_at: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    """Search result for video queries."""
    video_id: str
    filename: str
    relevance_score: float
    matched_segments: List[Dict[str, Any]]
    context: str
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None


class MomentAnalysis(BaseModel):
    """Detailed analysis of a specific moment in video."""
    video_id: str
    timestamp: float
    duration: float
    frames_analyzed: int
    description: str
    objects: List[str]
    activities: List[str]
    audio_transcript: Optional[str] = None
    confidence: float


class QueryRequest(BaseModel):
    """Request model for video queries."""
    video_id: Optional[str] = None
    question: str
    context_window: int = Field(default=60, ge=10, le=300)
    include_transcript: bool = True
    include_visual: bool = True
    max_results: int = Field(default=10, ge=1, le=50)


class QueryResponse(BaseModel):
    """Response model for video queries."""
    query: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time: float
    model_used: str