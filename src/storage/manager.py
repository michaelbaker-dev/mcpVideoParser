"""Storage manager for video files and metadata."""
import json
import shutil
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
import aiofiles
import asyncio
from ..utils.logging import get_logger
from ..utils.config import get_config
from .schemas import (
    VideoMetadata, ProcessingResult, VideoSummary,
    FrameAnalysis, ProcessingStatus
)


class StorageManager:
    """Manages storage of video files and associated data."""
    
    def __init__(self, base_path: Optional[str] = None):
        self.config = get_config()
        self.base_path = Path(base_path or self.config.storage.base_path)
        self.logger = get_logger(__name__)
        
        # Create directory structure
        self._init_directories()
        
        # Initialize database
        self._init_database()
        
    def _init_directories(self):
        """Initialize directory structure."""
        directories = [
            self.base_path / "locations",
            self.base_path / "processed",
            self.base_path / "index",
            self.base_path / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        db_path = self.base_path / "index" / "metadata.db"
        
        with self._get_db() as conn:
            cursor = conn.cursor()
            
            # Videos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    original_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    location TEXT NOT NULL,
                    recording_timestamp TIMESTAMP NOT NULL,
                    duration REAL NOT NULL,
                    fps REAL NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    codec TEXT,
                    size_bytes INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    processing_time REAL DEFAULT 0.0
                )
            """)
            
            # Frame analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frame_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    frame_path TEXT NOT NULL,
                    description TEXT,
                    objects_detected TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id),
                    UNIQUE(video_id, frame_number)
                )
            """)
            
            # Transcripts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    video_id TEXT PRIMARY KEY,
                    transcript TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id)
                )
            """)
            
            # Search index table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    timestamp_start REAL,
                    timestamp_end REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_location ON videos(location)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_timestamp ON videos(recording_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_location_timestamp ON videos(location, recording_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_video ON frame_analysis(video_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_video ON search_index(video_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_content ON search_index(content)")
            
            conn.commit()
            self.logger.info("Database initialized successfully")
    
    @contextmanager
    def _get_db(self):
        """Get database connection context manager."""
        db_path = self.base_path / "index" / "metadata.db"
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def generate_video_id(self, file_path: str) -> str:
        """Generate unique video ID from file path."""
        # Use hash of file path + timestamp for uniqueness
        content = f"{file_path}_{datetime.now().isoformat()}"
        hash_obj = hashlib.sha256(content.encode())
        return f"vid_{hash_obj.hexdigest()[:12]}"
    
    async def store_video(self, video_path: str, location: Optional[str] = None, 
                         recording_timestamp: Optional[datetime] = None) -> VideoMetadata:
        """Store original video file and create metadata entry.
        
        Args:
            video_path: Path to video file
            location: Location name (e.g., "shed", "driveway"). If not provided, 
                     will try to extract from path or filename
            recording_timestamp: When the video was recorded. If not provided,
                               will use file modification time
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract or validate location
        if location is None:
            location = self._extract_location_from_path(video_path)
        location = self._normalize_location(location)
        
        # Extract or use provided timestamp
        if recording_timestamp is None:
            recording_timestamp = self._extract_timestamp_from_file(video_path)
        
        # Generate video ID with timestamp for uniqueness
        video_id = self.generate_video_id(f"{location}_{recording_timestamp.isoformat()}_{video_path.name}")
        
        # Create destination path with location/date structure
        from ..utils.date_parser import DateParser
        year, month, day = DateParser.get_date_path_components(recording_timestamp)
        
        dest_dir = self.base_path / "locations" / location / year / month / day
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp_str = recording_timestamp.strftime("%H%M%S")
        dest_filename = f"{video_id}_{timestamp_str}{video_path.suffix}"
        dest_path = dest_dir / dest_filename
        
        try:
            # Copy file asynchronously
            await self._copy_file_async(video_path, dest_path)
            
            # Get video metadata (this would normally use ffprobe)
            metadata = VideoMetadata(
                video_id=video_id,
                original_path=str(video_path),
                filename=video_path.name,
                location=location,
                recording_timestamp=recording_timestamp,
                duration=0.0,  # Will be updated by processor
                fps=0.0,
                width=0,
                height=0,
                codec="unknown",
                size_bytes=video_path.stat().st_size,
                created_at=datetime.now()
            )
            
            # Store in database
            self._store_video_metadata(metadata)
            
            self.logger.info(f"Stored video {video_id} from {video_path} at {location} ({recording_timestamp})")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error storing video: {e}")
            # Clean up on error
            if dest_path.exists():
                dest_path.unlink()
            raise
    
    async def _copy_file_async(self, src: Path, dst: Path):
        """Copy file asynchronously."""
        chunk_size = 1024 * 1024  # 1MB chunks
        
        async with aiofiles.open(src, 'rb') as src_file:
            async with aiofiles.open(dst, 'wb') as dst_file:
                while chunk := await src_file.read(chunk_size):
                    await dst_file.write(chunk)
    
    def _extract_location_from_path(self, video_path: Path) -> str:
        """Extract location from video path or filename.
        
        Tries to extract location from:
        1. Parent directory name
        2. Filename pattern (e.g., shed_video.mp4)
        3. Default to "unknown"
        """
        # Check parent directory
        parent_name = video_path.parent.name.lower()
        common_locations = ['shed', 'driveway', 'front_door', 'backyard', 'garage', 'entrance']
        
        for location in common_locations:
            if location in parent_name:
                return location
        
        # Check filename
        filename_lower = video_path.stem.lower()
        for location in common_locations:
            if location in filename_lower:
                return location
        
        # Default
        return "unknown"
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location name for consistency."""
        # Remove special characters and spaces
        normalized = location.lower().strip()
        normalized = normalized.replace(' ', '_')
        normalized = ''.join(c for c in normalized if c.isalnum() or c == '_')
        
        # Common aliases
        aliases = {
            'front': 'front_door',
            'back': 'backyard',
            'drive': 'driveway',
            'car': 'driveway'
        }
        
        for alias, standard in aliases.items():
            if alias in normalized:
                return standard
        
        return normalized or "unknown"
    
    def _extract_timestamp_from_file(self, video_path: Path) -> datetime:
        """Extract recording timestamp from file.
        
        Tries:
        1. Parse from filename (e.g., 2024-12-06_143022.mp4)
        2. Use file modification time
        """
        import re
        
        # Try to extract from filename
        filename = video_path.stem
        
        # Common timestamp patterns
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})',  # YYYY-MM-DD_HHMMSS
            r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',    # YYYYMMDD_HHMMSS
            r'(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})', # YYYY-MM-DD HH-MM-SS
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 6:
                        year, month, day, hour, minute, second = map(int, groups)
                        return datetime(year, month, day, hour, minute, second)
                except (ValueError, TypeError):
                    continue
        
        # Fall back to file modification time
        stat = video_path.stat()
        return datetime.fromtimestamp(stat.st_mtime)
    
    def _store_video_metadata(self, metadata: VideoMetadata):
        """Store video metadata in database."""
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO videos (
                    video_id, original_path, filename, location, recording_timestamp,
                    duration, fps, width, height, codec, size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.video_id,
                metadata.original_path,
                metadata.filename,
                metadata.location,
                metadata.recording_timestamp.isoformat(),
                metadata.duration,
                metadata.fps,
                metadata.width,
                metadata.height,
                metadata.codec,
                metadata.size_bytes,
                metadata.created_at.isoformat()
            ))
            conn.commit()
    
    def get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Get video metadata by ID."""
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
            row = cursor.fetchone()
            
            if row:
                return VideoMetadata(
                    video_id=row['video_id'],
                    original_path=row['original_path'],
                    filename=row['filename'],
                    location=row['location'],
                    recording_timestamp=datetime.fromisoformat(row['recording_timestamp']),
                    duration=row['duration'],
                    fps=row['fps'],
                    width=row['width'],
                    height=row['height'],
                    codec=row['codec'],
                    size_bytes=row['size_bytes'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    processed_at=datetime.fromisoformat(row['processed_at']) if row['processed_at'] else None
                )
        return None
    
    def update_video_status(self, video_id: str, status: ProcessingStatus, error_message: Optional[str] = None, processing_time: Optional[float] = None):
        """Update video processing status."""
        with self._get_db() as conn:
            cursor = conn.cursor()
            
            if status == ProcessingStatus.COMPLETED:
                if processing_time is not None:
                    cursor.execute("""
                        UPDATE videos 
                        SET status = ?, processed_at = ?, error_message = NULL, processing_time = ?
                        WHERE video_id = ?
                    """, (status.value, datetime.now().isoformat(), processing_time, video_id))
                else:
                    cursor.execute("""
                        UPDATE videos 
                        SET status = ?, processed_at = ?, error_message = NULL
                        WHERE video_id = ?
                    """, (status.value, datetime.now().isoformat(), video_id))
            else:
                cursor.execute("""
                    UPDATE videos 
                    SET status = ?, error_message = ?
                    WHERE video_id = ?
                """, (status.value, error_message, video_id))
            
            conn.commit()
    
    def store_frame_analysis(self, video_id: str, analyses: List[FrameAnalysis]):
        """Store frame analysis results."""
        with self._get_db() as conn:
            cursor = conn.cursor()
            
            for analysis in analyses:
                cursor.execute("""
                    INSERT OR REPLACE INTO frame_analysis (
                        video_id, frame_number, timestamp, frame_path,
                        description, objects_detected, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    analysis.frame_number,
                    analysis.timestamp,
                    analysis.frame_path,
                    analysis.description,
                    json.dumps(analysis.objects_detected),
                    analysis.confidence
                ))
            
            conn.commit()
            self.logger.debug(f"Stored {len(analyses)} frame analyses for video {video_id}")
    
    def store_transcript(self, video_id: str, transcript: str, language: str = "en"):
        """Store video transcript."""
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO transcripts (video_id, transcript, language)
                VALUES (?, ?, ?)
            """, (video_id, transcript, language))
            conn.commit()
            
        # Also store in search index
        self._update_search_index(video_id, transcript, "transcript")
    
    def _update_search_index(self, video_id: str, content: str, content_type: str,
                           timestamp_start: Optional[float] = None,
                           timestamp_end: Optional[float] = None):
        """Update search index with content."""
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO search_index (
                    video_id, content, content_type, timestamp_start, timestamp_end
                ) VALUES (?, ?, ?, ?, ?)
            """, (video_id, content, content_type, timestamp_start, timestamp_end))
            conn.commit()
    
    def get_processing_result(self, video_id: str) -> Optional[ProcessingResult]:
        """Get processing result for a video."""
        metadata = self.get_video_metadata(video_id)
        if not metadata:
            return None
        
        # Get frame analyses
        with self._get_db() as conn:
            cursor = conn.cursor()
            
            # Get processing time
            cursor.execute("SELECT processing_time FROM videos WHERE video_id = ?", (video_id,))
            time_row = cursor.fetchone()
            processing_time = time_row['processing_time'] if time_row and time_row['processing_time'] else 0.0
            
            # Get frames
            cursor.execute("""
                SELECT * FROM frame_analysis 
                WHERE video_id = ? 
                ORDER BY frame_number
            """, (video_id,))
            
            frames = []
            for row in cursor.fetchall():
                frames.append(FrameAnalysis(
                    frame_number=row['frame_number'],
                    timestamp=row['timestamp'],
                    frame_path=row['frame_path'],
                    description=row['description'],
                    objects_detected=json.loads(row['objects_detected'] or '[]'),
                    confidence=row['confidence'] or 0.0
                ))
            
            # Get transcript
            cursor.execute("SELECT transcript FROM transcripts WHERE video_id = ?", (video_id,))
            transcript_row = cursor.fetchone()
            transcript = transcript_row['transcript'] if transcript_row else None
        
        return ProcessingResult(
            video_id=video_id,
            status=ProcessingStatus.COMPLETED,
            frames_extracted=len(frames),
            frames_analyzed=len([f for f in frames if f.description]),
            transcript=transcript,
            timeline=frames,
            processing_time=processing_time
        )
    
    def search_videos(self, query: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Search videos by content."""
        with self._get_db() as conn:
            cursor = conn.cursor()
            
            # Simple search - in production, use FTS5 or vector search
            cursor.execute("""
                SELECT DISTINCT s.video_id, v.filename, 
                       COUNT(*) as match_count
                FROM search_index s
                JOIN videos v ON s.video_id = v.video_id
                WHERE s.content LIKE ?
                GROUP BY s.video_id
                ORDER BY match_count DESC
                LIMIT ?
            """, (f"%{query}%", limit))
            
            return [(row['video_id'], row['filename'], row['match_count']) 
                    for row in cursor.fetchall()]
    
    def query_videos_by_location_and_time(
        self, 
        location: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        content_query: Optional[str] = None,
        limit: int = 50
    ) -> List[VideoMetadata]:
        """Query videos by location and time range.
        
        Args:
            location: Location to filter by (None for all locations)
            start_time: Start of time range
            end_time: End of time range
            content_query: Optional content search query
            limit: Maximum results
            
        Returns:
            List of VideoMetadata objects matching criteria
        """
        with self._get_db() as conn:
            cursor = conn.cursor()
            
            # Build query
            query_parts = ["SELECT DISTINCT v.* FROM videos v"]
            conditions = ["v.status = ?"]
            params = [ProcessingStatus.COMPLETED.value]
            
            # Join with search index if content query provided
            if content_query:
                query_parts.append("JOIN search_index s ON v.video_id = s.video_id")
                conditions.append("s.content LIKE ?")
                params.append(f"%{content_query}%")
            
            # Location filter
            if location:
                normalized_location = self._normalize_location(location)
                conditions.append("v.location = ?")
                params.append(normalized_location)
            
            # Time range filter
            if start_time:
                conditions.append("v.recording_timestamp >= ?")
                params.append(start_time.isoformat())
            
            if end_time:
                conditions.append("v.recording_timestamp <= ?")
                params.append(end_time.isoformat())
            
            # Combine query
            if conditions:
                query_parts.append("WHERE " + " AND ".join(conditions))
            
            query_parts.append("ORDER BY v.recording_timestamp DESC")
            query_parts.append("LIMIT ?")
            params.append(limit)
            
            query = " ".join(query_parts)
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append(VideoMetadata(
                    video_id=row['video_id'],
                    original_path=row['original_path'],
                    filename=row['filename'],
                    location=row['location'],
                    recording_timestamp=datetime.fromisoformat(row['recording_timestamp']),
                    duration=row['duration'],
                    fps=row['fps'],
                    width=row['width'],
                    height=row['height'],
                    codec=row['codec'],
                    size_bytes=row['size_bytes'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    processed_at=datetime.fromisoformat(row['processed_at']) if row['processed_at'] else None
                ))
            
            return results
    
    def cleanup_old_files(self, days: Optional[int] = None):
        """Clean up old processed files."""
        days = days or self.config.storage.cleanup_after_days
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT video_id FROM videos 
                WHERE created_at < ? AND status = ?
            """, (cutoff_date.isoformat(), ProcessingStatus.COMPLETED.value))
            
            for row in cursor.fetchall():
                video_id = row['video_id']
                # Remove processed files
                processed_dir = self.base_path / "processed" / video_id
                if processed_dir.exists():
                    shutil.rmtree(processed_dir)
                    self.logger.info(f"Cleaned up processed files for {video_id}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_videos": 0,
            "processed_videos": 0,
            "total_size_gb": 0.0,
            "processed_size_gb": 0.0
        }
        
        # Count videos
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM videos")
            stats["total_videos"] = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM videos 
                WHERE status = ?
            """, (ProcessingStatus.COMPLETED.value,))
            stats["processed_videos"] = cursor.fetchone()['count']
        
        # Calculate sizes
        for video_file in (self.base_path / "originals").glob("*"):
            if video_file.is_file():
                stats["total_size_gb"] += video_file.stat().st_size / (1024**3)
        
        for processed_dir in (self.base_path / "processed").iterdir():
            if processed_dir.is_dir():
                for file in processed_dir.rglob("*"):
                    if file.is_file():
                        stats["processed_size_gb"] += file.stat().st_size / (1024**3)
        
        return stats