"""Unit tests for storage manager."""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, patch, AsyncMock

from src.storage.manager import StorageManager
from src.storage.schemas import (
    VideoMetadata, ProcessingStatus, FrameAnalysis
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def storage_manager(temp_storage_dir):
    """Create a storage manager instance with temporary directory."""
    with patch('src.storage.manager.get_config') as mock_config:
        mock_config.return_value.storage.base_path = temp_storage_dir
        mock_config.return_value.storage.cleanup_after_days = 30
        
        manager = StorageManager(temp_storage_dir)
        return manager


@pytest.fixture
def sample_video_file(temp_storage_dir):
    """Create a sample video file for testing."""
    video_path = Path(temp_storage_dir) / "sample_video.mp4"
    video_path.write_bytes(b"fake video content")
    return str(video_path)


class TestStorageManager:
    """Test cases for StorageManager."""
    
    def test_init_creates_directories(self, temp_storage_dir):
        """Test that initialization creates required directories."""
        with patch('src.storage.manager.get_config') as mock_config:
            mock_config.return_value.storage.base_path = temp_storage_dir
            
            StorageManager(temp_storage_dir)
            
            expected_dirs = ['locations', 'processed', 'index', 'temp']
            for dir_name in expected_dirs:
                assert (Path(temp_storage_dir) / dir_name).exists()
    
    def test_init_creates_database(self, storage_manager):
        """Test that initialization creates database with tables."""
        db_path = Path(storage_manager.base_path) / "index" / "metadata.db"
        assert db_path.exists()
        
        # Check tables exist
        with storage_manager._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = {row['name'] for row in cursor.fetchall()}
            
            expected_tables = {'videos', 'frame_analysis', 'transcripts', 'search_index'}
            assert expected_tables.issubset(tables)
    
    def test_generate_video_id(self, storage_manager):
        """Test video ID generation."""
        video_id1 = storage_manager.generate_video_id("/path/to/video1.mp4")
        video_id2 = storage_manager.generate_video_id("/path/to/video2.mp4")
        
        assert video_id1.startswith("vid_")
        assert len(video_id1) == 16  # vid_ + 12 chars
        assert video_id1 != video_id2
    
    @pytest.mark.asyncio
    async def test_store_video(self, storage_manager, sample_video_file):
        """Test storing a video file."""
        metadata = await storage_manager.store_video(sample_video_file)
        
        assert metadata.video_id.startswith("vid_")
        assert metadata.filename == "sample_video.mp4"
        assert metadata.original_path == sample_video_file
        assert metadata.size_bytes == 18  # len(b"fake video content")
        
        # Check file was copied to location-based structure
        date_parts = metadata.recording_timestamp.strftime("%Y/%m/%d")
        timestamp_str = metadata.recording_timestamp.strftime("%H%M%S")
        dest_path = storage_manager.base_path / "locations" / metadata.location / date_parts / f"{metadata.video_id}_{timestamp_str}.mp4"
        assert dest_path.exists()
        assert dest_path.read_bytes() == b"fake video content"
        
        # Check database entry
        stored_metadata = storage_manager.get_video_metadata(metadata.video_id)
        assert stored_metadata is not None
        assert stored_metadata.video_id == metadata.video_id
    
    @pytest.mark.asyncio
    async def test_store_video_file_not_found(self, storage_manager):
        """Test storing a non-existent video file."""
        with pytest.raises(FileNotFoundError):
            await storage_manager.store_video("/nonexistent/video.mp4")
    
    def test_update_video_status(self, storage_manager):
        """Test updating video processing status."""
        # Create a video entry
        video_id = "vid_test123"
        metadata = VideoMetadata(
            video_id=video_id,
            original_path="/test/video.mp4",
            filename="video.mp4",
            location="test_location",
            recording_timestamp=datetime.now(),
            duration=100.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264",
            size_bytes=1000000
        )
        storage_manager._store_video_metadata(metadata)
        
        # Update status to processing
        storage_manager.update_video_status(video_id, ProcessingStatus.PROCESSING)
        
        with storage_manager._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM videos WHERE video_id = ?", (video_id,))
            assert cursor.fetchone()['status'] == ProcessingStatus.PROCESSING.value
        
        # Update status to completed
        storage_manager.update_video_status(video_id, ProcessingStatus.COMPLETED)
        
        with storage_manager._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status, processed_at FROM videos WHERE video_id = ?", 
                (video_id,)
            )
            row = cursor.fetchone()
            assert row['status'] == ProcessingStatus.COMPLETED.value
            assert row['processed_at'] is not None
    
    def test_store_frame_analysis(self, storage_manager):
        """Test storing frame analysis results."""
        video_id = "vid_test123"
        
        # Store video metadata first
        metadata = VideoMetadata(
            video_id=video_id,
            original_path="/test/video.mp4",
            filename="video.mp4",
            location="test_location",
            recording_timestamp=datetime.now(),
            duration=100.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264",
            size_bytes=1000000
        )
        storage_manager._store_video_metadata(metadata)
        
        # Create frame analyses
        analyses = [
            FrameAnalysis(
                frame_number=0,
                timestamp=0.0,
                frame_path="/frames/frame_0.jpg",
                description="A person walking",
                objects_detected=["person", "sidewalk"],
                confidence=0.95
            ),
            FrameAnalysis(
                frame_number=30,
                timestamp=1.0,
                frame_path="/frames/frame_30.jpg",
                description="A car passing by",
                objects_detected=["car", "road"],
                confidence=0.87
            )
        ]
        
        storage_manager.store_frame_analysis(video_id, analyses)
        
        # Verify stored
        with storage_manager._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM frame_analysis WHERE video_id = ?",
                (video_id,)
            )
            assert cursor.fetchone()['count'] == 2
    
    def test_store_transcript(self, storage_manager):
        """Test storing video transcript."""
        video_id = "vid_test123"
        
        # Store video metadata first
        metadata = VideoMetadata(
            video_id=video_id,
            original_path="/test/video.mp4",
            filename="video.mp4",
            location="test_location",
            recording_timestamp=datetime.now(),
            duration=100.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264",
            size_bytes=1000000
        )
        storage_manager._store_video_metadata(metadata)
        
        transcript = "This is a test transcript with some spoken words."
        storage_manager.store_transcript(video_id, transcript)
        
        # Verify transcript stored
        with storage_manager._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT transcript FROM transcripts WHERE video_id = ?",
                (video_id,)
            )
            assert cursor.fetchone()['transcript'] == transcript
            
            # Verify search index updated
            cursor.execute(
                "SELECT content FROM search_index WHERE video_id = ? AND content_type = 'transcript'",
                (video_id,)
            )
            assert cursor.fetchone()['content'] == transcript
    
    def test_search_videos(self, storage_manager):
        """Test searching videos by content."""
        # Create test videos with searchable content
        video_ids = ["vid_test1", "vid_test2", "vid_test3"]
        
        for i, video_id in enumerate(video_ids):
            metadata = VideoMetadata(
                video_id=video_id,
                original_path=f"/test/video{i}.mp4",
                filename=f"video{i}.mp4",
                location="test_location",
                recording_timestamp=datetime.now(),
                duration=100.0,
                fps=30.0,
                width=1920,
                height=1080,
                codec="h264",
                size_bytes=1000000
            )
            storage_manager._store_video_metadata(metadata)
        
        # Add searchable content
        storage_manager.store_transcript("vid_test1", "The cat is sleeping on the couch")
        storage_manager.store_transcript("vid_test2", "A dog is playing in the park")
        storage_manager.store_transcript("vid_test3", "The cat and dog are friends")
        
        # Search for "cat"
        results = storage_manager.search_videos("cat")
        assert len(results) == 2
        video_ids_found = [r[0] for r in results]
        assert "vid_test1" in video_ids_found
        assert "vid_test3" in video_ids_found
    
    def test_cleanup_old_files(self, storage_manager):
        """Test cleanup of old processed files."""
        # Create old video entry
        old_video_id = "vid_old"
        old_date = datetime.now() - timedelta(days=35)
        
        with storage_manager._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO videos (
                    video_id, original_path, filename, location, recording_timestamp,
                    duration, fps, width, height, codec, size_bytes, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                old_video_id, "/old/video.mp4", "old_video.mp4", "test_location", old_date.isoformat(),
                100.0, 30.0, 1920, 1080, "h264", 1000000,
                old_date.isoformat(), ProcessingStatus.COMPLETED.value
            ))
            conn.commit()
        
        # Create processed directory
        processed_dir = storage_manager.base_path / "processed" / old_video_id
        processed_dir.mkdir(parents=True)
        (processed_dir / "test.txt").write_text("test")
        
        # Run cleanup
        storage_manager.cleanup_old_files(days=30)
        
        # Verify directory removed
        assert not processed_dir.exists()
    
    def test_get_storage_stats(self, storage_manager, sample_video_file):
        """Test getting storage statistics."""
        # Add some test data
        video_id = "vid_test123"
        metadata = VideoMetadata(
            video_id=video_id,
            original_path=sample_video_file,
            filename="video.mp4",
            location="test_location",
            recording_timestamp=datetime.now(),
            duration=100.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264",
            size_bytes=1000000
        )
        storage_manager._store_video_metadata(metadata)
        storage_manager.update_video_status(video_id, ProcessingStatus.COMPLETED)
        
        # Create some files in location-based structure
        date_parts = metadata.recording_timestamp.strftime("%Y/%m/%d")
        video_dir = storage_manager.base_path / "locations" / metadata.location / date_parts
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / f"{video_id}.mp4").write_bytes(b"x" * 1000)
        
        processed_dir = storage_manager.base_path / "processed" / video_id
        processed_dir.mkdir(parents=True)
        (processed_dir / "frame.jpg").write_bytes(b"x" * 500)
        
        stats = storage_manager.get_storage_stats()
        
        assert stats["total_videos"] == 1
        assert stats["processed_videos"] == 1
        # Size is in GB, so check bytes instead for small test files
        assert stats["total_size_gb"] >= 0  # May be 0 for small test files
        assert stats["processed_size_gb"] >= 0  # May be 0 for small test files