"""Integration tests for complete video processing flow."""
import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, Mock
from datetime import datetime

from src.storage.manager import StorageManager
from src.processors.video import VideoProcessor
from src.llm.ollama_client import OllamaClient
from src.storage.schemas import ProcessingStatus

import pytest_asyncio

@pytest_asyncio.fixture
async def test_environment():
    """Set up test environment with all components."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Mock config
    mock_config = Mock()
    mock_config.storage.base_path = temp_dir
    mock_config.storage.cleanup_after_days = 30
    mock_config.processing.frame_sample_rate = 60  # Faster for testing
    mock_config.processing.enable_scene_detection = False
    mock_config.processing.max_frames_per_video = 10
    mock_config.processing.frame_quality = 85
    mock_config.processing.audio.enable_timestamps = True
    mock_config.processing.audio.model = "base"
    mock_config.processing.audio.language = "en"
    mock_config.performance.thread_count = 2
    mock_config.llm.ollama_host = "http://localhost:11434"
    mock_config.llm.vision_model = "llava:latest"
    mock_config.llm.text_model = "llama2:latest"
    mock_config.llm.temperature = 0.7
    mock_config.llm.timeout_seconds = 30
    
    with patch('src.storage.manager.get_config', return_value=mock_config), \
         patch('src.processors.video.get_config', return_value=mock_config), \
         patch('src.llm.ollama_client.get_config', return_value=mock_config):
        
        # Initialize components
        storage = StorageManager(temp_dir)
        llm_client = OllamaClient()
        processor = VideoProcessor(storage, llm_client=llm_client)
        
        yield {
            'storage': storage,
            'processor': processor,
            'llm_client': llm_client,
            'temp_dir': temp_dir,
            'config': mock_config
        }
        
        # Cleanup
        processor.cleanup()
        await llm_client.close()
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_video_path():
    """Get path to sample video."""
    # Check if sample video exists in originals or locations directory
    video_path = Path(__file__).parent.parent.parent / "video_data" / "originals" / "sample_video.mp4"
    if video_path.exists():
        return video_path
    
    # Check in locations directory (for migrated videos)
    locations_base = Path(__file__).parent.parent.parent / "video_data" / "locations"
    if locations_base.exists():
        # Search for sample_video.mp4 in locations subdirectories
        for video_file in locations_base.rglob("sample_video.mp4"):
            return video_file
    
    # Skip test if no sample video
    pytest.skip("Sample video not found. Download it first.")


class TestVideoProcessingFlow:
    """Integration tests for video processing."""
    
    @pytest.mark.asyncio
    async def test_complete_video_processing(self, test_environment, sample_video_path):
        """Test complete video processing workflow."""
        storage = test_environment['storage']
        processor = test_environment['processor']
        
        # Store video
        metadata = await storage.store_video(str(sample_video_path))
        assert metadata.video_id.startswith("vid_")
        
        # Verify video was copied to location-based structure
        # Extract date parts from recording timestamp
        date_parts = metadata.recording_timestamp.strftime("%Y/%m/%d")
        timestamp_str = metadata.recording_timestamp.strftime("%H%M%S")
        video_file = storage.base_path / "locations" / metadata.location / date_parts / f"{metadata.video_id}_{timestamp_str}.mp4"
        assert video_file.exists()
        
        # Mock whisper transcription
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'text': 'This is a sample video with some audio content.'
            }
            mock_load_model.return_value = mock_model
            
            # Process video
            result = await processor.process_video(metadata.video_id)
            
            assert result.status == ProcessingStatus.COMPLETED
            assert result.frames_extracted > 0
            assert result.frames_extracted <= test_environment['config'].processing.max_frames_per_video
            assert result.transcript is not None
            
            # Verify frames were extracted
            frames_dir = storage.base_path / "processed" / metadata.video_id / "frames"
            assert frames_dir.exists()
            frame_files = list(frames_dir.glob("*.jpg"))
            assert len(frame_files) == result.frames_extracted
            
            # Verify database was updated
            stored_result = storage.get_processing_result(metadata.video_id)
            assert stored_result is not None
            assert stored_result.transcript == result.transcript
    
    @pytest.mark.asyncio
    async def test_video_analysis_with_llm(self, test_environment, sample_video_path):
        """Test video analysis with LLM integration."""
        storage = test_environment['storage']
        processor = test_environment['processor']
        llm_client = test_environment['llm_client']
        
        # Check if Ollama is available
        if not await llm_client.is_available():
            pytest.skip("Ollama is not running. Start it with 'ollama serve'")
        
        # Store and process video
        metadata = await storage.store_video(str(sample_video_path))
        
        # Mock the frame analysis to avoid requiring LLaVA model
        from src.storage.schemas import FrameAnalysis
        with patch.object(processor, '_analyze_frames_placeholder') as mock_analyze:
            mock_analyze.return_value = [
                FrameAnalysis(
                    frame_number=0,
                    timestamp=0.0,
                    frame_path='frame_0.jpg',
                    description='Animated characters in a forest scene',
                    objects_detected=['trees', 'characters'],
                    confidence=0.9
                ),
                FrameAnalysis(
                    frame_number=1,
                    timestamp=1.0,
                    frame_path='frame_1.jpg',
                    description='Characters walking through the forest',
                    objects_detected=['trees', 'path', 'characters'],
                    confidence=0.85
                )
            ]
            
            # Process video
            result = await processor.process_video(metadata.video_id)
            assert result.status == ProcessingStatus.COMPLETED
            
            # Test Q&A functionality
            question = "What is happening in this video?"
            
            # Mock LLM response
            with patch.object(llm_client, 'answer_video_question') as mock_answer:
                mock_answer.return_value = "The video shows animated characters in a forest scene."
                
                answer = await llm_client.answer_video_question(
                    question,
                    "Frame descriptions show forest scenes with characters"
                )
                
                assert answer is not None
                assert "forest" in answer.lower()
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, test_environment):
        """Test video search functionality."""
        storage = test_environment['storage']
        
        # Create test videos with different content
        test_videos = [
            ("video1", "A cat playing with a ball in the living room"),
            ("video2", "Dogs running in the park"),
            ("video3", "A cat and dog sleeping together")
        ]
        
        for video_id, transcript in test_videos:
            # Create minimal video metadata
            from src.storage.schemas import VideoMetadata
            metadata = VideoMetadata(
                video_id=video_id,
                original_path=f'/test/{video_id}.mp4',
                filename=f'{video_id}.mp4',
                location='test_location',
                recording_timestamp=datetime.now(),
                duration=10.0,
                fps=30.0,
                width=640,
                height=480,
                codec='h264',
                size_bytes=1000000
            )
            storage._store_video_metadata(metadata)
            
            # Store transcript
            storage.store_transcript(video_id, transcript)
        
        # Search for "cat"
        results = storage.search_videos("cat")
        assert len(results) == 2
        video_ids = [r[0] for r in results]
        assert "video1" in video_ids
        assert "video3" in video_ids
        
        # Search for "park"
        results = storage.search_videos("park")
        assert len(results) == 1
        assert results[0][0] == "video2"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_environment):
        """Test error handling in video processing."""
        storage = test_environment['storage']
        processor = test_environment['processor']
        
        # Try to process non-existent video
        with pytest.raises(ValueError):
            await processor.process_video("non_existent_id")
        
        # Create video metadata without actual file
        from src.storage.schemas import VideoMetadata
        fake_metadata = VideoMetadata(
            video_id='vid_fake123',
            original_path='/fake/video.mp4',
            filename='fake.mp4',
            location='test_location',
            recording_timestamp=datetime.now(),
            duration=10.0,
            fps=30.0,
            width=640,
            height=480,
            codec='h264',
            size_bytes=1000000
        )
        storage._store_video_metadata(fake_metadata)
        
        # Try to process - should fail and update status
        with pytest.raises(FileNotFoundError):
            await processor.process_video('vid_fake123')
        
        # Check status was updated
        with storage._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, error_message FROM videos WHERE video_id = ?", ('vid_fake123',))
            row = cursor.fetchone()
            assert row['status'] == ProcessingStatus.FAILED.value
            assert row['error_message'] is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, test_environment, sample_video_path):
        """Test processing multiple videos concurrently."""
        storage = test_environment['storage']
        processor = test_environment['processor']
        
        # Limit concurrent processing for testing
        test_environment['config'].performance.max_concurrent_videos = 2
        
        # Store multiple copies of the video
        video_ids = []
        for i in range(3):
            metadata = await storage.store_video(str(sample_video_path))
            video_ids.append(metadata.video_id)
        
        # Mock whisper
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = {'text': 'Test transcript'}
            mock_load_model.return_value = mock_model
            
            # Process videos concurrently
            tasks = [processor.process_video(vid) for vid in video_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check all completed successfully
            for result in results:
                if isinstance(result, Exception):
                    pytest.fail(f"Processing failed: {result}")
                assert result.status == ProcessingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_storage_cleanup(self, test_environment):
        """Test storage cleanup functionality."""
        storage = test_environment['storage']
        
        # Create old video
        from datetime import datetime, timedelta
        old_date = datetime.now() - timedelta(days=35)
        
        with storage._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO videos (
                    video_id, original_path, filename, location, recording_timestamp,
                    duration, fps, width, height, codec, size_bytes, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'vid_old123', '/old/video.mp4', 'old.mp4', 'test_location', old_date.isoformat(),
                10.0, 30.0, 640, 480, 'h264', 1000000,
                old_date.isoformat(), ProcessingStatus.COMPLETED.value
            ))
            conn.commit()
        
        # Create processed files
        old_dir = storage.base_path / "processed" / "vid_old123"
        old_dir.mkdir(parents=True)
        (old_dir / "test.txt").write_text("old data")
        
        # Run cleanup
        storage.cleanup_old_files(days=30)
        
        # Verify cleaned up
        assert not old_dir.exists()