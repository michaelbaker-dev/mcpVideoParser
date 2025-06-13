"""Unit tests for video processor."""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import cv2
import numpy as np

from src.processors.video import VideoProcessor
from src.storage.schemas import VideoMetadata, ProcessingStatus


@pytest.fixture
def mock_storage():
    """Create a mock storage manager."""
    storage = Mock()
    storage.base_path = Path(tempfile.mkdtemp())
    yield storage
    shutil.rmtree(storage.base_path)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.processing.frame_sample_rate = 30
    config.processing.enable_scene_detection = False
    config.processing.scene_threshold = 0.3
    config.processing.max_frames_per_video = 100
    config.processing.frame_quality = 85
    config.processing.frame_format = "jpg"
    config.processing.audio.enable_timestamps = True
    config.processing.audio.model = "base"
    config.processing.audio.language = "en"
    config.performance.thread_count = 4
    return config


@pytest.fixture
def video_processor(mock_storage, mock_config):
    """Create a video processor instance."""
    with patch('src.processors.video.get_config', return_value=mock_config):
        processor = VideoProcessor(mock_storage, llm_client=None)
        yield processor
        processor.cleanup()


@pytest.fixture
def sample_video_metadata():
    """Create sample video metadata."""
    from datetime import datetime
    return VideoMetadata(
        video_id="vid_test123",
        original_path="/test/video.mp4",
        filename="test_video.mp4",
        location="test_location",
        recording_timestamp=datetime.now(),
        duration=10.0,
        fps=30.0,
        width=640,
        height=480,
        codec="h264",
        size_bytes=1000000
    )


def create_test_video(path: Path, duration_seconds: int = 1, fps: int = 30):
    """Create a simple test video file."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    
    # Generate frames
    for i in range(fps * duration_seconds):
        # Create a frame with changing color
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color_value = int((i / (fps * duration_seconds)) * 255)
        frame[:, :] = (color_value, 255 - color_value, 128)
        
        # Add frame number text
        cv2.putText(frame, f"Frame {i}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()


class TestVideoProcessor:
    """Test cases for VideoProcessor."""
    
    def test_init(self, mock_storage, mock_config):
        """Test video processor initialization."""
        with patch('src.processors.video.get_config', return_value=mock_config):
            processor = VideoProcessor(mock_storage)
            assert processor.storage == mock_storage
            assert processor.config == mock_config
            processor.cleanup()
    
    def test_get_video_path(self, video_processor, mock_storage):
        """Test getting video file path."""
        from datetime import datetime
        from src.storage.schemas import VideoMetadata
        
        video_id = "vid_test123"
        
        # Create mock metadata
        metadata = VideoMetadata(
            video_id=video_id,
            original_path="/test/video.mp4",
            filename="test_video.mp4",
            location="test_location",
            recording_timestamp=datetime(2025, 6, 12, 10, 30, 0),
            duration=10.0,
            fps=30.0,
            width=640,
            height=480,
            codec="h264",
            size_bytes=1000000
        )
        mock_storage.get_video_metadata.return_value = metadata
        
        # Create test video file in location-based structure
        date_parts = metadata.recording_timestamp.strftime("%Y/%m/%d")
        timestamp_str = metadata.recording_timestamp.strftime("%H%M%S")
        video_path = mock_storage.base_path / "locations" / metadata.location / date_parts / f"{video_id}_{timestamp_str}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
        
        found_path = video_processor._get_video_path(video_id)
        
        # The method should try various filename patterns
        assert found_path.parent == video_path.parent
        assert found_path.name.startswith(video_id)
    
    def test_get_video_path_not_found(self, video_processor, mock_storage):
        """Test getting video path when file doesn't exist."""
        # Mock get_video_metadata to return None for nonexistent video
        mock_storage.get_video_metadata.return_value = None
        
        with pytest.raises(FileNotFoundError):
            video_processor._get_video_path("nonexistent_id")
    
    @pytest.mark.asyncio
    async def test_extract_frames(self, video_processor, mock_storage):
        """Test frame extraction from video."""
        # Create test video
        video_path = mock_storage.base_path / "test_video.mp4"
        create_test_video(video_path, duration_seconds=1, fps=30)
        
        # Create output directory
        output_dir = mock_storage.base_path / "processed" / "test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        frame_paths = await video_processor._extract_frames(video_path, output_dir)
        
        # Check frames were extracted
        assert len(frame_paths) > 0
        assert all(p.exists() for p in frame_paths)
        
        # Check frame directory structure
        frames_dir = output_dir / "frames"
        assert frames_dir.exists()
        assert len(list(frames_dir.glob("*.jpg"))) == len(frame_paths)
    
    @pytest.mark.asyncio
    async def test_extract_frames_with_scene_detection(self, video_processor, mock_storage, mock_config):
        """Test frame extraction with scene detection enabled."""
        mock_config.processing.enable_scene_detection = True
        
        # Create test video
        video_path = mock_storage.base_path / "test_video.mp4"
        create_test_video(video_path, duration_seconds=2, fps=30)
        
        # Create output directory
        output_dir = mock_storage.base_path / "processed" / "test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        frame_paths = await video_processor._extract_frames(video_path, output_dir)
        
        # Should have extracted some frames
        assert len(frame_paths) > 0
        assert len(frame_paths) <= mock_config.processing.max_frames_per_video
    
    @pytest.mark.asyncio
    async def test_detect_scene_changes(self, video_processor, mock_storage):
        """Test scene change detection."""
        # Create test video with scene changes
        video_path = mock_storage.base_path / "test_video.mp4"
        width, height = 640, 480
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Create frames with distinct scenes
        total_frames = 90
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Change color every 30 frames (1 second)
            if i < 30:
                frame[:, :] = (255, 0, 0)  # Red
            elif i < 60:
                frame[:, :] = (0, 255, 0)  # Green
            else:
                frame[:, :] = (0, 0, 255)  # Blue
            
            out.write(frame)
        
        out.release()
        
        # Detect scene changes
        scene_frames = await video_processor._detect_scene_changes(video_path, total_frames)
        
        # Should detect at least the beginning and scene changes
        assert len(scene_frames) >= 3
        assert 0 in scene_frames  # First frame
        assert (total_frames - 1) in scene_frames  # Last frame
    
    @pytest.mark.asyncio
    async def test_extract_and_transcribe_audio(self, video_processor, mock_storage):
        """Test audio extraction and transcription."""
        # This test requires ffmpeg and whisper to be installed
        # Mock the whisper transcription
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'text': 'This is a test transcription.'
            }
            mock_load_model.return_value = mock_model
            
            # Create a simple video with audio (mocked)
            video_path = mock_storage.base_path / "test_video.mp4"
            video_path.touch()
            
            output_dir = mock_storage.base_path / "processed" / "test"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Mock ffmpeg subprocess
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b'', b'')
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                # Create dummy audio file that ffmpeg would create
                audio_path = output_dir / "audio.wav"
                audio_path.write_bytes(b'dummy audio data')
                
                transcript = await video_processor._extract_and_transcribe_audio(
                    video_path, output_dir
                )
                
                assert transcript == 'This is a test transcription.'
                assert not audio_path.exists()  # Should be cleaned up
    
    @pytest.mark.asyncio
    async def test_analyze_frames_placeholder(self, video_processor, mock_storage):
        """Test placeholder frame analysis."""
        # Create test frame paths
        frames_dir = mock_storage.base_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        for i in range(3):
            timestamp = i * 1.0
            frame_path = frames_dir / f"frame_{i:05d}_t{timestamp:.1f}s.jpg"
            frame_path.touch()
            frame_paths.append(frame_path)
        
        # Analyze frames
        analyses = await video_processor._analyze_frames_placeholder(
            frame_paths, "vid_test123"
        )
        
        assert len(analyses) == 3
        for i, analysis in enumerate(analyses):
            assert analysis.frame_number == i
            assert analysis.timestamp == i * 1.0
            assert analysis.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_process_video_complete(self, video_processor, mock_storage, sample_video_metadata):
        """Test complete video processing workflow."""
        video_id = sample_video_metadata.video_id
        
        # Setup mocks
        mock_storage.get_video_metadata.return_value = sample_video_metadata
        mock_storage.update_video_status = Mock()
        mock_storage.store_transcript = Mock()
        mock_storage.store_frame_analysis = Mock()
        
        # Create test video in location-based structure
        date_parts = sample_video_metadata.recording_timestamp.strftime("%Y/%m/%d")
        timestamp_str = sample_video_metadata.recording_timestamp.strftime("%H%M%S")
        video_dir = mock_storage.base_path / "locations" / sample_video_metadata.location / date_parts
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{video_id}_{timestamp_str}.mp4"
        create_test_video(video_path, duration_seconds=1, fps=30)
        
        # Mock audio transcription
        with patch.object(video_processor, '_extract_and_transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = "Test transcript"
            
            # Process video
            result = await video_processor.process_video(video_id)
            
            # Verify result
            assert result.video_id == video_id
            assert result.status == ProcessingStatus.COMPLETED
            assert result.frames_extracted > 0
            assert result.transcript == "Test transcript"
            
            # Verify storage calls
            assert mock_storage.update_video_status.call_count >= 2
            mock_storage.store_transcript.assert_called_once_with(
                video_id, "Test transcript"
            )
            mock_storage.store_frame_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_video_error_handling(self, video_processor, mock_storage, sample_video_metadata):
        """Test error handling during video processing."""
        video_id = sample_video_metadata.video_id
        
        # Setup mocks
        mock_storage.get_video_metadata.return_value = sample_video_metadata
        mock_storage.update_video_status = Mock()
        
        # Don't create video file to trigger error
        
        # Process video - should raise exception
        with pytest.raises(FileNotFoundError):
            await video_processor.process_video(video_id)
        
        # Verify error status was set
        # Check that the failed status was called
        calls = mock_storage.update_video_status.call_args_list
        failed_call_found = False
        for call in calls:
            if len(call[0]) >= 2 and call[0][1] == ProcessingStatus.FAILED:
                failed_call_found = True
                break
        assert failed_call_found, "ProcessingStatus.FAILED was not set"