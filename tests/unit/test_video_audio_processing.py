"""Comprehensive unit tests for video and audio processing functionality."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import os
import json
from datetime import datetime

from src.processors.video import VideoProcessor
from src.storage.manager import StorageManager
from src.storage.schemas import ProcessingStatus, ProcessingResult, VideoMetadata, FrameAnalysis
from src.llm.ollama_client import OllamaClient


class TestVideoProcessing:
    """Test video processing functionality."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage manager."""
        storage = Mock(spec=StorageManager)
        storage.get_video_metadata = AsyncMock()
        storage.update_video_status = AsyncMock()
        storage.save_processing_result = AsyncMock()
        storage.get_storage_path = Mock()
        return storage
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock(spec=OllamaClient)
        client.analyze_frame = AsyncMock(return_value="A test frame showing objects")
        client.chat = AsyncMock(return_value="This is a test response")
        return client
    
    @pytest.fixture
    def video_processor(self, mock_storage, mock_llm_client):
        """Create a video processor with mocked dependencies."""
        return VideoProcessor(mock_storage, mock_llm_client)
    
    @pytest.fixture
    def sample_video_metadata(self):
        """Create sample video metadata."""
        return VideoMetadata(
            video_id="test_video_123",
            original_path="/path/to/video.mp4",
            filename="video.mp4",
            location="test_location",
            recording_timestamp=datetime.now(),
            duration=120.5,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264",
            size_bytes=1024000
        )
    
    @pytest.mark.asyncio
    async def test_process_video_success(self, video_processor, mock_storage, mock_llm_client, sample_video_metadata):
        """Test successful video processing."""
        # Setup mocks
        mock_storage.get_video_metadata.return_value = sample_video_metadata
        mock_storage.get_storage_path.return_value = Path("/tmp/test_storage")
        
        # Mock video capture
        with patch('cv2.VideoCapture') as mock_cap:
            mock_capture = MagicMock()
            mock_cap.return_value = mock_capture
            mock_capture.isOpened.return_value = True
            mock_capture.get.side_effect = lambda prop: {
                5: 30.0,  # FPS
                7: 3600   # Frame count
            }.get(prop, 0)
            mock_capture.read.side_effect = [(True, Mock()) for _ in range(10)] + [(False, None)]
            
            # Mock frame saving
            with patch('cv2.imwrite', return_value=True):
                # Mock audio extraction
                with patch('subprocess.run') as mock_subprocess:
                    mock_subprocess.return_value = Mock(returncode=0)
                    
                    # Mock whisper
                    with patch('whisper.load_model') as mock_whisper_load:
                        mock_model = Mock()
                        mock_whisper_load.return_value = mock_model
                        mock_model.transcribe.return_value = {
                            'text': 'This is a test transcription'
                        }
                        
                        # Process video
                        result = await video_processor.process_video("test_video_123")
                        
                        # Assertions
                        assert result.video_id == "test_video_123"
                        assert result.status == ProcessingStatus.COMPLETED
                        assert result.frames_extracted > 0
                        assert result.frames_analyzed > 0
                        assert result.transcript == 'This is a test transcription'
                        assert result.processing_time > 0
                        
                        # Verify mocks were called
                        mock_storage.get_video_metadata.assert_called_once_with("test_video_123")
                        mock_storage.update_video_status.assert_called()
                        mock_storage.save_processing_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_video_with_scene_detection(self, video_processor, mock_storage, sample_video_metadata):
        """Test video processing with scene detection enabled."""
        mock_storage.get_video_metadata.return_value = sample_video_metadata
        mock_storage.get_storage_path.return_value = Path("/tmp/test_storage")
        
        # Enable scene detection in config
        video_processor.config.scene_detection_enabled = True
        video_processor.config.scene_detection_threshold = 30.0
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_capture = MagicMock()
            mock_cap.return_value = mock_capture
            mock_capture.isOpened.return_value = True
            mock_capture.get.side_effect = lambda prop: {
                5: 30.0,  # FPS
                7: 300    # Frame count
            }.get(prop, 0)
            
            # Simulate scene changes
            frames = []
            for i in range(10):
                frame = Mock()
                # Create different frame data to simulate scene changes
                frame.__array_interface__ = {'shape': (1080, 1920, 3)}
                frames.append((True, frame))
            frames.append((False, None))
            mock_capture.read.side_effect = frames
            
            with patch('cv2.imwrite', return_value=True):
                with patch('subprocess.run') as mock_subprocess:
                    mock_subprocess.return_value = Mock(returncode=0)
                    
                    with patch('whisper.load_model') as mock_whisper:
                        mock_model = Mock()
                        mock_whisper.return_value = mock_model
                        mock_model.transcribe.return_value = {'text': 'Test'}
                        
                        result = await video_processor.process_video("test_video_123")
                        
                        assert result.status == ProcessingStatus.COMPLETED
                        assert result.frames_extracted > 0
    
    @pytest.mark.asyncio
    async def test_process_video_extraction_failure(self, video_processor, mock_storage, sample_video_metadata):
        """Test video processing when frame extraction fails."""
        mock_storage.get_video_metadata.return_value = sample_video_metadata
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_capture = MagicMock()
            mock_cap.return_value = mock_capture
            mock_capture.isOpened.return_value = False
            
            result = await video_processor.process_video("test_video_123")
            
            assert result.status == ProcessingStatus.FAILED
            assert "Could not open video file" in result.error_message
    
    @pytest.mark.asyncio
    async def test_process_video_partial_analysis(self, video_processor, mock_storage, mock_llm_client, sample_video_metadata):
        """Test video processing with partial frame analysis failure."""
        mock_storage.get_video_metadata.return_value = sample_video_metadata
        mock_storage.get_storage_path.return_value = Path("/tmp/test_storage")
        
        # Mock LLM to fail on some frames
        mock_llm_client.analyze_frame.side_effect = [
            "Frame 1 description",
            Exception("LLM error"),
            "Frame 3 description"
        ]
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_capture = MagicMock()
            mock_cap.return_value = mock_capture
            mock_capture.isOpened.return_value = True
            mock_capture.get.side_effect = lambda prop: {
                5: 30.0,
                7: 90
            }.get(prop, 0)
            mock_capture.read.side_effect = [(True, Mock()) for _ in range(3)] + [(False, None)]
            
            with patch('cv2.imwrite', return_value=True):
                with patch('subprocess.run') as mock_subprocess:
                    mock_subprocess.return_value = Mock(returncode=0)
                    
                    with patch('whisper.load_model') as mock_whisper:
                        mock_model = Mock()
                        mock_whisper.return_value = mock_model
                        mock_model.transcribe.return_value = {'text': 'Test'}
                        
                        result = await video_processor.process_video("test_video_123")
                        
                        assert result.status == ProcessingStatus.PARTIAL
                        assert result.frames_analyzed < result.frames_extracted
                        assert len(result.warnings) > 0


class TestAudioProcessing:
    """Test audio processing functionality."""
    
    @pytest.fixture
    def video_processor(self):
        """Create a video processor for testing audio."""
        storage = Mock(spec=StorageManager)
        llm_client = Mock(spec=OllamaClient)
        return VideoProcessor(storage, llm_client)
    
    @pytest.mark.asyncio
    async def test_extract_audio_success(self, video_processor, tmp_path):
        """Test successful audio extraction."""
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            audio_path = await video_processor._extract_audio(str(video_path))
            
            assert audio_path.endswith('.wav')
            mock_run.assert_called_once()
            
            # Check ffmpeg command
            cmd = mock_run.call_args[0][0]
            assert 'ffmpeg' in cmd
            assert '-i' in cmd
            assert str(video_path) in cmd
            assert '-acodec' in cmd
            assert 'pcm_s16le' in cmd
    
    @pytest.mark.asyncio
    async def test_extract_audio_failure(self, video_processor, tmp_path):
        """Test audio extraction failure."""
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="FFmpeg error")
            
            audio_path = await video_processor._extract_audio(str(video_path))
            
            assert audio_path is None
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, video_processor, tmp_path):
        """Test successful audio transcription."""
        audio_path = tmp_path / "test_audio.wav"
        audio_path.touch()
        
        with patch('whisper.load_model') as mock_whisper:
            mock_model = Mock()
            mock_whisper.return_value = mock_model
            mock_model.transcribe.return_value = {
                'text': 'This is a test transcription with multiple words'
            }
            
            transcript = await video_processor._transcribe_audio(str(audio_path))
            
            assert transcript == 'This is a test transcription with multiple words'
            mock_whisper.assert_called_once_with("base")
            mock_model.transcribe.assert_called_once_with(str(audio_path))
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_empty(self, video_processor, tmp_path):
        """Test transcription with empty result."""
        audio_path = tmp_path / "test_audio.wav"
        audio_path.touch()
        
        with patch('whisper.load_model') as mock_whisper:
            mock_model = Mock()
            mock_whisper.return_value = mock_model
            mock_model.transcribe.return_value = {'text': ''}
            
            transcript = await video_processor._transcribe_audio(str(audio_path))
            
            assert transcript is None
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_exception(self, video_processor, tmp_path):
        """Test transcription with exception."""
        audio_path = tmp_path / "test_audio.wav"
        audio_path.touch()
        
        with patch('whisper.load_model') as mock_whisper:
            mock_whisper.side_effect = Exception("Model loading failed")
            
            transcript = await video_processor._transcribe_audio(str(audio_path))
            
            assert transcript is None
    
    def test_cleanup_audio_files(self, video_processor, tmp_path):
        """Test cleanup of temporary audio files."""
        # Create temporary audio file
        audio_file = tmp_path / "temp_audio.wav"
        audio_file.touch()
        
        video_processor._temp_audio_files = [str(audio_file)]
        
        # Verify file exists
        assert audio_file.exists()
        
        # Clean up
        video_processor.cleanup()
        
        # Verify file is removed
        assert not audio_file.exists()
        assert len(video_processor._temp_audio_files) == 0


class TestIntegrationVideoAudio:
    """Integration tests for video and audio processing together."""
    
    @pytest.mark.asyncio
    async def test_full_video_processing_with_audio(self, tmp_path):
        """Test complete video processing including audio transcription."""
        # Create test video file
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        # Setup storage
        storage = Mock(spec=StorageManager)
        storage.get_video_metadata = AsyncMock(return_value=VideoMetadata(
            video_id="test_123",
            original_path=str(video_file),
            filename="test_video.mp4",
            location="test",
            recording_timestamp=datetime.now(),
            duration=10.0,
            fps=30.0,
            width=640,
            height=480,
            codec="h264",
            size_bytes=1024
        ))
        storage.get_storage_path = Mock(return_value=tmp_path)
        storage.update_video_status = AsyncMock()
        storage.save_processing_result = AsyncMock()
        
        # Setup LLM client
        llm_client = Mock(spec=OllamaClient)
        llm_client.analyze_frame = AsyncMock(return_value="Test frame description")
        
        # Create processor
        processor = VideoProcessor(storage, llm_client)
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_capture = MagicMock()
            mock_cap.return_value = mock_capture
            mock_capture.isOpened.return_value = True
            mock_capture.get.side_effect = lambda prop: {
                5: 30.0,  # FPS
                7: 150    # Frame count (5 seconds at 30fps)
            }.get(prop, 0)
            mock_capture.read.side_effect = [(True, Mock()) for _ in range(5)] + [(False, None)]
            
            with patch('cv2.imwrite', return_value=True):
                with patch('subprocess.run') as mock_ffmpeg:
                    mock_ffmpeg.return_value = Mock(returncode=0)
                    
                    with patch('whisper.load_model') as mock_whisper:
                        mock_model = Mock()
                        mock_whisper.return_value = mock_model
                        mock_model.transcribe.return_value = {
                            'text': 'This is a comprehensive test of video and audio processing'
                        }
                        
                        # Process video
                        result = await processor.process_video("test_123")
                        
                        # Verify results
                        assert result.status == ProcessingStatus.COMPLETED
                        assert result.frames_extracted == 5
                        assert result.frames_analyzed == 5
                        assert result.transcript == 'This is a comprehensive test of video and audio processing'
                        assert len(result.timeline) == 5
                        
                        # Verify audio was extracted
                        mock_ffmpeg.assert_called_once()
                        ffmpeg_cmd = mock_ffmpeg.call_args[0][0]
                        assert 'ffmpeg' in ffmpeg_cmd
                        assert '-i' in ffmpeg_cmd
                        assert '.wav' in ffmpeg_cmd[-1]
                        
                        # Verify whisper was called
                        mock_whisper.assert_called_once_with("base")
                        mock_model.transcribe.assert_called_once()
                        
                        # Clean up
                        processor.cleanup()


class TestVideoProcessorConfiguration:
    """Test video processor configuration handling."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        storage = Mock(spec=StorageManager)
        llm_client = Mock(spec=OllamaClient)
        
        processor = VideoProcessor(storage, llm_client)
        
        assert processor.config.target_fps == 1.0
        assert processor.config.max_frames == 120
        assert processor.config.scene_detection_enabled == True
        assert processor.config.scene_detection_threshold == 30.0
        assert processor.config.audio_enabled == True
        assert processor.config.whisper_model == "base"
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        storage = Mock(spec=StorageManager)
        llm_client = Mock(spec=OllamaClient)
        
        custom_config = {
            "target_fps": 2.0,
            "max_frames": 60,
            "scene_detection_enabled": False,
            "audio_enabled": False
        }
        
        with patch('src.processors.video.load_config') as mock_load_config:
            mock_config = Mock()
            mock_config.processing = Mock(**custom_config)
            mock_load_config.return_value = mock_config
            
            processor = VideoProcessor(storage, llm_client)
            
            assert processor.config.target_fps == 2.0
            assert processor.config.max_frames == 60
            assert processor.config.scene_detection_enabled == False
            assert processor.config.audio_enabled == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])