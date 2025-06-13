"""Tests for the process_new_video.py script."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from process_new_video import process_video
from src.storage.schemas import VideoMetadata, ProcessingResult, ProcessingStatus


class TestProcessNewVideo:
    """Test the process_new_video script functionality."""
    
    @pytest.mark.asyncio
    async def test_process_video_success(self, tmp_path):
        """Test successful video processing through the script."""
        # Create a test video file
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        # Mock the components
        with patch('process_new_video.StorageManager') as MockStorage:
            with patch('process_new_video.OllamaClient') as MockLLM:
                with patch('process_new_video.VideoProcessor') as MockProcessor:
                    # Setup storage mock
                    mock_storage = MockStorage.return_value
                    mock_storage.store_video = AsyncMock(return_value=VideoMetadata(
                        video_id="test_vid_123",
                        original_path=str(video_file),
                        filename="test_video.mp4",
                        location="test_location",
                        recording_timestamp="2024-01-01T00:00:00",
                        duration=60.0,
                        fps=30.0,
                        width=1920,
                        height=1080,
                        codec="h264",
                        size_bytes=1024000
                    ))
                    
                    # Setup LLM mock
                    mock_llm = MockLLM.return_value
                    mock_llm.close = AsyncMock()
                    
                    # Setup processor mock
                    mock_processor = MockProcessor.return_value
                    mock_processor.process_video = AsyncMock(return_value=ProcessingResult(
                        video_id="test_vid_123",
                        status=ProcessingStatus.COMPLETED,
                        frames_extracted=10,
                        frames_analyzed=10,
                        transcript="This is a test transcript from the video",
                        timeline=[],
                        processing_time=5.5,
                        error_message=None,
                        warnings=[]
                    ))
                    mock_processor.cleanup = Mock()
                    
                    # Run the process
                    video_id = await process_video(str(video_file), "test_location")
                    
                    # Assertions
                    assert video_id == "test_vid_123"
                    
                    # Verify calls
                    mock_storage.store_video.assert_called_once_with(str(video_file), location="test_location")
                    mock_processor.process_video.assert_called_once_with("test_vid_123")
                    mock_llm.close.assert_called_once()
                    mock_processor.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_video_with_transcript(self, tmp_path, capsys):
        """Test video processing with transcript output."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        with patch('process_new_video.StorageManager') as MockStorage:
            with patch('process_new_video.OllamaClient') as MockLLM:
                with patch('process_new_video.VideoProcessor') as MockProcessor:
                    # Setup mocks
                    mock_storage = MockStorage.return_value
                    mock_storage.store_video = AsyncMock(return_value=VideoMetadata(
                        video_id="test_vid_456",
                        original_path=str(video_file),
                        filename="test_video.mp4",
                        location="test",
                        recording_timestamp="2024-01-01T00:00:00",
                        duration=30.0,
                        fps=30.0,
                        width=1280,
                        height=720,
                        codec="h264",
                        size_bytes=512000
                    ))
                    
                    mock_llm = MockLLM.return_value
                    mock_llm.close = AsyncMock()
                    
                    # Create result with transcript
                    mock_processor = MockProcessor.return_value
                    mock_processor.process_video = AsyncMock(return_value=ProcessingResult(
                        video_id="test_vid_456",
                        status=ProcessingStatus.COMPLETED,
                        frames_extracted=5,
                        frames_analyzed=5,
                        transcript="This is a longer transcript that should be truncated in the output. " * 10,
                        timeline=[],
                        processing_time=3.2,
                        error_message=None,
                        warnings=[]
                    ))
                    mock_processor.cleanup = Mock()
                    
                    # Run process
                    await process_video(str(video_file), "test")
                    
                    # Check output
                    captured = capsys.readouterr()
                    assert "Processing video:" in captured.out
                    assert "Stored video with ID: test_vid_456" in captured.out
                    assert "Processing complete!" in captured.out
                    assert "Status: completed" in captured.out
                    assert "Frames extracted: 5" in captured.out
                    assert "Frames analyzed: 5" in captured.out
                    assert "Processing time: 3.20s" in captured.out
                    assert "Transcript preview:" in captured.out
                    assert "..." in captured.out  # Truncation indicator
    
    @pytest.mark.asyncio
    async def test_process_video_no_transcript(self, tmp_path, capsys):
        """Test video processing without transcript."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        with patch('process_new_video.StorageManager') as MockStorage:
            with patch('process_new_video.OllamaClient') as MockLLM:
                with patch('process_new_video.VideoProcessor') as MockProcessor:
                    # Setup mocks
                    mock_storage = MockStorage.return_value
                    mock_storage.store_video = AsyncMock(return_value=VideoMetadata(
                        video_id="test_vid_789",
                        original_path=str(video_file),
                        filename="test_video.mp4",
                        location="test",
                        recording_timestamp="2024-01-01T00:00:00",
                        duration=15.0,
                        fps=24.0,
                        width=640,
                        height=480,
                        codec="h264",
                        size_bytes=256000
                    ))
                    
                    mock_llm = MockLLM.return_value
                    mock_llm.close = AsyncMock()
                    
                    # Result without transcript
                    mock_processor = MockProcessor.return_value
                    mock_processor.process_video = AsyncMock(return_value=ProcessingResult(
                        video_id="test_vid_789",
                        status=ProcessingStatus.COMPLETED,
                        frames_extracted=3,
                        frames_analyzed=3,
                        transcript=None,  # No transcript
                        timeline=[],
                        processing_time=2.1,
                        error_message=None,
                        warnings=[]
                    ))
                    mock_processor.cleanup = Mock()
                    
                    # Run process
                    await process_video(str(video_file), "test")
                    
                    # Check output
                    captured = capsys.readouterr()
                    assert "Transcript preview:" not in captured.out
    
    @pytest.mark.asyncio
    async def test_process_video_cleanup_on_error(self, tmp_path):
        """Test that cleanup happens even on error."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        with patch('process_new_video.StorageManager') as MockStorage:
            with patch('process_new_video.OllamaClient') as MockLLM:
                with patch('process_new_video.VideoProcessor') as MockProcessor:
                    # Setup storage to raise error
                    mock_storage = MockStorage.return_value
                    mock_storage.store_video = AsyncMock(side_effect=Exception("Storage error"))
                    
                    mock_llm = MockLLM.return_value
                    mock_llm.close = AsyncMock()
                    
                    mock_processor = MockProcessor.return_value
                    mock_processor.cleanup = Mock()
                    
                    # Run process and expect error
                    with pytest.raises(Exception, match="Storage error"):
                        await process_video(str(video_file), "test")
                    
                    # Verify cleanup was called
                    mock_llm.close.assert_called_once()
                    mock_processor.cleanup.assert_called_once()
    
    def test_main_script_file_not_found(self):
        """Test main script with non-existent file."""
        from process_new_video import main
        
        # Mock sys.argv with a file path that doesn't exist
        test_args = ["process_new_video.py", "/path/to/nonexistent/video.mp4"]
        
        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                # Mock exit to prevent actual exit
                mock_exit.side_effect = SystemExit(1)
                
                with patch('builtins.print') as mock_print:
                    # Expect SystemExit to be raised
                    with pytest.raises(SystemExit):
                        main()
                    
                    mock_exit.assert_called_once_with(1)
                    # Verify error message was printed
                    calls = [str(call) for call in mock_print.call_args_list]
                    error_msg_found = any("Error: Video file not found:" in call for call in calls)
                    assert error_msg_found
    
    def test_main_script_success(self, tmp_path):
        """Test main script with valid file."""
        from process_new_video import main
        
        # Create test file
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        # Mock sys.argv
        test_args = ["process_new_video.py", str(video_file), "--location", "test_location"]
        
        with patch('sys.argv', test_args):
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = "test_vid_success"
                
                # Capture print output
                with patch('builtins.print') as mock_print:
                    main()
                    
                    # Verify asyncio.run was called
                    mock_run.assert_called_once()
                    
                    # Check final success message was printed
                    calls = [str(call) for call in mock_print.call_args_list]
                    success_msg_found = any("Video processed successfully! ID: test_vid_success" in call for call in calls)
                    assert success_msg_found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])