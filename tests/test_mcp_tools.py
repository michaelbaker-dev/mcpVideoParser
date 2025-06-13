#!/usr/bin/env python3
"""Unit tests for MCP tools."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.mcp_tools import VideoTools
from src.storage.schemas import VideoMetadata, ProcessingResult, ProcessingStatus


class TestVideoTools:
    """Test VideoTools class methods."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage manager."""
        storage = Mock()
        storage.query_videos_by_location_and_time = Mock(return_value=[])
        storage.get_processing_result = Mock(return_value=None)
        storage.store_video = AsyncMock()
        return storage
    
    @pytest.fixture
    def mock_processor(self):
        """Create mock video processor."""
        processor = Mock()
        processor.process_video = AsyncMock()
        processor.config.processing.frame_sample_rate = 30
        return processor
    
    @pytest.fixture
    def video_tools(self, mock_processor, mock_storage):
        """Create VideoTools instance with mocks."""
        return VideoTools(mock_processor, mock_storage)
    
    def test_format_time_ago_method_exists(self, video_tools):
        """Test that _format_time_ago method exists and is callable."""
        assert hasattr(video_tools, '_format_time_ago')
        assert callable(video_tools._format_time_ago)
    
    def test_format_time_ago_minutes(self, video_tools):
        """Test formatting for minutes ago."""
        now = datetime.now()
        
        # 0 minutes
        timestamp = now
        assert video_tools._format_time_ago(timestamp) == "0 minutes ago"
        
        # 30 minutes
        timestamp = now - timedelta(minutes=30)
        assert video_tools._format_time_ago(timestamp) == "30 minutes ago"
        
        # 59 minutes
        timestamp = now - timedelta(minutes=59)
        assert video_tools._format_time_ago(timestamp) == "59 minutes ago"
    
    def test_format_time_ago_hours(self, video_tools):
        """Test formatting for hours ago."""
        now = datetime.now()
        
        # 1 hour
        timestamp = now - timedelta(hours=1)
        assert video_tools._format_time_ago(timestamp) == "1 hours ago"
        
        # 5 hours
        timestamp = now - timedelta(hours=5)
        assert video_tools._format_time_ago(timestamp) == "5 hours ago"
        
        # 23 hours
        timestamp = now - timedelta(hours=23)
        assert video_tools._format_time_ago(timestamp) == "23 hours ago"
    
    def test_format_time_ago_days(self, video_tools):
        """Test formatting for days ago."""
        now = datetime.now()
        
        # Yesterday
        timestamp = now - timedelta(days=1)
        assert video_tools._format_time_ago(timestamp) == "yesterday"
        
        # 3 days
        timestamp = now - timedelta(days=3)
        assert video_tools._format_time_ago(timestamp) == "3 days ago"
        
        # 6 days
        timestamp = now - timedelta(days=6)
        assert video_tools._format_time_ago(timestamp) == "6 days ago"
    
    def test_format_time_ago_weeks(self, video_tools):
        """Test formatting for weeks ago."""
        now = datetime.now()
        
        # 1 week
        timestamp = now - timedelta(weeks=1)
        assert video_tools._format_time_ago(timestamp) == "1 weeks ago"
        
        # 3 weeks
        timestamp = now - timedelta(weeks=3)
        assert video_tools._format_time_ago(timestamp) == "3 weeks ago"
    
    def test_format_time_ago_months(self, video_tools):
        """Test formatting for months ago."""
        now = datetime.now()
        
        # 2 months ago - should show actual date
        timestamp = now - timedelta(days=60)
        result = video_tools._format_time_ago(timestamp)
        assert "ago" not in result  # Should be formatted date
        assert timestamp.strftime("%B") in result  # Should contain month name
    
    @pytest.mark.asyncio
    async def test_register_creates_tools(self, video_tools):
        """Test that register method creates all expected tools."""
        mock_mcp = Mock()
        mock_mcp.tool = Mock(return_value=lambda x: x)
        
        # Register tools
        video_tools.register(mock_mcp)
        
        # Should have called mcp.tool() multiple times
        assert mock_mcp.tool.call_count >= 6  # At least 6 tools should be registered
    
    @pytest.mark.asyncio
    async def test_query_with_format_time_ago(self, video_tools, mock_storage):
        """Test that query_location_time uses _format_time_ago correctly."""
        # Create mock video
        mock_video = Mock()
        mock_video.video_id = "vid_123"
        mock_video.location = "test"
        mock_video.recording_timestamp = datetime.now() - timedelta(hours=2)
        mock_video.duration = 10.5
        
        mock_storage.query_videos_by_location_and_time.return_value = [mock_video]
        
        # Create a mock MCP server
        mock_mcp = Mock()
        tools_registered = {}
        
        def register_tool(func):
            tools_registered[func.__name__] = func
            return func
        
        mock_mcp.tool = Mock(side_effect=lambda: register_tool)
        
        # Register tools
        video_tools.register(mock_mcp)
        
        # Get the query_location_time tool
        query_tool = tools_registered.get('query_location_time')
        assert query_tool is not None
        
        # Call the tool
        result = await query_tool(time_query="recent")
        
        # Check that the result contains properly formatted time
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["time_ago"] == "2 hours ago"


class TestMCPToolErrors:
    """Test error handling in MCP tools."""
    
    @pytest.mark.asyncio
    async def test_process_video_error_handling(self):
        """Test error handling in process_video tool."""
        mock_storage = Mock()
        mock_storage.store_video = AsyncMock(side_effect=Exception("Storage error"))
        
        mock_processor = Mock()
        tools = VideoTools(mock_processor, mock_storage)
        
        # Register tools
        mock_mcp = Mock()
        tools_registered = {}
        
        def register_tool(func):
            tools_registered[func.__name__] = func
            return func
        
        mock_mcp.tool = Mock(side_effect=lambda: register_tool)
        tools.register(mock_mcp)
        
        # Get process_video tool
        process_tool = tools_registered.get('process_video')
        
        # Should handle error gracefully
        result = await process_tool(video_path="/fake/path.mp4")
        assert "error" in result
        assert result["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_query_error_handling(self):
        """Test error handling in query_location_time tool."""
        mock_storage = Mock()
        mock_storage.query_videos_by_location_and_time = Mock(
            side_effect=Exception("Query error")
        )
        
        mock_processor = Mock()
        tools = VideoTools(mock_processor, mock_storage)
        
        # Register tools
        mock_mcp = Mock()
        tools_registered = {}
        
        def register_tool(func):
            tools_registered[func.__name__] = func
            return func
        
        mock_mcp.tool = Mock(side_effect=lambda: register_tool)
        tools.register(mock_mcp)
        
        # Get query tool
        query_tool = tools_registered.get('query_location_time')
        
        # Should handle error gracefully
        result = await query_tool(time_query="recent")
        assert "error" in result
        assert "Query error" in result["error"]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])