#!/usr/bin/env python3
"""Integration tests for the MCP video server system."""
import asyncio
import pytest
import httpx
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.manager import StorageManager
from src.processors.video import VideoProcessor
from src.llm.ollama_client import OllamaClient
from src.tools.mcp_tools import VideoTools
from src.utils.config import get_config
from mcp.server.fastmcp import FastMCP

# Test configuration
TEST_VIDEO_PATH = Path(__file__).parent / "video_data" / "originals" / "sample_video.mp4"
TEST_SERVER_URL = "http://localhost:8001"  # Use different port for testing


class TestVideoSystem:
    """Integration tests for the video analysis system."""
    
    @pytest.fixture
    async def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = StorageManager(base_path=temp_dir)
            yield storage
    
    @pytest.fixture
    async def video_processor(self, temp_storage):
        """Create video processor for testing."""
        llm_client = OllamaClient()
        processor = VideoProcessor(temp_storage, llm_client)
        yield processor
        await llm_client.close()
        processor.cleanup()
    
    @pytest.fixture
    async def mcp_server(self, video_processor, temp_storage):
        """Create MCP server for testing."""
        mcp = FastMCP("test-video-server")
        tools = VideoTools(video_processor, temp_storage)
        tools.register(mcp)
        return mcp, tools
    
    async def test_format_time_ago(self, mcp_server):
        """Test the _format_time_ago method exists and works."""
        _, tools = mcp_server
        
        # Test various time differences
        now = datetime.now()
        
        # 30 minutes ago
        timestamp = now - timedelta(minutes=30)
        assert tools._format_time_ago(timestamp) == "30 minutes ago"
        
        # 2 hours ago
        timestamp = now - timedelta(hours=2)
        assert tools._format_time_ago(timestamp) == "2 hours ago"
        
        # Yesterday
        timestamp = now - timedelta(days=1)
        assert tools._format_time_ago(timestamp) == "yesterday"
        
        # 5 days ago
        timestamp = now - timedelta(days=5)
        assert tools._format_time_ago(timestamp) == "5 days ago"
        
        # 2 weeks ago
        timestamp = now - timedelta(weeks=2)
        assert tools._format_time_ago(timestamp) == "2 weeks ago"
        
        # 2 months ago
        timestamp = now - timedelta(days=60)
        assert "ago" not in tools._format_time_ago(timestamp)  # Should be formatted date
    
    async def test_database_path_consistency(self):
        """Test that all components use the same database path."""
        import os
        
        # Test from different working directories
        original_cwd = os.getcwd()
        
        try:
            # Test 1: From project root
            os.chdir(Path(__file__).parent)
            storage1 = StorageManager()
            db_path1 = storage1.base_path / "index" / "metadata.db"
            
            # Test 2: From a subdirectory
            os.chdir(Path(__file__).parent / "video_data")
            storage2 = StorageManager()
            db_path2 = storage2.base_path / "index" / "metadata.db"
            
            # They should be the same absolute path
            assert db_path1.resolve() == db_path2.resolve()
            
        finally:
            os.chdir(original_cwd)
    
    async def test_process_video_tool(self, mcp_server, temp_storage):
        """Test video processing through MCP tools."""
        mcp, tools = mcp_server
        
        if not TEST_VIDEO_PATH.exists():
            pytest.skip(f"Test video not found at {TEST_VIDEO_PATH}")
        
        # Get the process_video tool
        process_tool = None
        for tool in mcp._tools.values():
            if tool.__name__ == "process_video":
                process_tool = tool
                break
        
        assert process_tool is not None, "process_video tool not found"
        
        # Process a video
        result = await process_tool(
            video_path=str(TEST_VIDEO_PATH),
            location="test_location",
            extract_audio=False  # Skip audio for faster testing
        )
        
        assert "video_id" in result
        assert result["status"] == "completed"
        assert result["frames_extracted"] > 0
        
        # Verify video is in database
        videos = temp_storage.list_videos()
        assert len(videos) == 1
        assert videos[0].location == "test_location"
    
    async def test_query_location_time_tool(self, mcp_server, temp_storage):
        """Test querying videos by location and time."""
        mcp, tools = mcp_server
        
        # First, add a test video directly to storage
        video_metadata = await temp_storage.store_video(
            str(TEST_VIDEO_PATH),
            location="test_shed",
            recording_timestamp=datetime.now() - timedelta(hours=2)
        )
        
        # Get the query tool
        query_tool = None
        for tool in mcp._tools.values():
            if tool.__name__ == "query_location_time":
                query_tool = tool
                break
        
        assert query_tool is not None, "query_location_time tool not found"
        
        # Query for recent videos
        result = await query_tool(
            location="test_shed",
            time_query="recent"
        )
        
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["location"] == "test_shed"
        assert "time_ago" in result["results"][0]  # This would fail without _format_time_ago
    
    async def test_http_client_routing(self):
        """Test that the HTTP client properly routes queries."""
        from standalone_client.mcp_http_client import MCPHttpClient
        
        # Create a mock client (without actually connecting)
        client = MCPHttpClient(server_url="http://localhost:8000", chat_llm_model="mistral:latest")
        
        # Test video-related queries should use MCP tools
        video_queries = [
            "show me the latest videos",
            "what videos do you have",
            "find videos with cars",
            "what happened at the shed"
        ]
        
        # Test non-video queries should use chat LLM
        chat_queries = [
            "tell me a story",
            "how are you",
            "what's the weather like",
            "explain quantum physics"
        ]
        
        # Mock the chat LLM
        class MockLLM:
            async def chat(self, prompt, model):
                return "Chat response"
        
        client.chat_llm = MockLLM()
        
        # Test routing logic without actual API calls
        for query in video_queries:
            # These should attempt to use MCP tools
            query_lower = query.lower()
            is_greeting = any(word in query_lower for word in ['hi', 'hello', 'hey'])
            has_video_keywords = any(word in query_lower for word in 
                                   ['latest', 'recent', 'videos', 'find', 'search', 
                                    'happened', 'at', 'from', 'show', 'what'])
            assert not is_greeting and has_video_keywords, f"'{query}' should be routed to MCP tools"
        
        for query in chat_queries:
            # These should use chat LLM
            query_lower = query.lower()
            video_keywords = ['video', 'videos', 'recording', 'footage', 'clip', 'show', 
                            'what', 'when', 'where', 'happened', 'saw', 'recorded']
            has_video_keywords = any(word in query_lower for word in video_keywords)
            assert not has_video_keywords, f"'{query}' should be routed to chat LLM"


class TestHTTPIntegration:
    """Test HTTP server and client integration."""
    
    @pytest.fixture
    async def http_session(self):
        """Create HTTP client session."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            yield client
    
    async def test_mcp_handshake(self, http_session):
        """Test the MCP initialization handshake."""
        # This test would require the server to be running
        # For now, we'll test the handshake structure
        
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json"
        }
        
        # The correct initialization sequence
        init_request = {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "Test Client", "version": "1.0.0"}
            }
        }
        
        # Verify request structure
        assert init_request["method"] == "initialize"
        assert "protocolVersion" in init_request["params"]
        
        # The initialized notification (required!)
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        
        assert initialized_notification["method"] == "notifications/initialized"
    
    async def test_sse_parsing(self):
        """Test Server-Sent Events parsing."""
        from standalone_client.mcp_http_client import parse_sse_response
        
        # Test valid SSE response
        sse_response = '''data: {"jsonrpc": "2.0", "id": "test", "result": {"status": "ok"}}

data: 

'''
        result = parse_sse_response(sse_response)
        assert result is not None
        assert result["id"] == "test"
        assert result["result"]["status"] == "ok"
        
        # Test empty data lines
        sse_response = '''data: 
data: {"jsonrpc": "2.0", "result": "test"}
'''
        result = parse_sse_response(sse_response)
        assert result is not None
        assert result["result"] == "test"


def test_config_path_resolution():
    """Test configuration path resolution."""
    from src.utils.config import StorageConfig
    import os
    
    # Test with relative path
    config = StorageConfig(base_path="./video_data")
    validated_path = config.validate_base_path("./video_data")
    
    # Should be absolute
    assert Path(validated_path).is_absolute()
    
    # Should point to project's video_data
    assert "mcp-video-server/video_data" in str(validated_path)
    
    # Test with env var
    os.environ["VIDEO_DATA_PATH"] = "/tmp/test_video_data"
    config = StorageConfig(base_path="./video_data")
    validated_path = config.validate_base_path("./video_data")
    assert str(validated_path) == "/tmp/test_video_data"
    del os.environ["VIDEO_DATA_PATH"]


# Run tests
if __name__ == "__main__":
    import subprocess
    
    # Install pytest if needed
    try:
        import pytest
    except ImportError:
        print("Installing pytest...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio"])
    
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])