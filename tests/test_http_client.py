#!/usr/bin/env python3
"""Unit tests for HTTP client query routing."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from standalone_client.mcp_http_client import MCPHttpClient, parse_sse_response


class TestQueryRouting:
    """Test query routing in the HTTP client."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock HTTP client."""
        client = MCPHttpClient(server_url="http://localhost:8000", chat_llm_model="mistral:latest")
        
        # Mock the HTTP session
        client.session = Mock()
        client.mcp_session_id = "test_session"
        client.available_tools = {
            "query_location_time": {"description": "Query by location/time"},
            "search_videos": {"description": "Search videos"},
            "get_video_summary": {"description": "Get summary"}
        }
        
        # Mock the LLM
        client.chat_llm = Mock()
        client.chat_llm.chat = AsyncMock(return_value="Chat response")
        
        # Mock call_tool
        client.call_tool = AsyncMock(return_value={"results": []})
        
        return client
    
    @pytest.mark.asyncio
    async def test_greeting_uses_chat_llm(self, mock_client):
        """Test that greetings use the chat LLM."""
        greetings = ["hi", "hello", "hey", "good morning", "how are you"]
        
        for greeting in greetings:
            response = await mock_client.process_query(greeting)
            
            # Should have called chat LLM
            mock_client.chat_llm.chat.assert_called()
            assert response == "Chat response"
            
            # Should NOT have called MCP tools
            mock_client.call_tool.assert_not_called()
            
            # Reset mocks
            mock_client.chat_llm.chat.reset_mock()
            mock_client.call_tool.reset_mock()
    
    @pytest.mark.asyncio
    async def test_video_queries_use_mcp_tools(self, mock_client):
        """Test that video queries use MCP tools."""
        video_queries = [
            "show me the latest videos",
            "what videos do you have",
            "find videos with cars",
            "summarize video vid_123"
        ]
        
        for query in video_queries:
            response = await mock_client.process_query(query)
            
            # Should have called MCP tools
            mock_client.call_tool.assert_called()
            
            # Should NOT have called chat LLM (except for formatting)
            # Reset mocks
            mock_client.call_tool.reset_mock()
    
    @pytest.mark.asyncio
    async def test_general_queries_use_chat_llm(self, mock_client):
        """Test that general queries use chat LLM."""
        general_queries = [
            "tell me a story",
            "explain quantum physics",
            "write a poem",
            "how's the weather"
        ]
        
        for query in general_queries:
            response = await mock_client.process_query(query)
            
            # Should have called chat LLM
            mock_client.chat_llm.chat.assert_called()
            assert response == "Chat response"
            
            # Should NOT have called MCP tools
            mock_client.call_tool.assert_not_called()
            
            # Reset mocks
            mock_client.chat_llm.chat.reset_mock()
            mock_client.call_tool.reset_mock()
    
    @pytest.mark.asyncio
    async def test_ambiguous_queries(self, mock_client):
        """Test queries that could be interpreted either way."""
        # "what" is in video keywords, but context matters
        query = "what is the meaning of life"
        response = await mock_client.process_query(query)
        
        # Should use chat LLM because no other video indicators
        mock_client.chat_llm.chat.assert_called()
        
        # Reset
        mock_client.chat_llm.chat.reset_mock()
        mock_client.call_tool.reset_mock()
        
        # But "what happened" should use MCP
        query = "what happened yesterday"
        response = await mock_client.process_query(query)
        
        # Should use MCP tools
        mock_client.call_tool.assert_called()
    
    @pytest.mark.asyncio
    async def test_chat_llm_error_handling(self, mock_client):
        """Test handling of chat LLM errors."""
        # Make chat LLM raise an error
        mock_client.chat_llm.chat = AsyncMock(side_effect=Exception("LLM error"))
        
        response = await mock_client.process_query("tell me a story")
        
        # Should return error message
        assert "error" in response.lower()
        assert "chat system" in response.lower()


class TestSSEParsing:
    """Test Server-Sent Events parsing."""
    
    def test_parse_valid_sse(self):
        """Test parsing valid SSE responses."""
        sse_response = 'data: {"jsonrpc": "2.0", "id": "test", "result": {"status": "ok"}}\n\n'
        result = parse_sse_response(sse_response)
        
        assert result is not None
        assert result["id"] == "test"
        assert result["result"]["status"] == "ok"
    
    def test_parse_empty_data_lines(self):
        """Test parsing SSE with empty data lines."""
        sse_response = '''data: 
data: {"jsonrpc": "2.0", "result": "test"}
data: 
'''
        result = parse_sse_response(sse_response)
        
        assert result is not None
        assert result["result"] == "test"
    
    def test_parse_invalid_json(self):
        """Test parsing SSE with invalid JSON."""
        sse_response = 'data: {invalid json}\n\n'
        result = parse_sse_response(sse_response)
        
        assert result is None
    
    def test_parse_no_data_prefix(self):
        """Test parsing response without data: prefix."""
        sse_response = '{"jsonrpc": "2.0", "result": "test"}\n'
        result = parse_sse_response(sse_response)
        
        assert result is None


class TestToolCalling:
    """Test MCP tool calling."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock HTTP client."""
        client = MCPHttpClient()
        
        # Mock the HTTP session
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json = Mock(return_value={
            "jsonrpc": "2.0",
            "id": "test",
            "result": {
                "content": [{"text": '{"video_id": "vid_123", "status": "ok"}'}]
            }
        })
        
        client.session = Mock()
        client.session.post = AsyncMock(return_value=mock_response)
        client.mcp_session_id = "test_session"
        
        return client
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_client):
        """Test successful tool call."""
        result = await mock_client.call_tool("test_tool", {"param": "value"})
        
        # Should have made POST request
        mock_client.session.post.assert_called_once()
        
        # Check request structure
        call_args = mock_client.session.post.call_args
        request_data = call_args[1]["json"]
        
        assert request_data["method"] == "tools/call"
        assert request_data["params"]["name"] == "test_tool"
        assert request_data["params"]["arguments"] == {"param": "value"}
        
        # Check result parsing
        assert result == '{"video_id": "vid_123", "status": "ok"}'
    
    @pytest.mark.asyncio
    async def test_call_tool_with_sse_response(self, mock_client):
        """Test tool call with SSE response."""
        # Mock SSE response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/event-stream'}
        mock_response.text = 'data: {"jsonrpc": "2.0", "result": {"content": [{"text": "SSE result"}]}}\n\n'
        
        mock_client.session.post = AsyncMock(return_value=mock_response)
        
        result = await mock_client.call_tool("test_tool", {})
        
        assert result == "SSE result"
    
    @pytest.mark.asyncio  
    async def test_call_tool_error_response(self, mock_client):
        """Test tool call with error response."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json = Mock(return_value={
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": "Tool error"}
        })
        
        mock_client.session.post = AsyncMock(return_value=mock_response)
        
        result = await mock_client.call_tool("test_tool", {})
        
        assert result is None  # Error returns None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])