# MCP Streamable HTTP Protocol Documentation

## Overview

The Model Context Protocol (MCP) Streamable HTTP transport is a modern, efficient protocol for exposing MCP servers via HTTP. It simplifies deployment and enables sophisticated server-client interactions while maintaining backwards compatibility.

## Key Features

1. **Single Endpoint Communication**: All MCP interactions flow through one endpoint (`/mcp` by default)
2. **Bi-directional Communication**: Servers can send notifications and requests back to clients
3. **Automatic Connection Upgrades**: Connections start as standard HTTP requests but can upgrade to SSE (Server-Sent Events) for streaming
4. **Session Management**: Optional session IDs maintain state across multiple connections

## Protocol Details

### Request Format

**HTTP Method**: POST  
**Endpoint**: `/mcp` (configurable)  
**Headers**:
- `Content-Type: application/json`
- `Accept: application/json, text/event-stream`
- `Mcp-Session-Id: <session-id>` (required after initialization, except for stateless servers)

**Body**: JSON-RPC 2.0 format
- Single request/notification/response
- Batched requests/notifications
- Batched responses

### Response Format

**Status Codes**:
- `200 OK`: Successful response with JSON body
- `202 Accepted`: For notifications/responses (no response body expected)
- `400 Bad Request`: Missing session ID or invalid request
- `404 Not Found`: Session terminated or not found

**Content Types**:
- `application/json`: For single responses
- `text/event-stream`: For streaming responses (SSE)

### Session Management

#### Session Creation
1. Client sends initial request WITHOUT `Mcp-Session-Id` header
2. Server creates new session with unique ID (UUID hex format)
3. Server returns session ID in response header: `Mcp-Session-Id: <session-id>`
4. Client MUST include this session ID in all subsequent requests

#### Session Requirements
- Session IDs must be:
  - Globally unique
  - Cryptographically secure (UUID4)
  - Contain only visible ASCII characters
- Servers MAY require session IDs for all requests except initialization
- Servers MAY terminate sessions at any time

#### Session Termination
- Server responds with `404 Not Found` for terminated sessions
- Client can explicitly terminate: `DELETE /mcp` with `Mcp-Session-Id` header

### Stateless vs Stateful Modes

**Stateful Mode** (default):
- Maintains session state between requests
- Requires session ID handling
- Suitable for long-running conversations

**Stateless Mode**:
- No session persistence
- Each request creates new transport
- No session ID required
- Configure with: `FastMCP("Server", stateless_http=True)`

## Implementation Example

### Server Setup (FastMCP)

```python
from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("VideoAnalysisServer")

# Register tools
@mcp.tool()
async def process_video(video_path: str) -> str:
    return f"Processing {video_path}"

# Run with streamable HTTP
if __name__ == "__main__":
    # Default configuration
    mcp.run(transport="streamable-http")
    
    # Custom configuration
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        stateless_http=False,  # Enable session management
        json_response=False    # Allow SSE streaming
    )
```

### Client Implementation

```python
import httpx
import json

class MCPHttpClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id = None
        self.client = httpx.AsyncClient()
    
    async def initialize(self):
        """Initialize connection and get session ID"""
        response = await self.client.post(
            f"{self.server_url}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0",
                    "clientInfo": {
                        "name": "MyClient",
                        "version": "1.0"
                    }
                }
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        )
        
        # Store session ID from response header
        self.session_id = response.headers.get("mcp-session-id")
        return response.json()
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call an MCP tool"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Include session ID if available
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        
        response = await self.client.post(
            f"{self.server_url}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": f"call_{tool_name}",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            },
            headers=headers
        )
        
        if response.status_code == 400:
            # Likely missing session ID
            raise Exception("Session required but not established")
        
        return response.json()
```

## Request Flow

1. **Initial Connection**:
   ```
   POST /mcp
   Headers: Accept: application/json, text/event-stream
   Body: {"method": "initialize", ...}
   
   Response:
   Headers: Mcp-Session-Id: abc123...
   Body: {"result": {"protocolVersion": "1.0", ...}}
   ```

2. **Tool Discovery**:
   ```
   POST /mcp
   Headers: Mcp-Session-Id: abc123...
   Body: {"method": "tools/list", ...}
   
   Response:
   Body: {"result": {"tools": [...]}}
   ```

3. **Tool Invocation**:
   ```
   POST /mcp
   Headers: Mcp-Session-Id: abc123...
   Body: {"method": "tools/call", "params": {"name": "process_video", ...}}
   
   Response:
   Body: {"result": {"content": [{"type": "text", "text": "..."}]}}
   ```

## Streaming Responses (SSE)

For long-running operations, the server can upgrade to Server-Sent Events:

1. Client sends request with `Accept: text/event-stream`
2. Server responds with `Content-Type: text/event-stream`
3. Server sends events in SSE format:
   ```
   event: message
   data: {"jsonrpc": "2.0", "method": "progress", ...}
   
   event: message
   data: {"jsonrpc": "2.0", "result": {...}}
   ```

## Security Considerations

1. **DNS Rebinding Protection**: Servers MUST validate Origin header
2. **Local Binding**: Servers should bind to `127.0.0.1` for localhost
3. **Authentication**: Optional OAuth2/Bearer token support
4. **HTTPS**: Recommended for production deployments

## Error Handling

Common error scenarios:

1. **Missing Session ID**: 
   - Response: 400 Bad Request
   - Message: "Session ID required"

2. **Invalid Session**:
   - Response: 404 Not Found
   - Message: "Session not found or terminated"

3. **Protocol Error**:
   - Response: 400 Bad Request
   - Body: JSON-RPC error response

## Testing the Protocol

```bash
# Health check
curl http://localhost:8000/health

# Initialize session
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{}}'

# Use session ID from response
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: <session-id>" \
  -d '{"jsonrpc":"2.0","id":"2","method":"tools/list","params":{}}'
```

## FastMCP Internal Implementation

The streamable HTTP transport is implemented using:

1. **StreamableHTTPSessionManager**: Manages session lifecycle
2. **StreamableHTTPServerTransport**: Handles individual sessions
3. **EventStore**: Optional persistent event storage
4. **Starlette/ASGI**: Web framework integration

Key files in FastMCP:
- `fastmcp/server/http.py`: HTTP app creation
- `mcp/server/streamable_http_manager.py`: Session management
- `mcp/server/streamable_http.py`: Transport implementation