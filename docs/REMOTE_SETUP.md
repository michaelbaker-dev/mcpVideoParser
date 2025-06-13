# Remote MCP Server Setup

The MCP Video Chat Client is designed to work with both local and remote MCP servers, making it suitable for future mobile apps or distributed deployments.

## Architecture for Remote Access

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Mobile App    │     │   Web Client    │     │  Desktop Client │
│ (Future)        │     │ (Future)        │     │  (Current)      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         │ MCP Protocol over:    │                         │
         │ • WebSocket           │                         │
         │ • SSH Tunnel          │                         │
         │ • Custom Transport    │                         │
         ▼                       ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Video Server (Remote)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Current Remote Options

### 1. SSH Tunnel Method (Recommended)

On the server machine:
```bash
# Start the MCP server
python server.py
```

On the client machine:
```bash
# Create SSH tunnel to forward MCP stdio
ssh -L 9999:localhost:9999 user@server-ip "cd /path/to/mcp-video-server && python server.py"

# In another terminal, connect the chat client
./standalone_client/video_client.py chat --server-command "nc localhost 9999"
```

### 2. Custom Transport Script

Create a wrapper script that handles remote communication:

```python
#!/usr/bin/env python3
# remote_mcp_wrapper.py
import sys
import subprocess
import paramiko  # for SSH

def remote_mcp_stdio(host, username, password, mcp_path):
    """Connect to remote MCP server via SSH."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=username, password=password)
    
    # Execute MCP server remotely
    stdin, stdout, stderr = ssh.exec_command(f"cd {mcp_path} && python server.py")
    
    # Bridge stdio
    # ... implementation details ...
```

Then use:
```bash
./standalone_client/video_client.py chat --server-command "./remote_mcp_wrapper.py server-ip"
```

### 3. WebSocket Bridge (Future)

For web and mobile clients, a WebSocket bridge can be implemented:

```python
# mcp_websocket_bridge.py
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

@app.websocket("/mcp")
async def mcp_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Bridge WebSocket to MCP stdio
    # ... implementation ...
```

## Configuration for Different Environments

### Local Development
```bash
# Default - server starts automatically
./standalone_client/video_client.py chat
```

### Remote Server
```bash
# Specify custom server command
./standalone_client/video_client.py chat \
    --server-command "ssh user@192.168.1.100 'cd /opt/mcp-video && python server.py'"
```

### Docker Container
```bash
# Run server in Docker
docker run -d --name mcp-video -p 8080:8080 mcp-video-server

# Connect client
./standalone_client/video_client.py chat \
    --server-command "docker exec -i mcp-video python server.py"
```

## Mobile App Architecture (Future)

For a mobile app, the architecture would be:

1. **Mobile App (React Native / Flutter)**
   - UI for chat interface
   - Sends queries via REST API or WebSocket

2. **API Gateway**
   - Handles authentication
   - Routes to appropriate MCP server
   - Manages session state

3. **MCP Server Farm**
   - Multiple MCP servers for scalability
   - Shared storage backend (S3, NFS, etc.)
   - Shared database (PostgreSQL, etc.)

Example mobile client code:
```javascript
// React Native example
class MCPVideoClient {
  async connect(serverUrl, apiKey) {
    this.ws = new WebSocket(`${serverUrl}/mcp`);
    this.ws.on('open', () => this.authenticate(apiKey));
  }
  
  async sendQuery(query) {
    this.ws.send(JSON.stringify({
      type: 'query',
      content: query
    }));
  }
}
```

## Security Considerations

1. **Authentication**: Add API keys or OAuth for remote access
2. **Encryption**: Use TLS/SSL for all remote connections
3. **Authorization**: Implement user-based access control
4. **Rate Limiting**: Prevent abuse of video processing resources

## Scaling Considerations

1. **Load Balancing**: Distribute requests across multiple MCP servers
2. **Caching**: Cache frequently accessed video analyses
3. **Queue System**: Use message queues for video processing
4. **Storage**: Use distributed storage for video files

## Testing Remote Setup

```bash
# Test local connection
python test_mcp_connection.py --server localhost

# Test remote connection
python test_mcp_connection.py --server 192.168.1.100 --method ssh

# Test WebSocket connection
python test_mcp_connection.py --server ws://api.example.com/mcp
```

The MCP protocol's stdio-based design makes it flexible for various transport mechanisms, ensuring the chat client can work seamlessly whether the server is local, remote, or even distributed across multiple machines.