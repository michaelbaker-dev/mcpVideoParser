#!/usr/bin/env python3
"""Quick script to process video through MCP HTTP server."""
import asyncio
import httpx
import json

async def process_video():
    """Process the sample video through MCP server."""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
        # Initialize session
        print("Initializing MCP session...")
        init_response = await client.post(
            "http://localhost:8000/mcp/",  # Note the trailing slash
            json={
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "Quick Process", "version": "1.0.0"}
                }
            }
        )
        
        # Get session ID
        session_id = init_response.headers.get('mcp-session-id')
        if session_id:
            client.headers['mcp-session-id'] = session_id
            print(f"Session ID: {session_id[:8]}...")
        
        # Send initialized notification
        await client.post(
            "http://localhost:8000/mcp/",
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
        )
        
        # Process video
        print("\nProcessing video...")
        process_response = await client.post(
            "http://localhost:8000/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": "process",
                "method": "tools/call",
                "params": {
                    "name": "process_video",
                    "arguments": {
                        "video_path": "/Users/michaelbaker/mcp/video/mcp-video-server/video_data/originals/sample_video.mp4",
                        "location": "test"
                    }
                }
            }
        )
        
        # Parse response
        if process_response.headers.get('content-type', '').startswith('text/event-stream'):
            lines = process_response.text.strip().split('\n')
            for line in lines:
                if line.startswith('data: '):
                    result = json.loads(line[6:])
                    break
        else:
            result = process_response.json()
        
        if "result" in result:
            content = result["result"]["content"][0]["text"]
            data = json.loads(content)
            print(f"\n✅ Video processed successfully!")
            print(f"Video ID: {data['video_id']}")
            print(f"Status: {data['status']}")
            print(f"Frames analyzed: {data['frames_analyzed']}")
        else:
            print(f"❌ Error: {result}")

if __name__ == "__main__":
    print("Make sure the MCP server is running on http://localhost:8000")
    asyncio.run(process_video())