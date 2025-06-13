#!/usr/bin/env python3
"""Debug script to test server startup."""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from src.storage.manager import StorageManager
from src.processors.video import VideoProcessor
from src.llm.ollama_client import OllamaClient
from src.tools.mcp_tools import VideoTools
from src.utils.config import get_config


async def debug_server():
    """Debug server startup."""
    print("🔍 Debug: Starting server components...")
    
    try:
        # Load config
        config = get_config()
        print("✅ Config loaded")
        
        # Initialize components
        storage = StorageManager()
        print("✅ Storage manager initialized")
        
        llm_client = OllamaClient()
        print("✅ LLM client initialized")
        
        processor = VideoProcessor(storage, llm_client)
        print("✅ Video processor initialized")
        
        # Create MCP server
        mcp = FastMCP("debug-test")
        print("✅ MCP server created")
        
        # Register tools
        tools = VideoTools(processor, storage)
        tools.register(mcp)
        print("✅ Tools registered")
        
        # Test streamable HTTP app
        app = mcp.streamable_http_app()
        print(f"✅ HTTP app created: {type(app)}")
        print(f"   App routes: {len(getattr(app, 'routes', []))}")
        
        # Try to add health route
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        
        async def health_check(request):
            return JSONResponse({"status": "healthy"})
        
        app.routes.append(Route("/health", health_check))
        print("✅ Health route added")
        
        # Test uvicorn config
        import uvicorn
        config = uvicorn.Config(app, host="localhost", port=8001, log_level="debug")
        print("✅ Uvicorn config created")
        
        # Test if server would start (don't actually start)
        print("🎯 All components ready! Server should work.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_server())