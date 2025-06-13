#!/usr/bin/env python3
"""MCP Video Analysis Server - Main entry point."""
import asyncio
import signal
import sys
import os
import psutil
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from src.storage.manager import StorageManager
from src.processors.video import VideoProcessor
from src.llm.ollama_client import OllamaClient
from src.tools.mcp_tools import VideoTools
from src.utils.config import get_config, ConfigManager
from src.utils.logging import setup_logging, get_logger

# Server metadata
SERVER_NAME = "mcp-video-analysis"
SERVER_VERSION = "0.1.1"


def check_and_kill_existing_server(port: int, script_name: str = "mcp_video_server.py"):
    """Check if server is already running and kill it if necessary."""
    current_pid = os.getpid()
    killed_processes = []
    
    # Check for processes listening on the port (requires elevated permissions on macOS)
    try:
        for conn in psutil.net_connections():
            if conn.laddr and conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    process = psutil.Process(conn.pid)
                    if process.pid != current_pid:
                        process.terminate()
                        killed_processes.append(f"Process {conn.pid} on port {port}")
                        try:
                            process.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except psutil.AccessDenied:
        # Fall back to checking with lsof on macOS/Linux
        try:
            import subprocess
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.stdout:
                for pid_str in result.stdout.strip().split('\n'):
                    try:
                        pid = int(pid_str)
                        if pid != current_pid:
                            os.kill(pid, signal.SIGTERM)
                            killed_processes.append(f"Process {pid} on port {port}")
                            # Give it time to terminate gracefully
                            import time
                            time.sleep(0.5)
                            # Force kill if still running
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    except (ValueError, ProcessLookupError):
                        pass
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    # Check for processes running the script
    script_path = Path(__file__).resolve()
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if process.pid == current_pid:
                continue
                
            cmdline = process.info.get('cmdline', [])
            if cmdline and any(script_name in arg or str(script_path) in arg for arg in cmdline):
                process.terminate()
                killed_processes.append(f"Process {process.pid} running {script_name}")
                try:
                    process.wait(timeout=3)
                except psutil.TimeoutExpired:
                    process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return killed_processes


async def main():
    """Main server entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description=f"{SERVER_NAME} v{SERVER_VERSION}")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server")
    parser.add_argument("--host", default="localhost", help="HTTP host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Setup logging
    setup_logging(config.logging.model_dump())
    logger = get_logger(__name__)
    
    # Check and kill existing server if running in HTTP mode
    if args.http:
        killed = check_and_kill_existing_server(args.port)
        if killed:
            logger.info(f"Killed existing processes: {', '.join(killed)}")
            # Brief pause to ensure port is released
            await asyncio.sleep(0.5)
    
    logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
    
    try:
        # Initialize components
        logger.info("Initializing storage manager...")
        storage = StorageManager()
        
        logger.info("Initializing LLM client...")
        llm_client = OllamaClient()
        
        logger.info("Initializing video processor...")
        processor = VideoProcessor(storage, llm_client)
        
        # Create MCP server
        logger.info("Creating MCP server...")
        mcp = FastMCP(SERVER_NAME)
        
        # Register tools
        logger.info("Registering MCP tools...")
        tools = VideoTools(processor, storage)
        tools.register(mcp)
        
        # Setup shutdown handlers
        shutdown_event = asyncio.Event()
        
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run server
        logger.info(f"About to check HTTP flag: {args.http}")
        if args.http:
            logger.info(f"{SERVER_NAME} HTTP server starting on {args.host}:{args.port}")
            # Get the ASGI app and run with uvicorn
            import uvicorn
            app = mcp.streamable_http_app()
            
            # Run with uvicorn
            await uvicorn.Server(
                uvicorn.Config(
                    app=app,
                    host=args.host,
                    port=args.port,
                    log_level="info"
                )
            ).serve()
        else:
            logger.info(f"{SERVER_NAME} is ready to accept stdio connections")
            await mcp.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if 'processor' in locals():
            processor.cleanup()
        if 'tools' in locals():
            await tools.cleanup()


if __name__ == "__main__":
    # Call the main function which handles arguments
    asyncio.run(main())