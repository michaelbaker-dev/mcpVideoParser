#!/usr/bin/env python3
"""Test HTTP MCP client and server integration."""
import asyncio
import subprocess
import sys
import time
import socket
import psutil
from pathlib import Path

import httpx
from rich.console import Console

console = Console()


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def kill_processes_on_port(port: int):
    """Kill any processes using the specified port."""
    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.info['connections'] or []:
                if conn.laddr.port == port:
                    console.print(f"[yellow]Killing process {proc.info['pid']} ({proc.info['name']}) on port {port}[/yellow]")
                    proc.terminate()
                    killed.append(proc.info['pid'])
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return killed


def kill_existing_mcp_servers():
    """Kill any existing MCP video server processes."""
    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'server.py' in cmdline and 'mcp-video' in cmdline:
                console.print(f"[yellow]Killing existing MCP server process {proc.info['pid']}[/yellow]")
                proc.terminate()
                killed.append(proc.info['pid'])
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return killed


async def test_http_integration():
    """Test the HTTP MCP server and client integration."""
    port = 8000
    server_script = Path(__file__).parent / "server.py"
    client_script = Path(__file__).parent / "standalone_client" / "mcp_http_client.py"
    
    console.print("[bold]Testing HTTP MCP Integration[/bold]")
    
    # Step 0: Clean up existing processes
    console.print("\n[yellow]0. Checking for existing processes...[/yellow]")
    
    # Kill existing MCP servers
    killed_servers = kill_existing_mcp_servers()
    if killed_servers:
        console.print(f"   Killed {len(killed_servers)} existing MCP server(s)")
        await asyncio.sleep(2)  # Give time for cleanup
    
    # Check if port is in use and kill processes if needed
    if is_port_in_use(port):
        console.print(f"   Port {port} is in use, killing processes...")
        killed_port = kill_processes_on_port(port)
        if killed_port:
            console.print(f"   Killed {len(killed_port)} process(es) on port {port}")
            await asyncio.sleep(2)  # Give time for cleanup
    
    # Verify port is now free
    if is_port_in_use(port):
        console.print(f"[red]❌ Port {port} is still in use after cleanup[/red]")
        return False
    
    console.print(f"[green]✅ Port {port} is available[/green]")
    
    # Step 1: Start HTTP server
    console.print("\n[yellow]1. Starting HTTP MCP Server...[/yellow]")
    
    server_process = subprocess.Popen([
        sys.executable, str(server_script), 
        "--http", "--host", "localhost", "--port", str(port)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for server to start
        console.print("   Waiting for server to start...")
        start_time = time.time()
        server_ready = False
        
        while time.time() - start_time < 15:  # 15 second timeout
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://localhost:{port}/health", timeout=2.0)
                    if response.status_code == 200:
                        server_ready = True
                        break
            except Exception as e:
                # Print server output if available for debugging
                if server_process.poll() is not None:
                    stdout, stderr = server_process.communicate()
                    console.print(f"[red]Server exited early. Output: {stdout.decode()[:500]}[/red]")
                    console.print(f"[red]Server errors: {stderr.decode()[:500]}[/red]")
                    break
            await asyncio.sleep(1)
        
        if not server_ready:
            console.print("[red]❌ Server failed to start[/red]")
            return False
        
        console.print("[green]✅ Server started successfully[/green]")
        
        # Step 2: Test basic connectivity
        console.print("\n[yellow]2. Testing basic connectivity...[/yellow]")
        
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get(f"http://localhost:{port}/health")
            console.print(f"   Health check: {response.status_code}")
            
            # Test tools list
            tools_response = await client.post(
                f"http://localhost:{port}/mcp/list_tools",
                json={"method": "tools/list", "params": {}}
            )
            console.print(f"   Tools list: {tools_response.status_code}")
            
            if tools_response.status_code == 200:
                tools_data = tools_response.json()
                if "result" in tools_data and "tools" in tools_data["result"]:
                    tool_count = len(tools_data["result"]["tools"])
                    console.print(f"   Found {tool_count} tools")
                    
                    # List tool names
                    tool_names = [tool["name"] for tool in tools_data["result"]["tools"]]
                    console.print(f"   Tools: {', '.join(tool_names)}")
                else:
                    console.print("[yellow]   Warning: Unexpected tools response format[/yellow]")
            
            # Test a simple tool call
            console.print("\n[yellow]3. Testing tool call...[/yellow]")
            tool_response = await client.post(
                f"http://localhost:{port}/mcp/call_tool",
                json={
                    "method": "tools/call",
                    "params": {
                        "name": "get_video_stats",
                        "arguments": {}
                    }
                }
            )
            console.print(f"   Tool call status: {tool_response.status_code}")
            
            if tool_response.status_code == 200:
                result = tool_response.json()
                console.print("[green]✅ Tool call successful[/green]")
                console.print(f"   Response: {result}")
            else:
                console.print(f"[yellow]Tool call failed: {tool_response.text}[/yellow]")
        
        console.print("\n[green]✅ HTTP MCP integration test passed![/green]")
        
        # Step 3: Show usage instructions
        console.print(f"\n[bold blue]🎉 Your HTTP MCP server is working![/bold blue]")
        console.print(f"\nTo use the HTTP client:")
        console.print(f"[dim]python {client_script} --server http://localhost:{port}[/dim]")
        
        console.print(f"\nTo use from a remote machine:")
        console.print(f"[dim]python {client_script} --server http://YOUR_SERVER_IP:{port}[/dim]")
        
        console.print(f"\nTo start server for remote access:")
        console.print(f"[dim]python {server_script} --http --host 0.0.0.0 --port {port}[/dim]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        return False
        
    finally:
        # Clean up
        console.print("\n[dim]Stopping server...[/dim]")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()


if __name__ == "__main__":
    success = asyncio.run(test_http_integration())
    sys.exit(0 if success else 1)