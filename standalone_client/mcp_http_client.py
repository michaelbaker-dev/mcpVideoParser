#!/usr/bin/env python3
"""HTTP-based MCP client for video analysis that can run remotely."""
import asyncio
import sys
import json
import httpx
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner
from rich.columns import Columns
from rich.text import Text
from rich import box

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.utils.logging import get_logger
from src.llm.ollama_client import OllamaClient

console = Console()


def parse_sse_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse SSE format response to extract JSON data."""
    lines = response_text.strip().split('\n')
    for i, line in enumerate(lines):
        if line.startswith('data: '):
            # Skip empty data lines
            if line == 'data: ':
                continue
            try:
                return json.loads(line[6:])  # Skip "data: " prefix
            except json.JSONDecodeError:
                continue
    return None


class MCPHttpClient:
    """HTTP-based MCP client for video analysis."""
    
    def __init__(
        self, 
        server_url: str = "http://localhost:8000",
        chat_llm_model: str = "llama2:latest"
    ):
        self.server_url = server_url.rstrip('/')
        self.session: Optional[httpx.AsyncClient] = None
        self.mcp_session_id: Optional[str] = None
        self.mcp_endpoint: str = "/mcp"  # Will be determined during connection
        self.available_tools: Dict[str, Any] = {}
        self.running = True
        
        # Chat LLM for formatting responses
        self.chat_llm_model = chat_llm_model
        self.chat_llm: Optional[OllamaClient] = None
        
        # Config
        self.config = get_config()
        self.logger = get_logger(__name__)
    
    async def connect(self):
        """Connect to the HTTP MCP server."""
        console.print(f"[yellow]Connecting to MCP server at {self.server_url}...[/yellow]")
        
        # Create HTTP client with proper headers for MCP
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json"
        }
        self.session = httpx.AsyncClient(
            timeout=30.0, 
            headers=headers,
            follow_redirects=True  # Follow redirects automatically
        )
        
        try:
            console.print("[green]✅ Connected to MCP Video Analysis Server[/green]")
            
            # Initialize MCP session with first request
            console.print("[yellow]Initializing MCP session...[/yellow]")
            
            # Try different endpoint paths
            mcp_endpoints = ["/mcp", "/mcp/", "/"]
            init_response = None
            
            for endpoint in mcp_endpoints:
                try:
                    # Make initial request to establish session
                    init_response = await self.session.post(
                        f"{self.server_url}{endpoint}",
                        json={
                            "jsonrpc": "2.0",
                            "id": "initialize",
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2025-03-26",
                                "capabilities": {},
                                "clientInfo": {
                                    "name": "MCP HTTP Video Client",
                                    "version": "1.0.0"
                                }
                            }
                        }
                    )
                    
                    if init_response.status_code in [200, 201]:
                        console.print(f"[green]✅ Found MCP endpoint at {endpoint}[/green]")
                        self.mcp_endpoint = endpoint
                        break
                    elif init_response.status_code == 307:
                        console.print(f"[yellow]Redirect from {endpoint} to {init_response.headers.get('location', 'unknown')}[/yellow]")
                except Exception as e:
                    console.print(f"[dim]Failed {endpoint}: {e}[/dim]")
            
            if not init_response or init_response.status_code not in [200, 201]:
                console.print("[red]Failed to find working MCP endpoint[/red]")
                return
            
            # Extract session ID from response header
            session_id_found = False
            # Check for session ID in response headers (case-insensitive)
            for key, value in init_response.headers.items():
                if key.lower() == 'mcp-session-id':
                    self.mcp_session_id = value
                    console.print(f"[green]✅ Session established: {self.mcp_session_id[:8]}...[/green]")
                    # Add session ID to all future requests using the exact case from server
                    self.session.headers[key] = self.mcp_session_id
                    console.print(f"[dim]Using header '{key}' for session ID[/dim]")
                    session_id_found = True
                    break
            
            if not session_id_found:
                console.print("[yellow]Warning: No session ID received from server[/yellow]")
                console.print(f"[dim]Response headers: {dict(init_response.headers)}[/dim]")
            
            # Parse initialization response
            try:
                # Check if response is SSE format
                if init_response.headers.get('content-type', '').startswith('text/event-stream'):
                    init_data = parse_sse_response(init_response.text)
                    if not init_data:
                        raise ValueError("No data found in SSE response")
                else:
                    init_data = init_response.json()
                
                if "result" in init_data:
                    console.print(f"[dim]Server initialized: protocol {init_data['result'].get('protocolVersion', 'unknown')}[/dim]")
                elif "error" in init_data:
                    console.print(f"[red]Initialization error: {init_data['error']}[/red]")
                    return
            except Exception as e:
                console.print(f"[red]Failed to parse initialization response: {e}[/red]")
                console.print(f"[dim]Response: {init_response.text[:200]}...[/dim]")
                return
            
            # Send initialized notification to complete handshake
            console.print("[dim]Completing initialization handshake...[/dim]")
            initialized_response = await self.session.post(
                f"{self.server_url}{self.mcp_endpoint}",
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
            )
            
            if initialized_response.status_code not in [200, 202]:
                console.print(f"[yellow]Warning: Initialized notification returned {initialized_response.status_code}[/yellow]")
            
            # Initialize chat LLM
            console.print(f"[yellow]Initializing chat LLM ({self.chat_llm_model})...[/yellow]")
            self.chat_llm = OllamaClient()
            
            # Small delay after initialization
            await asyncio.sleep(0.5)
            
            # Now list available tools
            console.print("[yellow]Fetching available tools...[/yellow]")
            # Debug: show all headers
            console.print(f"[dim]Headers: {dict(self.session.headers)}[/dim]")
            tools_response = await self.session.post(
                f"{self.server_url}{self.mcp_endpoint}",
                json={
                    "jsonrpc": "2.0",
                    "id": "list_tools",
                    "method": "tools/list",
                    "params": {}
                }
            )
            
            if tools_response.status_code == 200:
                try:
                    # Parse response (SSE or JSON)
                    if tools_response.headers.get('content-type', '').startswith('text/event-stream'):
                        tools_data = parse_sse_response(tools_response.text)
                        if not tools_data:
                            raise ValueError("No data found in SSE response")
                    else:
                        tools_data = tools_response.json()
                    
                    if "result" in tools_data and "tools" in tools_data["result"]:
                        self.available_tools = {
                            tool["name"]: tool 
                            for tool in tools_data["result"]["tools"]
                        }
                        console.print(f"[dim]Available tools: {len(self.available_tools)}[/dim]")
                    else:
                        console.print("[yellow]Warning: No tools found in response[/yellow]")
                        console.print(f"[dim]Response: {tools_data}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not parse tools response: {e}[/yellow]")
                    console.print(f"[dim]Response: {tools_response.text[:200]}...[/dim]")
            else:
                console.print(f"[yellow]Warning: Could not list tools ({tools_response.status_code})[/yellow]")
                if tools_response.status_code == 400:
                    console.print(f"[red]HTTP error {tools_response.status_code}:[/red]")
                    console.print(f"[dim]{tools_response.text}[/dim]")
            
            # Get initial stats
            try:
                stats_result = await self.call_tool("get_video_stats", {})
                if stats_result and "storage" in stats_result:
                    total_videos = stats_result["storage"].get("total_videos", 0)
                    console.print(f"[dim]System ready with {total_videos} videos[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not get initial stats: {e}[/yellow]")
            
            # Run the chat interface
            await self.run_chat()
            
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to server at {self.server_url}")
        except Exception as e:
            console.print(f"[red]Connection error: {e}[/red]")
            raise
    
    async def disconnect(self):
        """Disconnect from the server."""
        # Just close the HTTP session - the server will clean up
        if self.session:
            await self.session.aclose()
        if self.chat_llm:
            await self.chat_llm.close()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call an MCP tool via HTTP."""
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        try:
            # Make MCP tool call
            request_data = {
                "jsonrpc": "2.0",
                "id": f"call_{tool_name}",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = await self.session.post(
                f"{self.server_url}{self.mcp_endpoint}",
                json=request_data
            )
            
            if response.status_code == 200:
                # Parse response (SSE or JSON)
                if response.headers.get('content-type', '').startswith('text/event-stream'):
                    result = parse_sse_response(response.text)
                    if not result:
                        console.print(f"[red]Failed to parse SSE response[/red]")
                        return None
                else:
                    result = response.json()
                    
                if "result" in result:
                    # Handle tool response structure
                    tool_result = result["result"]
                    if isinstance(tool_result, dict) and "content" in tool_result:
                        # Extract text from content array
                        content = tool_result["content"]
                        if isinstance(content, list) and len(content) > 0:
                            return content[0].get("text", str(content[0]))
                    # If direct result, return as is
                    return tool_result
                elif "error" in result:
                    console.print(f"[red]Tool error: {result['error']['message']}[/red]")
                    return None
            else:
                console.print(f"[red]HTTP error {response.status_code}: {response.text}[/red]")
                return None
                
        except Exception as e:
            console.print(f"[red]Error calling tool {tool_name}: {e}[/red]")
            return None
    
    async def run_chat(self):
        """Run the interactive chat loop."""
        self.display_welcome()
        
        # Get initial status
        console.print("\n[dim]Checking system status...[/dim]")
        status = await self.handle_command("status")
        if status:
            console.print(status)
        
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if not user_input.strip():
                    continue
                
                # Show thinking indicator
                with Live(Spinner("dots", text="Thinking..."), refresh_per_second=10):
                    response = await self.handle_command(user_input)
                
                if response:
                    # Display response
                    if response.startswith("##") or response.startswith("**"):
                        # Markdown response
                        console.print(Panel(Markdown(response), border_style="blue"))
                    else:
                        # Plain text response
                        console.print(f"[bold green]Assistant:[/bold green] {response}")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit gracefully[/yellow]")
            except EOFError:
                self.running = False
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def display_welcome(self):
        """Display welcome message."""
        welcome_text = f"""
# 🎬 MCP HTTP Video Chat Client

Connected to: `{self.server_url}`
Chat LLM: `{self.chat_llm_model}`

Ask me anything about your videos! I can help with:
- **Transcript analysis** - "What was said in the video?"
- **Visual analysis** - "Describe what you can see"
- **Location queries** - "What happened at the shed today?"
- **Content search** - "Find videos with cars"
- **Video summaries** - "Summarize video vid_abc123"

Type **help** for comprehensive query examples or **exit** to quit.
        """
        console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="green"))
    
    async def handle_command(self, user_input: str) -> Optional[str]:
        """Handle user commands and queries."""
        command = user_input.strip().lower()
        
        # System commands
        if command in ['exit', 'quit', 'bye']:
            self.running = False
            return "Goodbye! 👋"
        
        if command == 'help':
            help_text = """
## 📚 Video Analysis Commands

### 🎤 For Transcript Content (What Was Said)
- **"What was said in the video?"** - Get transcript details
- **"Give me the transcript summary"** - Focus on spoken content  
- **"What did they talk about in [video_id]?"** - Transcript from specific video
- **"Find videos mentioning [topic]"** - Search through all transcripts
- **"What was said at 2 minutes into the video?"** - Transcript at specific time

### 🎬 For Visual Analysis (What Was Shown)
- **"Summarize the video [video_id]"** - AI visual summary from frames
- **"What cars were shown in the video?"** - Visual content description
- **"Describe what you can see in the video"** - Frame analysis focus
- **"What's happening visually at 3 minutes?"** - Visual content at timestamp

### 🔍 For Combined Analysis
- **"Tell me everything about [video_id]"** - Both transcript + visual
- **"What happened in this race video?"** - Complete story
- **"Show me the latest videos"** - List recent videos with summaries

### 📍 Location & Time Queries  
- **"videos from [location] [time]"** - Filter by location/date
- **"what happened at the shed today?"** - Location + time specific
- **"show videos from last week"** - Time-based filtering
- **"find videos from the driveway yesterday"** - Combined filters

### ⏱️ Moment Analysis
- **"what happens at [time] in [video_id]"** - Specific timestamp analysis
- **"analyze the conversation at 120 seconds"** - Audio + visual at time
- **"what was happening at 2:30?"** - Moment breakdown

### 🔧 System Commands  
- **status** - Check system status and video count
- **tools** - List all available MCP tools
- **clear** - Clear the chat screen
- **help** - Show this comprehensive help
- **exit** - Exit the chat client

### 💡 Pro Tips
- The system keeps **complete transcripts** (1000+ words per video)
- **Visual analysis** comes from AI examining video frames  
- **Search** works across both transcripts and visual descriptions
- Use **specific video IDs** for targeted analysis
- **Recent/latest** queries show recently processed videos
            """
            return help_text.strip()
        
        if command == 'clear':
            console.clear()
            return None
        
        if command == 'tools':
            if self.available_tools:
                tools_list = "\n".join([f"- **{name}**: {tool.get('description', 'No description')}" 
                                       for name, tool in self.available_tools.items()])
                return f"## Available Tools\n\n{tools_list}"
            return "No tools available"
        
        if command == 'status':
            result = await self.call_tool('get_video_stats', {})
            if result:
                try:
                    # Parse the JSON result if it's a string
                    if isinstance(result, str):
                        result = json.loads(result)
                    
                    storage = result.get('storage', {})
                    system = result.get('system', {})
                    
                    status_text = f"""
## 📊 System Status

- **Total videos**: {storage.get('total_videos', 0)}
- **Processed videos**: {storage.get('processed_videos', 0)}  
- **Storage used**: {storage.get('total_size_gb', 0):.2f} GB
- **Ollama available**: {'✅' if system.get('ollama_available') else '❌'}
- **Chat LLM**: {self.chat_llm_model}
- **Video LLM**: {self.config.llm.vision_model}
- **Server**: {self.server_url}
                    """
                    return status_text.strip()
                except (json.JSONDecodeError, AttributeError):
                    return f"## 📊 System Status\n\nRaw response: {result}"
            return "❌ Could not get system status"
        
        # Process as natural language query
        return await self.process_query(user_input)
    
    async def get_video_context(self, limit: int = 5) -> str:
        """Get recent video context for chat LLM."""
        try:
            # Get recent videos
            result = await self.call_tool('query_location_time', {
                'time_query': 'recent',
                'limit': limit
            })
            
            if result and isinstance(result, str):
                result = json.loads(result)
            
            if not result or not result.get('results'):
                return "No videos currently in the system."
            
            # Format video context
            context_parts = [f"Current videos in the system ({len(result['results'])} total):"]
            for video in result['results']:
                context_parts.append(f"\n- Video {video['video_id']} at {video['location']} ({video['time_ago']})")
                if video.get('summary'):
                    context_parts.append(f"  Content: {video['summary'][:200]}...")
            
            return "\n".join(context_parts)
        except Exception as e:
            self.logger.error(f"Error getting video context: {e}")
            return "Unable to retrieve video information."
    
    async def process_query(self, query: str) -> str:
        """Process natural language query."""
        # Parse intent from query using simple patterns
        query_lower = query.lower()
        
        # Check if this is a general greeting or chat message
        greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                         'good evening', 'how are you', 'what can you do']
        if any(word in query_lower for word in greeting_words):
            # Use chat LLM for general conversation
            if self.chat_llm:
                try:
                    # Get video context
                    video_context = await self.get_video_context()
                    
                    prompt = f"""You are a helpful video analysis assistant. You help users find and understand videos.

{video_context}

User said: {query}

Respond naturally and mention that you can help them:
- Find recent videos
- Search for specific content
- Summarize videos
- Answer questions about what happened in videos

Keep your response brief and friendly."""
                    
                    response = await self.chat_llm.generate_text(prompt, model=self.chat_llm_model)
                    return response
                except Exception as e:
                    console.print(f"[yellow]Chat LLM error: {e}[/yellow]")
            
            # Fallback response if chat LLM fails
            return """Hello! I'm your video analysis assistant. I can help you:
            
- Show recent videos ("show latest videos")
- Search for specific content ("find videos with cars")  
- Summarize videos ("summarize video vid_xyz")
- Answer questions about locations ("what happened at the shed today?")

What would you like to know about your videos?"""
        
        # Route to appropriate tool based on query
        elif any(word in query_lower for word in ['latest', 'recent', 'new', 'list']):
            # List recent videos
            result = await self.call_tool('query_location_time', {
                'time_query': 'recent',
                'limit': 10
            })
        elif 'summarize' in query_lower or 'summary' in query_lower:
            # Check if asking for summary of all videos or a specific video
            words = query.split()
            video_id = None
            for word in words:
                if word.startswith('vid_'):
                    video_id = word
                    break
            
            if video_id:
                # Summary of specific video
                result = await self.call_tool('get_video_summary', {
                    'video_id': video_id,
                    'detail_level': 'medium'
                })
            elif any(phrase in query_lower for phrase in ['all videos', 'all the videos', 'latest videos', 'recent videos', 'videos over', 'videos by date']):
                # Summary of multiple videos - use chat LLM
                videos_result = await self.call_tool('query_location_time', {
                    'time_query': 'recent',
                    'limit': 20
                })
                
                if videos_result:
                    try:
                        videos_data = json.loads(videos_result) if isinstance(videos_result, str) else videos_result
                        if videos_data.get('results'):
                            # Use chat LLM to summarize
                            videos_desc = []
                            for v in videos_data['results']:
                                videos_desc.append(f"- {v['video_id']} at {v['location']} ({v['time_ago']}): {v.get('summary', 'No description')}")
                            
                            prompt = f"""Summarize these videos based on the user's request.

User asked: {query}

Videos:
{chr(10).join(videos_desc)}

Provide a concise summary that addresses the user's request. If they asked for a short summary, keep it brief. 
If they asked by date, organize by time. Focus on the key events and patterns across the videos."""
                            
                            response = await self.chat_llm.generate_text(prompt, model=self.chat_llm_model)
                            return response
                    except Exception as e:
                        console.print(f"[yellow]Error summarizing videos: {e}[/yellow]")
                
                return "No videos found to summarize."
            else:
                return "Please specify a video ID (e.g., 'summarize video vid_abc123') or ask for 'summary of all videos'"
        
        elif any(word in query_lower for word in ['find', 'search', 'with']):
            # Search content
            # Extract search terms after 'find', 'search', or 'with'
            for keyword in ['find videos with ', 'search for ', 'find ', 'search ']:
                if keyword in query_lower:
                    search_term = query_lower.split(keyword, 1)[1]
                    break
            else:
                search_term = query_lower
            
            result = await self.call_tool('search_videos', {
                'query': search_term,
                'limit': 10
            })
        
        elif any(phrase in query_lower for phrase in ['happened', 'what happened', 'at the', 'from the']) or \
             (('at' in query_lower.split() or 'from' in query_lower.split()) and 
              any(loc in query_lower for loc in ['shed', 'door', 'garage', 'camera', 'location'])):
            # Location/time query - be more specific about "at" and "from"
            result = await self.call_tool('query_location_time', {
                'time_query': query,
                'limit': 10
            })
        
        else:
            # Check if this seems to be a video-related query
            # Use more specific patterns for better routing
            video_patterns = [
                'video', 'videos', 'recording', 'footage', 'clip', 'recorded',
                'what happened', 'what was', 'show me', 'find me', 'search for',
                'when did', 'where did', 'who was', 'saw'
            ]
            
            # Check for video-related patterns
            is_video_query = False
            for pattern in video_patterns:
                if pattern in query_lower:
                    is_video_query = True
                    break
            
            # Also check for ambiguous words with context
            if not is_video_query and 'what' in query_lower:
                # "what" alone is too ambiguous - check for video context
                video_context = ['happened', 'recorded', 'saw', 'showed', 'appeared', 'did']
                is_video_query = any(word in query_lower for word in video_context)
            
            if is_video_query:
                # Default to general search for video-related queries
                result = await self.call_tool('search_videos', {
                    'query': query,
                    'limit': 5
                })
            else:
                # Use chat LLM for non-video queries
                if self.chat_llm:
                    try:
                        # Get video context for awareness
                        video_context = await self.get_video_context()
                        
                        prompt = f"""You are a helpful assistant that can both discuss videos and have general conversations.

Current system context:
{video_context}

User said: {query}

This doesn't seem to be a direct video query, so respond naturally to their request.
If they're asking for a story, tell them one. If they want to chat, chat with them.
You can reference the videos in the system if it's relevant to the conversation.

Keep your response engaging and appropriate to their request."""
                        
                        response = await self.chat_llm.generate_text(prompt, model=self.chat_llm_model)
                        return response
                    except Exception as e:
                        console.print(f"[yellow]Chat LLM error: {e}[/yellow]")
                        return "I encountered an error with the chat system. Please try again."
                else:
                    return "I'm having trouble with the chat system. Please try asking about videos instead."
        
        # Format result
        if result:
            try:
                # Try to parse as JSON if it's a string
                if isinstance(result, str):
                    try:
                        parsed_result = json.loads(result)
                        return await self.format_response(parsed_result, query)
                    except json.JSONDecodeError:
                        return result
                else:
                    return await self.format_response(result, query)
            except Exception as e:
                return f"Got response but couldn't format it: {result}"
        else:
            return "I couldn't find any relevant information. Please try rephrasing your question."
    
    async def format_response(self, result: Any, original_query: str) -> str:
        """Format the tool response nicely."""
        if not result:
            return "No results found."
        
        # Handle different result types
        if isinstance(result, dict):
            if 'results' in result and isinstance(result['results'], list):
                # Video list results
                videos = result['results']
                if not videos:
                    # Use chat LLM to provide a more helpful response
                    if self.chat_llm and hasattr(self, 'last_query'):
                        try:
                            prompt = f"""The user asked: "{original_query}"
                            
No videos were found in the system. Provide a helpful response that:
1. Acknowledges no videos were found
2. Suggests they might need to process some videos first
3. Offers to help when they have videos

Keep it brief and friendly."""
                            response = await self.chat_llm.generate_text(prompt, model=self.chat_llm_model)
                            return response
                        except:
                            pass
                    
                    return "No videos found. To get started, process some videos using the 'process_video' command or wait for videos to be added to your monitored locations."
                
                response = f"## Found {len(videos)} video(s)\n\n"
                for i, video in enumerate(videos[:5], 1):
                    video_id = video.get('video_id', 'unknown')
                    location = video.get('location', 'unknown')
                    timestamp = video.get('timestamp', video.get('time_ago', 'unknown'))
                    summary = video.get('summary', video.get('description', 'No description'))
                    
                    response += f"**{i}. Video {video_id}**\n"
                    response += f"- Location: {location}\n"
                    response += f"- Time: {timestamp}\n"
                    response += f"- Summary: {summary}\n\n"
                
                if len(videos) > 5:
                    response += f"*... and {len(videos) - 5} more videos*\n"
                
                return response
            
            elif 'summary' in result:
                # Video summary result
                return f"## Video Summary\n\n{result['summary']}"
            
            elif 'answer' in result:
                # Q&A result
                return result['answer']
        
        # Default: return as formatted text
        return str(result)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP HTTP Video Chat Client")
    parser.add_argument("--server", default="http://localhost:8000", 
                       help="MCP server URL (default: http://localhost:8000)")
    parser.add_argument("--chat-llm", default="llama2:latest",
                       help="Chat LLM model (default: llama2:latest)")
    
    args = parser.parse_args()
    
    console.print(f"[bold]Starting MCP HTTP Video Analysis Chat...[/bold]")
    console.print(f"Server: {args.server}")
    console.print(f"Chat LLM: {args.chat_llm}")
    
    client = MCPHttpClient(
        server_url=args.server,
        chat_llm_model=args.chat_llm
    )
    
    try:
        await client.connect()
    except Exception as e:
        console.print(f"[red]Failed to connect: {e}[/red]")
        console.print("\n[yellow]Make sure the MCP server is running with HTTP mode:[/yellow]")
        console.print("[dim]python mcp_video_server.py --http --host 0.0.0.0 --port 8000[/dim]")
        sys.exit(1)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())