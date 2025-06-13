#!/usr/bin/env python3
"""MCP-based chat client that connects to the video analysis MCP server."""
import asyncio
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import re

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt
from rich import box

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import TextContent
from src.llm.ollama_client import OllamaClient
from src.utils.config import get_config


console = Console()


class ChatLLM:
    """Separate LLM for chat interactions (can be different from video analysis LLM)."""
    
    def __init__(self, model_name: str = "llama2:latest"):
        self.model_name = model_name
        self.ollama_client = OllamaClient()
        # Override the model for chat
        self.ollama_client.text_model = model_name
    
    async def process_user_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user query and determine intent."""
        prompt = f"""You are a helpful assistant for a video analysis system. 
        Analyze this user query and extract the intent and parameters.
        
        User query: "{query}"
        
        Available locations: {context.get('locations', ['shed', 'driveway', 'front_door', 'garage', 'sample'])}
        
        Respond with JSON containing:
        - intent: one of [query_videos, process_video, ask_about_video, search_videos, get_summary, show_video, general_question]
        - location: extracted location or null
        - time_query: extracted time reference or null
        - content_query: what to search for or null
        - video_id: if asking about specific video or null
        - limit: number of results requested or null
        
        Examples:
        "show latest videos" -> {{"intent": "query_videos", "time_query": "today", "limit": 10}}
        "what happened at the shed yesterday" -> {{"intent": "query_videos", "location": "shed", "time_query": "yesterday"}}
        "tell me about vid_abc123" -> {{"intent": "show_video", "video_id": "vid_abc123"}}
        """
        
        try:
            response = await self.ollama_client.generate_text(prompt, temperature=0.1)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to simple pattern matching
                return self._fallback_parse(query)
        except Exception as e:
            console.print(f"[yellow]Chat LLM parsing failed, using fallback: {e}[/yellow]")
            return self._fallback_parse(query)
    
    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """Fallback parsing if LLM fails."""
        query_lower = query.lower()
        
        # Determine intent
        intent = "general_question"
        if any(word in query_lower for word in ['list', 'show', 'latest', 'recent', 'videos', 'what happened']):
            intent = "query_videos"
        elif 'process' in query_lower:
            intent = "process_video"
        elif re.search(r'vid_[a-f0-9]+', query_lower):
            intent = "show_video"
        elif any(word in query_lower for word in ['search', 'find']):
            intent = "search_videos"
        elif 'summary' in query_lower:
            intent = "get_summary"
        
        # Extract parameters
        result = {"intent": intent}
        
        # Location
        locations = ['shed', 'driveway', 'front_door', 'garage', 'sample']
        for loc in locations:
            if loc in query_lower:
                result['location'] = loc
                break
        
        # Time
        if 'today' in query_lower:
            result['time_query'] = 'today'
        elif 'yesterday' in query_lower:
            result['time_query'] = 'yesterday'
        elif 'last week' in query_lower:
            result['time_query'] = 'last week'
        elif 'latest' in query_lower or 'recent' in query_lower:
            result['time_query'] = 'last 7 days'
        
        # Video ID
        vid_match = re.search(r'(vid_[a-f0-9]+)', query_lower)
        if vid_match:
            result['video_id'] = vid_match.group(1)
        
        # Limit
        limit_match = re.search(r'(?:last|top|first)\s+(\d+)', query_lower)
        if limit_match:
            result['limit'] = int(limit_match.group(1))
        
        return result
    
    async def format_response(self, mcp_result: Any, query_intent: Dict[str, Any]) -> str:
        """Use LLM to format MCP results into natural language."""
        if isinstance(mcp_result, dict) and 'error' in mcp_result:
            return f"❌ {mcp_result['error']}"
        
        # For simple intents, format directly
        if query_intent['intent'] == 'query_videos' and 'results' in mcp_result:
            return self._format_video_list(mcp_result)
        
        # For complex responses, use LLM
        prompt = f"""Format this technical response into a friendly, conversational message.
        
        User intent: {query_intent}
        System response: {json.dumps(mcp_result, indent=2)[:1000]}...
        
        Guidelines:
        - Be concise and friendly
        - Highlight important information
        - Use emojis sparingly but appropriately
        - If it's a list, mention the count
        - If it's video details, summarize key points
        """
        
        try:
            response = await self.ollama_client.generate_text(prompt, temperature=0.7)
            return response
        except:
            # Fallback formatting
            if isinstance(mcp_result, dict):
                if 'message' in mcp_result:
                    return mcp_result['message']
                elif 'summary' in mcp_result:
                    return mcp_result['summary']
            return str(mcp_result)
    
    def _format_video_list(self, result: Dict[str, Any]) -> str:
        """Format video list results."""
        if not result.get('results'):
            return "No videos found matching your criteria."
        
        count = result.get('total_results', len(result['results']))
        
        # Build response
        parts = [f"Found {count} video{'s' if count != 1 else ''}"]
        
        query = result.get('query', {})
        if query.get('location'):
            parts.append(f"from {query['location']}")
        if query.get('time'):
            parts.append(query['time'])
        if query.get('content'):
            parts.append(f"with {query['content']}")
        
        return " ".join(parts) + ":"
    
    async def close(self):
        """Close LLM client."""
        await self.ollama_client.close()


class MCPChatClient:
    """Chat client that communicates with MCP video server."""
    
    def __init__(self, chat_llm_model: str = "llama2:latest", server_command: Optional[List[str]] = None):
        self.server_command = server_command
        self.session = None
        self.chat_llm = ChatLLM(chat_llm_model)
        self.running = True
        self.config = get_config()
        
        # Track context
        self.last_results = []
        self.current_video = None
        self.available_locations = []
        self.available_tools = {}
    
    async def connect(self):
        """Connect to the MCP server."""
        # Default to local server
        if not self.server_command:
            server_script = Path(__file__).parent.parent / "server.py"
            
            if not server_script.exists():
                raise FileNotFoundError(f"MCP server script not found at {server_script}")
            
            self.server_command = [sys.executable, str(server_script)]
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=self.server_command[0],
            args=self.server_command[1:] if len(self.server_command) > 1 else []
        )
        
        # Connect to the server
        async with stdio_client(server_params) as (read_stream, write_stream):
            self.session = ClientSession(read_stream, write_stream)
            
            # Initialize the session
            await self.session.initialize()
            
            console.print("[green]✅ Connected to MCP Video Analysis Server[/green]")
            
            # List available tools
            tools_response = await self.session.list_tools()
            self.available_tools = {tool.name: tool for tool in tools_response.tools}
            console.print(f"[dim]Available tools: {len(self.available_tools)}[/dim]")
            
            # Get initial stats
            try:
                stats_result = await self.call_tool("get_video_stats", {})
                if stats_result:
                    console.print(f"[dim]System ready with {stats_result.get('storage', {}).get('total_videos', 0)} videos[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not get initial stats: {e}[/yellow]")
            
            # Run the chat interface
            await self.run_chat()
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        await self.chat_llm.close()
    
    def display_welcome(self):
        """Display welcome message."""
        welcome_text = """
# 🎬 MCP-Based Video Chat Client

This chat interface connects to your MCP Video Analysis Server.

## Features:
- 🤖 Separate LLMs for chat and video analysis
- 🔌 Uses MCP protocol (same as Claude Desktop)
- 💬 Natural language understanding
- 📊 Real-time video analysis

## Example Queries:
- "Show me the latest videos"
- "What happened at the shed today?"
- "Process /path/to/video.mp4"
- "Find videos with cars"

Type **help** for more commands, **exit** to quit.
        """
        console.print(Panel(Markdown(welcome_text), title="MCP Video Chat", border_style="blue"))
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and get results."""
        try:
            # Check if tool exists
            if tool_name not in self.available_tools:
                return {"error": f"Tool '{tool_name}' not found"}
            
            # Call the tool
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            # Parse the response
            if result.content:
                # MCP tools return TextContent objects
                for content in result.content:
                    if hasattr(content, 'text'):
                        try:
                            return json.loads(content.text)
                        except json.JSONDecodeError:
                            return {"message": content.text}
            
            return {"error": "No content in tool response"}
            
        except Exception as e:
            return {"error": f"Tool call failed: {str(e)}"}
    
    async def process_query(self, user_input: str) -> str:
        """Process user query through chat LLM and MCP server."""
        # Parse query with chat LLM
        context = {
            "locations": self.available_locations or ['shed', 'driveway', 'front_door', 'garage', 'sample'],
            "last_results": len(self.last_results),
            "current_video": self.current_video
        }
        
        query_intent = await self.chat_llm.process_user_query(user_input, context)
        
        # Execute appropriate MCP tool based on intent
        result = None
        
        if query_intent['intent'] == 'query_videos':
            # Use query_location_time tool
            result = await self.call_tool('query_location_time', {
                'location': query_intent.get('location'),
                'time_query': query_intent.get('time_query'),
                'content_query': query_intent.get('content_query'),
                'limit': query_intent.get('limit', 20)
            })
            
            if result and 'results' in result:
                self.last_results = result['results']
        
        elif query_intent['intent'] == 'process_video':
            # Extract path from query
            path_match = re.search(r'([/\\].+?\.\w+)', user_input)
            if path_match:
                video_path = path_match.group(1)
                result = await self.call_tool('process_video', {
                    'video_path': video_path,
                    'location': query_intent.get('location')
                })
                if result and 'video_id' in result:
                    self.current_video = result['video_id']
        
        elif query_intent['intent'] == 'show_video':
            video_id = query_intent.get('video_id')
            if not video_id and self.last_results:
                # Check if asking about "first", "last", etc.
                if 'first' in user_input.lower():
                    video_id = self.last_results[0]['video_id']
                elif 'last' in user_input.lower():
                    video_id = self.last_results[-1]['video_id']
            
            if video_id:
                # We don't have a direct "get video details" tool, so use summary
                result = await self.call_tool('get_video_summary', {
                    'video_id': video_id,
                    'detail_level': 'detailed'
                })
                self.current_video = video_id
        
        elif query_intent['intent'] == 'search_videos':
            result = await self.call_tool('search_videos', {
                'query': query_intent.get('content_query', user_input),
                'limit': query_intent.get('limit', 10)
            })
        
        elif query_intent['intent'] == 'ask_about_video':
            if self.current_video:
                result = await self.call_tool('ask_video', {
                    'video_id': self.current_video,
                    'question': user_input
                })
        
        elif query_intent['intent'] == 'get_summary':
            video_id = query_intent.get('video_id', self.current_video)
            if video_id:
                result = await self.call_tool('get_video_summary', {
                    'video_id': video_id,
                    'detail_level': 'detailed'
                })
        
        else:
            # General question - if we have a current video context, ask about it
            if self.current_video:
                result = await self.call_tool('ask_video', {
                    'video_id': self.current_video,
                    'question': user_input
                })
            else:
                return "I'm not sure what you're asking. Try 'show latest videos' or 'help' for examples."
        
        # Format response with chat LLM
        if result:
            formatted = await self.chat_llm.format_response(result, query_intent)
            
            # If it's a video list, also show a table
            if query_intent['intent'] == 'query_videos' and 'results' in result:
                self._display_video_table(result['results'])
            
            return formatted
        else:
            return "I couldn't process that request. Please try again."
    
    def _display_video_table(self, videos: List[Dict[str, Any]]):
        """Display videos in a nice table."""
        if not videos:
            return
        
        table = Table(box=box.SIMPLE)
        table.add_column("Time", style="yellow", width=20)
        table.add_column("Location", style="green", width=15)
        table.add_column("Video ID", style="cyan", width=20)
        table.add_column("Summary", style="white", width=50)
        
        for video in videos[:10]:  # Show max 10
            time_str = self._format_time(video.get('timestamp', video.get('time_ago', 'Unknown')))
            location = video.get('location', 'unknown')
            video_id = video.get('video_id', 'unknown')
            summary = video.get('summary', 'No summary')
            if len(summary) > 80:
                summary = summary[:77] + "..."
            
            table.add_row(time_str, location, video_id, summary)
        
        console.print(table)
        
        if len(videos) > 10:
            console.print(f"[dim](Showing 10 of {len(videos)} videos)[/dim]")
    
    def _format_time(self, time_value: str) -> str:
        """Format time string nicely."""
        if 'ago' in str(time_value):
            return time_value
        
        try:
            # Try to parse ISO format
            dt = datetime.fromisoformat(time_value.replace('Z', '+00:00'))
            now = datetime.now()
            diff = now - dt
            
            if diff.days == 0:
                if diff.seconds < 3600:
                    return f"{diff.seconds // 60}m ago"
                else:
                    return f"{diff.seconds // 3600}h ago"
            elif diff.days == 1:
                return "Yesterday"
            else:
                return dt.strftime("%b %d")
        except:
            return time_value
    
    async def handle_command(self, user_input: str) -> Optional[str]:
        """Handle user commands."""
        command = user_input.strip().lower()
        
        # System commands
        if command in ['exit', 'quit', 'bye']:
            self.running = False
            return "Goodbye! 👋"
        
        if command == 'help':
            help_text = """
## 📚 Commands

### Video Queries
- "show latest videos" - List recent videos
- "videos from [location] [time]" - Query by location/time
- "find videos with [content]" - Search content

### Video Processing
- "process /path/to/video.mp4" - Process new video
- "process /path/to/video.mp4 at shed" - With location

### Video Analysis
- "tell me about vid_xxx" - Show video details
- "summarize vid_xxx" - Get video summary
- Ask questions when viewing a video

### System
- **status** - Check system status
- **clear** - Clear screen
- **help** - Show this help
- **exit** - Exit chat
            """
            console.print(Panel(Markdown(help_text), title="Help", border_style="green"))
            return None
        
        if command == 'clear':
            console.clear()
            return None
        
        if command == 'status':
            result = await self.call_tool('get_video_stats', {})
            if result and 'storage' in result:
                storage = result.get('storage', {})
                system = result.get('system', {})
                
                status_text = f"""
📊 System Status:
- Total videos: {storage.get('total_videos', 0)}
- Processed videos: {storage.get('processed_videos', 0)}
- Storage used: {storage.get('total_size_gb', 0):.2f} GB
- Ollama available: {'✅' if system.get('ollama_available') else '❌'}
- Chat LLM: {self.chat_llm.model_name}
- Video LLM: {self.config.llm.vision_model}
                """
                return status_text.strip()
            return "❌ Could not get system status"
        
        # Process as natural language query
        return await self.process_query(user_input)
    
    async def run_chat(self):
        """Run the chat interface loop."""
        self.display_welcome()
        
        # Get initial stats
        console.print("\n[dim]Checking system status...[/dim]")
        status = await self.handle_command('status')
        if status:
            console.print(status)
        
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if not user_input.strip():
                    continue
                
                # Process command
                with console.status("[bold green]Thinking..."):
                    response = await self.handle_command(user_input)
                
                # Display response
                if response:
                    console.print(f"\n[bold green]Assistant:[/bold green] {response}")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")


async def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='MCP Video Chat Client')
    parser.add_argument('--chat-llm', default='llama2:latest', 
                       help='Ollama model for chat interactions (default: llama2:latest)')
    parser.add_argument('--server-command', nargs='+',
                       help='Custom command to start MCP server (for remote servers)')
    args = parser.parse_args()
    
    try:
        client = MCPChatClient(
            chat_llm_model=args.chat_llm,
            server_command=args.server_command
        )
        await client.connect()
        await client.disconnect()
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())