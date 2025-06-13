#!/usr/bin/env python3
"""Interactive chat interface for video analysis."""
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

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

from src.storage.manager import StorageManager
from src.processors.video import VideoProcessor
from src.llm.ollama_client import OllamaClient
from src.utils.config import get_config
from src.utils.date_parser import DateParser
from src.storage.schemas import ProcessingStatus


console = Console()


class ChatSession:
    """Manages chat context and history."""
    
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.current_video_context: Optional[str] = None
        self.last_query_results: List[Any] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context(self, last_n: int = 5) -> str:
        """Get conversation context for LLM."""
        recent = self.history[-last_n:] if len(self.history) > last_n else self.history
        context_parts = []
        for msg in recent:
            context_parts.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(context_parts)


class VideoChatInterface:
    """Interactive chat interface for video analysis."""
    
    def __init__(self):
        self.storage = StorageManager()
        self.llm_client = None
        self.processor = None  # Will be initialized with LLM client
        self.config = get_config()
        self.session = ChatSession()
        self.running = True
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.llm_client = OllamaClient()
        self.processor = VideoProcessor(self.storage, llm_client=self.llm_client)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.llm_client:
            await self.llm_client.close()
        if self.processor:
            self.processor.cleanup()
    
    def display_welcome(self):
        """Display welcome message."""
        welcome_text = """
# 🎬 Video Analysis Chat Interface

Welcome! I can help you analyze and query your video collection.

## Available Commands:
- **Natural language queries**: "What happened at the shed today?"
- **Process videos**: "process /path/to/video.mp4"
- **Search content**: "find videos with cars"
- **Ask about videos**: "tell me about vid_abc123"
- **Time queries**: "show me yesterday's videos"

Type **help** for more commands, **exit** to quit.
        """
        console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))
    
    def display_help(self):
        """Display help information."""
        help_text = """
## 📚 Command Reference

### Natural Language Queries
- "What happened at [location] [time]?"
- "Show me videos from [time]"
- "Were there any [objects] at [location]?"
- "Find videos with [content]"

### Video Processing
- "process [path]" - Process a new video
- "process [path] at [location]" - Process with location

### Video Analysis
- "tell me about [video_id]"
- "summarize [video_id]"
- "what's in [video_id]?"

### Search & Browse
- "list all videos"
- "list today's videos"
- "search for [term]"
- "find [content] in [location]"

### System
- "status" - Check system status
- "clear" - Clear screen
- "history" - Show chat history
- "exit/quit" - Exit chat

### Time Expressions
- today, yesterday, tomorrow
- this week, last week
- last 3 days, last month
- December 15th, 2024-12-15
        """
        console.print(Panel(Markdown(help_text), title="Help", border_style="green"))
    
    async def process_natural_language_query(self, query: str) -> str:
        """Process natural language queries about videos."""
        query_lower = query.lower()
        
        # Determine query type and extract parameters
        location = None
        time_query = None
        content_query = None
        
        # Extract location
        locations = ['shed', 'driveway', 'front_door', 'garage', 'backyard', 'entrance']
        for loc in locations:
            if loc in query_lower:
                location = loc
                break
        
        # Extract time references
        time_keywords = ['today', 'yesterday', 'tomorrow', 'week', 'month', 'day', 'hour']
        for keyword in time_keywords:
            if keyword in query_lower:
                # Try to extract the full time phrase
                if 'last' in query_lower:
                    if 'week' in query_lower:
                        time_query = "last week"
                    elif 'month' in query_lower:
                        time_query = "last month"
                    elif 'days' in query_lower:
                        # Extract number of days
                        words = query_lower.split()
                        for i, word in enumerate(words):
                            if word == 'last' and i+2 < len(words) and words[i+2] == 'days':
                                try:
                                    num = int(words[i+1])
                                    time_query = f"last {num} days"
                                except:
                                    pass
                else:
                    time_query = keyword
                break
        
        # Extract content/object queries
        content_keywords = ['car', 'person', 'delivery', 'package', 'motion', 'movement', 'activity']
        for keyword in content_keywords:
            if keyword in query_lower:
                content_query = keyword
                break
        
        # Query videos based on extracted parameters
        if location or time_query:
            return await self.query_videos_by_location_time(location, time_query, content_query)
        elif content_query:
            return await self.search_videos(content_query)
        else:
            # Try to understand the query using LLM
            return await self.ask_general_question(query)
    
    async def query_videos_by_location_time(self, location: Optional[str], 
                                          time_query: Optional[str], 
                                          content: Optional[str]) -> str:
        """Query videos by location and time."""
        start_time = None
        end_time = None
        
        if time_query:
            try:
                start_time, end_time = DateParser.parse_date_query(time_query)
            except:
                return f"❌ Couldn't understand time query: '{time_query}'"
        
        results = self.storage.query_videos_by_location_and_time(
            location=location,
            start_time=start_time,
            end_time=end_time,
            content_query=content,
            limit=20
        )
        
        self.session.last_query_results = results
        
        if not results:
            parts = []
            if location:
                parts.append(f"at {location}")
            if time_query:
                parts.append(time_query)
            if content:
                parts.append(f"containing '{content}'")
            
            return f"No videos found {' '.join(parts)}."
        
        # Build response
        response_parts = [f"Found {len(results)} video(s)"]
        if location:
            response_parts.append(f"at {location}")
        if time_query:
            response_parts.append(time_query)
        if content:
            response_parts.append(f"containing '{content}'")
        
        response = " ".join(response_parts) + ":\n\n"
        
        # Create a table for results
        table = Table(title=None, box=box.SIMPLE)
        table.add_column("Video ID", style="cyan")
        table.add_column("Location", style="green")
        table.add_column("Time", style="yellow")
        table.add_column("Description", style="white")
        
        for video in results[:10]:  # Show max 10 results
            # Get processing result for description
            processing_result = self.storage.get_processing_result(video.video_id)
            description = "No analysis available"
            
            if processing_result and processing_result.timeline:
                # Get first few frame descriptions
                descriptions = []
                for frame in processing_result.timeline[:2]:
                    if frame.description:
                        descriptions.append(frame.description)
                if descriptions:
                    description = "; ".join(descriptions)[:60] + "..."
            
            time_str = video.recording_timestamp.strftime("%Y-%m-%d %H:%M")
            table.add_row(
                video.video_id,
                video.location,
                time_str,
                description
            )
        
        console.print(table)
        
        if len(results) > 10:
            response += f"\n(Showing first 10 of {len(results)} results)"
        
        return ""
    
    async def search_videos(self, query: str) -> str:
        """Search videos by content."""
        results = self.storage.search_videos(query, limit=10)
        
        if not results:
            return f"No videos found matching '{query}'."
        
        response = f"Found {len(results)} video(s) matching '{query}':\n\n"
        
        table = Table(title=None, box=box.SIMPLE)
        table.add_column("Video ID", style="cyan")
        table.add_column("Filename", style="green")
        table.add_column("Relevance", style="yellow")
        
        for video_id, filename, relevance in results:
            table.add_row(video_id, filename, str(relevance))
        
        console.print(table)
        return ""
    
    async def ask_general_question(self, question: str) -> str:
        """Answer general questions using context."""
        # Build context from recent queries and results
        context_parts = []
        
        # Add conversation history
        context_parts.append("Recent conversation:")
        context_parts.append(self.session.get_context())
        
        # Add last query results if any
        if self.session.last_query_results:
            context_parts.append("\nRecent videos viewed:")
            for video in self.session.last_query_results[:3]:
                context_parts.append(f"- {video.video_id}: {video.location} at {video.recording_timestamp}")
        
        # Add current video context if any
        if self.session.current_video_context:
            context_parts.append(f"\nCurrently discussing video: {self.session.current_video_context}")
        
        context = "\n".join(context_parts)
        
        # Get LLM response
        response = await self.llm_client.answer_video_question(question, context)
        return response or "I couldn't understand that question. Try asking about specific videos or locations."
    
    async def process_video_command(self, command: str) -> str:
        """Process a video file."""
        parts = command.split()
        if len(parts) < 2:
            return "❌ Usage: process /path/to/video.mp4 [at location]"
        
        video_path = parts[1]
        location = None
        
        # Check for location
        if len(parts) > 3 and parts[2].lower() == 'at':
            location = parts[3]
        
        if not Path(video_path).exists():
            return f"❌ Video file not found: {video_path}"
        
        # Process with progress indicator
        with console.status("[bold green]Processing video...") as status:
            try:
                metadata = await self.storage.store_video(video_path, location=location)
                self.session.current_video_context = metadata.video_id
                
                status.update("[bold green]Analyzing frames and audio...")
                result = await self.processor.process_video(metadata.video_id)
                
                if result.status == ProcessingStatus.COMPLETED:
                    response = f"✅ Successfully processed video!\n"
                    response += f"  Video ID: {result.video_id}\n"
                    response += f"  Location: {metadata.location}\n"
                    response += f"  Frames analyzed: {result.frames_analyzed}\n"
                    if result.transcript:
                        response += f"  Audio transcribed: Yes"
                    return response
                else:
                    return f"❌ Processing failed: {result.error_message}"
                    
            except Exception as e:
                return f"❌ Error processing video: {str(e)}"
    
    async def show_video_details(self, video_id: str) -> str:
        """Show detailed information about a video."""
        metadata = self.storage.get_video_metadata(video_id)
        if not metadata:
            return f"❌ Video not found: {video_id}"
        
        self.session.current_video_context = video_id
        
        # Get processing result
        processing_result = self.storage.get_processing_result(video_id)
        
        # Build detailed view
        info = Panel.fit(
            f"[bold]Video ID:[/bold] {video_id}\n"
            f"[bold]Location:[/bold] {metadata.location}\n"
            f"[bold]Recorded:[/bold] {metadata.recording_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"[bold]Duration:[/bold] {metadata.duration:.1f} seconds\n"
            f"[bold]Resolution:[/bold] {metadata.width}x{metadata.height}\n"
            f"[bold]Status:[/bold] {metadata.status}",
            title="Video Information",
            border_style="cyan"
        )
        console.print(info)
        
        if processing_result:
            # Show timeline summary
            if processing_result.timeline:
                console.print("\n[bold]Key Moments:[/bold]")
                for i, frame in enumerate(processing_result.timeline[:5]):
                    if frame.description:
                        console.print(f"  [{frame.timestamp:.1f}s] {frame.description}")
                
                if len(processing_result.timeline) > 5:
                    console.print(f"  ... and {len(processing_result.timeline) - 5} more frames")
            
            # Show transcript excerpt
            if processing_result.transcript:
                console.print(f"\n[bold]Transcript excerpt:[/bold]")
                console.print(f"  {processing_result.transcript[:200]}...")
        
        return ""
    
    async def handle_command(self, user_input: str) -> Optional[str]:
        """Handle user commands and queries."""
        command = user_input.strip().lower()
        
        # System commands
        if command in ['exit', 'quit', 'bye']:
            self.running = False
            return "Goodbye! 👋"
        
        if command == 'help':
            self.display_help()
            return None
        
        if command == 'clear':
            console.clear()
            return None
        
        if command == 'history':
            for msg in self.session.history[-10:]:
                role_style = "cyan" if msg['role'] == 'user' else "green"
                console.print(f"[{role_style}]{msg['role']}:[/{role_style}] {msg['content']}")
            return None
        
        if command == 'status':
            stats = self.storage.get_storage_stats()
            ollama_status = await self.llm_client.is_available()
            
            status_text = f"""
System Status:
- Total videos: {stats['total_videos']}
- Processed videos: {stats['processed_videos']}
- Storage used: {stats['total_size_gb']:.2f} GB
- Ollama available: {'✅ Yes' if ollama_status else '❌ No'}
            """
            return status_text.strip()
        
        # Video-specific commands
        if command.startswith('process '):
            return await self.process_video_command(command)
        
        if command.startswith('list'):
            if 'today' in command:
                return await self.query_videos_by_location_time(None, "today", None)
            else:
                # List recent videos
                return await self.query_videos_by_location_time(None, "last 7 days", None)
        
        # Video ID queries
        if command.startswith('vid_') or 'vid_' in command:
            # Extract video ID
            import re
            video_ids = re.findall(r'vid_[a-f0-9]{12}', command)
            if video_ids:
                return await self.show_video_details(video_ids[0])
        
        # Natural language processing
        return await self.process_natural_language_query(user_input)
    
    async def run(self):
        """Run the interactive chat session."""
        self.display_welcome()
        
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if not user_input.strip():
                    continue
                
                # Add to history
                self.session.add_message("user", user_input)
                
                # Process command
                with console.status("[bold green]Thinking..."):
                    response = await self.handle_command(user_input)
                
                # Display response
                if response:
                    console.print(f"\n[bold green]Assistant:[/bold green] {response}")
                    self.session.add_message("assistant", response)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")


async def main():
    """Main entry point."""
    try:
        async with VideoChatInterface() as chat:
            await chat.run()
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())