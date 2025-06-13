#!/usr/bin/env python3
"""Enhanced interactive chat interface for video analysis with smarter NLP."""
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import re

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


class QueryParser:
    """Smart query parser for natural language understanding."""
    
    # Expanded location patterns
    LOCATION_PATTERNS = {
        'shed': ['shed', 'sheds'],
        'driveway': ['driveway', 'drive', 'drive way'],
        'front_door': ['front door', 'front-door', 'frontdoor', 'entrance', 'door'],
        'garage': ['garage'],
        'backyard': ['backyard', 'back yard', 'back-yard'],
        'sample': ['sample', 'test'],  # For test videos
        'unknown': ['unknown', 'other']
    }
    
    # Time patterns
    TIME_PATTERNS = {
        'latest': lambda: ('last 24 hours', None),
        'recent': lambda: ('last 7 days', None),
        'new': lambda: ('last 24 hours', None),
        'today': lambda: ('today', None),
        'yesterday': lambda: ('yesterday', None),
        'this week': lambda: ('this week', None),
        'last week': lambda: ('last week', None),
        'this month': lambda: ('this month', None),
        'last month': lambda: ('last month', None),
    }
    
    @classmethod
    def extract_location(cls, query: str) -> Optional[str]:
        """Extract location from query with fuzzy matching."""
        query_lower = query.lower()
        
        # First check for explicit location patterns with word boundaries
        for location, patterns in cls.LOCATION_PATTERNS.items():
            for pattern in patterns:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(pattern) + r'\b', query_lower):
                    return location
        
        # Check if query contains "from [location]" or "at [location]" or "in [location]"
        location_prep_match = re.search(r'(?:from|at|in)\s+(?:the\s+)?(\w+)', query_lower)
        if location_prep_match:
            potential_location = location_prep_match.group(1)
            # Check if it matches any known location
            for location, patterns in cls.LOCATION_PATTERNS.items():
                if potential_location in patterns or potential_location == location:
                    return location
        
        return None
    
    @classmethod
    def extract_time_query(cls, query: str) -> Optional[str]:
        """Extract time reference from query."""
        query_lower = query.lower()
        
        # Check for "last N videos/days/hours"
        last_n_match = re.search(r'last\s+(\d+)\s+(video|videos|day|days|hour|hours|minute|minutes)', query_lower)
        if last_n_match:
            number = int(last_n_match.group(1))
            unit = last_n_match.group(2)
            
            if 'video' in unit:
                # This is a count limit, not a time query
                return None
            elif 'day' in unit:
                return f"last {number} days"
            elif 'hour' in unit:
                return f"last {number} hours"
            elif 'minute' in unit:
                return f"last {number} minutes"
        
        # Check for predefined patterns
        for pattern, time_func in cls.TIME_PATTERNS.items():
            if pattern in query_lower:
                time_query, _ = time_func()
                return time_query
        
        # Check for specific dates
        date_match = re.search(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?', query)
        if date_match:
            # Parse as MM/DD or MM/DD/YYYY
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            year = date_match.group(3)
            if year:
                year = int(year)
                if year < 100:
                    year += 2000
            else:
                year = datetime.now().year
            
            try:
                target_date = datetime(year, month, day)
                return target_date.strftime("%Y-%m-%d")
            except:
                pass
        
        return None
    
    @classmethod
    def extract_limit(cls, query: str) -> Optional[int]:
        """Extract result limit from query."""
        query_lower = query.lower()
        
        # Look for "last N videos" or "latest N" or "top N"
        limit_match = re.search(r'(?:last|latest|top|first)\s+(\d+)\s*(?:video|videos)?', query_lower)
        if limit_match:
            return int(limit_match.group(1))
        
        # Look for "N videos"
        count_match = re.search(r'(\d+)\s+(?:video|videos)', query_lower)
        if count_match:
            return int(count_match.group(1))
        
        return None
    
    @classmethod
    def extract_content_filter(cls, query: str) -> Optional[str]:
        """Extract content/object filter from query."""
        query_lower = query.lower()
        
        # Common objects/activities to search for
        content_keywords = [
            'car', 'cars', 'vehicle', 'truck',
            'person', 'people', 'someone',
            'delivery', 'package', 'mail',
            'motion', 'movement', 'activity',
            'animal', 'cat', 'dog', 'bird'
        ]
        
        for keyword in content_keywords:
            if keyword in query_lower:
                # Normalize some keywords
                if keyword in ['cars', 'vehicle', 'truck']:
                    return 'car'
                elif keyword in ['people', 'someone']:
                    return 'person'
                elif keyword == 'mail':
                    return 'delivery'
                return keyword
        
        # Check for "with [content]" or "containing [content]"
        content_match = re.search(r'(?:with|containing|showing)\s+(\w+)', query_lower)
        if content_match:
            return content_match.group(1)
        
        return None
    
    @classmethod
    def parse_query(cls, query: str) -> Dict[str, Any]:
        """Parse query and extract all components."""
        return {
            'location': cls.extract_location(query),
            'time_query': cls.extract_time_query(query),
            'limit': cls.extract_limit(query),
            'content': cls.extract_content_filter(query),
            'original_query': query
        }


class ChatSession:
    """Manages chat context and history."""
    
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.current_video_context: Optional[str] = None
        self.last_query_results: List[Any] = []
        self.last_parsed_query: Optional[Dict[str, Any]] = None
    
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
    """Enhanced interactive chat interface for video analysis."""
    
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
# 🎬 Video Analysis Chat Interface v2

Welcome! I can help you analyze and query your video collection using natural language.

## Example Queries:
- "Show me the latest videos"
- "List the last 10 videos from the shed"
- "What happened at the driveway yesterday?"
- "Any deliveries today?"
- "Show recent videos with cars"
- "List videos from 12/5"

Type **help** for more commands, **exit** to quit.
        """
        console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))
    
    def display_help(self):
        """Display help information."""
        help_text = """
## 📚 Natural Language Queries

### Time-based Queries
- "latest videos" / "recent videos" / "new videos"
- "last 10 videos" / "last 5 videos from [location]"
- "videos from today" / "yesterday's videos"
- "this week" / "last week" / "last month"
- "last 3 days" / "last 24 hours"
- "videos from 12/5" / "December 5th"

### Location Queries
- "videos from the shed"
- "what happened at the driveway?"
- "show me front door activity"
- "any motion in the garage?"

### Content Queries
- "videos with cars"
- "any deliveries?"
- "show videos with people"
- "find motion or activity"

### Combined Queries
- "last 5 videos from the shed with motion"
- "deliveries at the front door today"
- "cars in the driveway yesterday"
- "recent activity at the garage"

### Video Commands
- "process /path/to/video.mp4 [at location]"
- "tell me about vid_abc123"
- "summarize the last video"

### System Commands
- **status** - Check system status
- **clear** - Clear screen
- **history** - Show chat history
- **help** - Show this help
- **exit/quit** - Exit chat
        """
        console.print(Panel(Markdown(help_text), title="Help", border_style="green"))
    
    async def process_smart_query(self, query: str) -> str:
        """Process queries with smart natural language understanding."""
        # Parse the query
        parsed = QueryParser.parse_query(query)
        self.session.last_parsed_query = parsed
        
        # If asking about "the last video" or "that video", check context
        if re.search(r'\b(the last|that|previous)\s+video\b', query.lower()):
            if self.session.last_query_results:
                video = self.session.last_query_results[0]
                return await self.show_video_details(video.video_id)
        
        # Determine if this is primarily a listing query
        is_listing_query = any(word in query.lower() for word in [
            'list', 'show', 'what', 'latest', 'recent', 'last', 'videos'
        ])
        
        # Query videos based on parsed parameters
        limit = parsed['limit'] or (10 if is_listing_query else 20)
        
        # If no specific time but asking for "latest" or "recent"
        if not parsed['time_query'] and any(word in query.lower() for word in ['latest', 'recent', 'new']):
            parsed['time_query'] = 'last 7 days'
        
        return await self.query_videos_smart(
            location=parsed['location'],
            time_query=parsed['time_query'],
            content=parsed['content'],
            limit=limit,
            original_query=query
        )
    
    async def query_videos_smart(self, location: Optional[str], 
                                time_query: Optional[str], 
                                content: Optional[str],
                                limit: int,
                                original_query: str) -> str:
        """Smart video query with natural language response."""
        start_time = None
        end_time = None
        
        if time_query:
            try:
                # Handle special time queries
                if time_query.startswith('last') and 'hour' in time_query:
                    # Extract hours
                    hours_match = re.search(r'last (\d+) hour', time_query)
                    if hours_match:
                        hours = int(hours_match.group(1))
                        start_time = datetime.now() - timedelta(hours=hours)
                        end_time = datetime.now()
                    else:
                        start_time, end_time = DateParser.parse_date_query(time_query)
                else:
                    start_time, end_time = DateParser.parse_date_query(time_query)
            except:
                return f"❌ I couldn't understand the time reference: '{time_query}'"
        
        # Get all videos if no time specified and asking for "all" or "list"
        if not time_query and not location and not content:
            if any(word in original_query.lower() for word in ['all', 'list']):
                # Default to last 30 days for "all videos" queries
                start_time = datetime.now() - timedelta(days=30)
                end_time = datetime.now()
        
        results = self.storage.query_videos_by_location_and_time(
            location=location,
            start_time=start_time,
            end_time=end_time,
            content_query=content,
            limit=limit
        )
        
        self.session.last_query_results = results
        
        if not results:
            # Build natural response
            response_parts = ["I couldn't find any videos"]
            if location:
                response_parts.append(f"from {location}")
            if time_query:
                response_parts.append(time_query)
            if content:
                response_parts.append(f"containing {content}")
            
            return " ".join(response_parts) + "."
        
        # Build natural response
        response = self._build_natural_response(results, location, time_query, content, original_query)
        
        # Create a table for results
        table = Table(box=box.SIMPLE)
        table.add_column("Time", style="yellow", width=20)
        table.add_column("Location", style="green", width=15)
        table.add_column("Video ID", style="cyan", width=20)
        table.add_column("Summary", style="white", width=50)
        
        for video in results[:limit]:
            # Get processing result for description
            processing_result = self.storage.get_processing_result(video.video_id)
            summary = "No analysis available"
            
            if processing_result and processing_result.timeline:
                # Get first few frame descriptions
                descriptions = []
                for frame in processing_result.timeline[:2]:
                    if frame.description:
                        descriptions.append(frame.description)
                if descriptions:
                    summary = "; ".join(descriptions)[:80] + "..."
            elif processing_result and processing_result.transcript:
                summary = processing_result.transcript[:80] + "..."
            
            # Format time nicely
            time_str = self._format_time_relative(video.recording_timestamp)
            
            table.add_row(
                time_str,
                video.location,
                video.video_id,
                summary
            )
        
        console.print(response)
        console.print(table)
        
        if len(results) > limit:
            console.print(f"\n[dim](Showing {limit} of {len(results)} videos)[/dim]")
        
        # Add helpful suggestions
        if results:
            console.print("\n[dim]💡 You can ask about specific videos, e.g., 'tell me about the first one' or 'vid_abc123'[/dim]")
        
        return ""
    
    def _build_natural_response(self, results: List[Any], location: Optional[str], 
                               time_query: Optional[str], content: Optional[str],
                               original_query: str) -> str:
        """Build a natural language response based on query and results."""
        count = len(results)
        
        # Handle specific query patterns
        if 'latest' in original_query.lower() or 'recent' in original_query.lower():
            if count == 1:
                return f"Here's the most recent video:"
            else:
                return f"Here are the {min(count, 10)} most recent videos:"
        
        if 'last' in original_query.lower() and re.search(r'last \d+', original_query.lower()):
            return f"Found {count} video{'s' if count != 1 else ''}:"
        
        # Build general response
        response_parts = [f"Found {count} video{'s' if count != 1 else ''}"]
        
        if location:
            response_parts.append(f"from {location}")
        
        if time_query:
            if time_query == "today":
                response_parts.append("from today")
            elif time_query == "yesterday":
                response_parts.append("from yesterday")
            else:
                response_parts.append(time_query)
        
        if content:
            response_parts.append(f"containing {content}")
        
        return " ".join(response_parts) + ":"
    
    def _format_time_relative(self, timestamp: datetime) -> str:
        """Format timestamp as relative time with absolute time."""
        now = datetime.now()
        diff = now - timestamp
        
        # Time of day
        time_str = timestamp.strftime("%I:%M %p")
        
        if diff.days == 0:
            if diff.seconds < 3600:
                relative = f"{diff.seconds // 60}m ago"
            else:
                relative = f"{diff.seconds // 3600}h ago"
            return f"{relative} ({time_str})"
        elif diff.days == 1:
            return f"Yesterday {time_str}"
        elif diff.days < 7:
            return f"{timestamp.strftime('%A')} {time_str}"
        else:
            return timestamp.strftime("%b %d, %I:%M %p")
    
    async def show_video_details(self, video_id: str) -> str:
        """Show detailed information about a video."""
        metadata = self.storage.get_video_metadata(video_id)
        if not metadata:
            # Try to find by partial ID
            if len(video_id) >= 4:
                # Search for videos with matching partial ID
                with self.storage._get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT video_id FROM videos WHERE video_id LIKE ? LIMIT 1",
                        (f"%{video_id}%",)
                    )
                    row = cursor.fetchone()
                    if row:
                        video_id = row['video_id']
                        metadata = self.storage.get_video_metadata(video_id)
            
            if not metadata:
                return f"❌ I couldn't find a video with ID: {video_id}"
        
        self.session.current_video_context = video_id
        
        # Get processing result
        processing_result = self.storage.get_processing_result(video_id)
        
        # Build detailed view
        info = Panel.fit(
            f"[bold]Video ID:[/bold] {video_id}\n"
            f"[bold]Location:[/bold] {metadata.location}\n"
            f"[bold]Recorded:[/bold] {self._format_time_relative(metadata.recording_timestamp)} ({metadata.recording_timestamp.strftime('%Y-%m-%d %H:%M:%S')})\n"
            f"[bold]Duration:[/bold] {metadata.duration:.1f} seconds\n"
            f"[bold]Resolution:[/bold] {metadata.width}x{metadata.height}\n"
            f"[bold]File:[/bold] {metadata.filename}\n"
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
                console.print(f"\n[bold]Audio Transcript:[/bold]")
                console.print(f"  {processing_result.transcript[:300]}...")
                
            # Show detected objects
            if processing_result.timeline:
                all_objects = set()
                for frame in processing_result.timeline:
                    all_objects.update(frame.objects_detected)
                if all_objects:
                    console.print(f"\n[bold]Detected Objects:[/bold] {', '.join(sorted(all_objects))}")
        
        return ""
    
    async def process_video_command(self, command: str) -> str:
        """Process a video file."""
        parts = command.split()
        if len(parts) < 2:
            return "❌ Usage: process /path/to/video.mp4 [at location]"
        
        video_path = parts[1]
        location = None
        
        # Check for location
        if 'at' in command.lower():
            at_index = parts.index('at') if 'at' in parts else -1
            if at_index > 0 and at_index + 1 < len(parts):
                location = parts[at_index + 1]
        
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
                    response += f"  Time: {self._format_time_relative(metadata.recording_timestamp)}\n"
                    response += f"  Frames analyzed: {result.frames_analyzed}\n"
                    if result.transcript:
                        response += f"  Audio transcribed: Yes\n"
                    response += f"\n💡 You can now ask questions about this video!"
                    return response
                else:
                    return f"❌ Processing failed: {result.error_message}"
                    
            except Exception as e:
                return f"❌ Error processing video: {str(e)}"
    
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
📊 System Status:
- Total videos: {stats['total_videos']}
- Processed videos: {stats['processed_videos']}
- Storage used: {stats['total_size_gb']:.2f} GB
- Ollama available: {'✅ Yes' if ollama_status else '❌ No'}
- Video locations: {', '.join(self._get_all_locations())}
            """
            return status_text.strip()
        
        # Video-specific commands
        if command.startswith('process '):
            return await self.process_video_command(user_input)
        
        # Direct video ID query
        video_id_match = re.search(r'vid_[a-f0-9]{4,12}', command)
        if video_id_match:
            return await self.show_video_details(video_id_match.group())
        
        # Smart natural language processing
        return await self.process_smart_query(user_input)
    
    def _get_all_locations(self) -> List[str]:
        """Get all unique locations from database."""
        with self.storage._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT location FROM videos ORDER BY location")
            return [row['location'] for row in cursor.fetchall()]
    
    async def run(self):
        """Run the interactive chat session."""
        self.display_welcome()
        
        # Show initial status
        stats = self.storage.get_storage_stats()
        if stats['total_videos'] > 0:
            console.print(f"\n📊 You have {stats['total_videos']} videos in your collection.")
            locations = self._get_all_locations()
            if locations:
                console.print(f"📍 Locations: {', '.join(locations)}")
        
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