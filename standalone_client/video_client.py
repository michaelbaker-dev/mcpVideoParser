#!/usr/bin/env python3
"""Standalone client for video analysis using Ollama."""
import asyncio
import click
import sys
from pathlib import Path
from typing import Optional
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.manager import StorageManager
from src.processors.video import VideoProcessor
from src.llm.ollama_client import OllamaClient
from src.utils.config import get_config


console = Console()


class VideoAnalysisClient:
    """Standalone client for video analysis."""
    
    def __init__(self):
        self.storage = StorageManager()
        self.llm_client = None
        self.processor = None  # Will be initialized with LLM client
        self.config = get_config()
    
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
    
    async def process_video(self, video_path: str, location: Optional[str] = None, 
                          timestamp: Optional[str] = None, show_progress: bool = True) -> str:
        """Process a video file."""
        video_path = Path(video_path).absolute()
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Parse timestamp if provided
        from datetime import datetime
        from src.utils.date_parser import DateParser
        
        parsed_timestamp = None
        if timestamp:
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp)
            except:
                # Try natural language parsing
                start_time, _ = DateParser.parse_date_query(timestamp)
                parsed_timestamp = start_time
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Store video
            task = progress.add_task("Storing video...", total=None)
            metadata = await self.storage.store_video(
                str(video_path), 
                location=location,
                recording_timestamp=parsed_timestamp
            )
            progress.update(task, description=f"Video ID: {metadata.video_id}")
            
            # Process video
            progress.update(task, description="Extracting frames...")
            
            result = await self.processor.process_video(metadata.video_id)
            progress.update(task, description="✅ Processing complete!")
            
            return metadata.video_id
    
    
    async def ask_question(self, video_id: str, question: str) -> str:
        """Ask a question about a video."""
        if not self.llm_client:
            self.llm_client = OllamaClient()
        
        # Get video data
        result = self.storage.get_processing_result(video_id)
        if not result:
            raise ValueError(f"Video {video_id} not found or not processed")
        
        # Build context
        context_parts = []
        
        if result.timeline:
            context_parts.append("Visual timeline:")
            for frame in result.timeline[:30]:
                context_parts.append(f"[{frame.timestamp:.1f}s] {frame.description}")
        
        if result.transcript:
            context_parts.append("\nAudio transcript:")
            context_parts.append(result.transcript[:1500])
        
        context = "\n".join(context_parts)
        
        # Get answer
        answer = await self.llm_client.answer_video_question(question, context)
        return answer or "Unable to generate answer"
    
    async def search_videos(self, query: str, limit: int = 10):
        """Search for videos."""
        results = self.storage.search_videos(query, limit)
        return results
    
    async def list_videos(self):
        """List all videos."""
        with self.storage._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT video_id, filename, duration, status, created_at 
                FROM videos 
                ORDER BY created_at DESC
            """)
            
            videos = []
            for row in cursor.fetchall():
                videos.append({
                    'video_id': row['video_id'],
                    'filename': row['filename'],
                    'duration': row['duration'],
                    'status': row['status'],
                    'created_at': row['created_at']
                })
            
            return videos
    
    async def get_summary(self, video_id: str) -> str:
        """Get video summary."""
        if not self.llm_client:
            self.llm_client = OllamaClient()
        
        result = self.storage.get_processing_result(video_id)
        if not result:
            raise ValueError(f"Video {video_id} not found")
        
        # Get frame descriptions
        descriptions = [f.description for f in result.timeline if f.description]
        
        # Generate summary
        summary = await self.llm_client.generate_video_summary(
            descriptions[:20],
            result.transcript
        )
        
        return summary or "Unable to generate summary"
    
    def _format_time_ago(self, timestamp):
        """Format timestamp as time ago."""
        from datetime import datetime
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days == 0:
            if diff.seconds < 3600:
                return f"{diff.seconds // 60} minutes ago"
            else:
                return f"{diff.seconds // 3600} hours ago"
        elif diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        else:
            return timestamp.strftime("%B %d, %Y")


# CLI Commands
@click.group()
def cli():
    """Video Analysis CLI - Analyze videos using local AI models."""
    pass


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--location', help='Location name (e.g., shed, driveway)')
@click.option('--timestamp', help='Recording timestamp (ISO format or natural language)')
@click.option('--no-progress', is_flag=True, help='Disable progress display')
def process(video_path, location, timestamp, no_progress):
    """Process a video file for analysis."""
    async def run():
        async with VideoAnalysisClient() as client:
            try:
                video_id = await client.process_video(
                    video_path, 
                    location=location,
                    timestamp=timestamp,
                    show_progress=not no_progress
                )
                
                console.print(Panel(
                    f"✅ Video processed successfully!\n\n"
                    f"Video ID: [bold cyan]{video_id}[/bold cyan]\n\n"
                    f"You can now ask questions about this video using:\n"
                    f"[dim]video_client ask {video_id} \"Your question here\"[/dim]",
                    title="Success",
                    border_style="green"
                ))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command()
@click.argument('video_id')
@click.argument('question')
def ask(video_id, question):
    """Ask a question about a processed video."""
    async def run():
        async with VideoAnalysisClient() as client:
            try:
                with console.status("Thinking..."):
                    answer = await client.ask_question(video_id, question)
                
                console.print(Panel(
                    Markdown(f"**Question:** {question}\n\n**Answer:** {answer}"),
                    title=f"Video {video_id}",
                    border_style="blue"
                ))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command()
@click.argument('query')
@click.option('--limit', default=10, help='Maximum results')
def search(query, limit):
    """Search for videos by content."""
    async def run():
        async with VideoAnalysisClient() as client:
            try:
                results = await client.search_videos(query, limit)
                
                if not results:
                    console.print(f"No videos found matching '{query}'")
                    return
                
                table = Table(title=f"Search Results for '{query}'")
                table.add_column("Video ID", style="cyan")
                table.add_column("Filename", style="green")
                table.add_column("Matches", style="yellow")
                
                for video_id, filename, matches in results:
                    table.add_row(video_id, filename, str(matches))
                
                console.print(table)
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command()
def list():
    """List all videos in the system."""
    async def run():
        async with VideoAnalysisClient() as client:
            try:
                videos = await client.list_videos()
                
                if not videos:
                    console.print("No videos found")
                    return
                
                table = Table(title="Videos")
                table.add_column("Video ID", style="cyan")
                table.add_column("Filename", style="green")
                table.add_column("Duration", style="yellow")
                table.add_column("Status", style="magenta")
                table.add_column("Created", style="blue")
                
                for video in videos:
                    duration = f"{video['duration']:.1f}s" if video['duration'] else "N/A"
                    status_color = {
                        'completed': 'green',
                        'processing': 'yellow',
                        'failed': 'red',
                        'pending': 'dim'
                    }.get(video['status'], 'white')
                    
                    table.add_row(
                        video['video_id'],
                        video['filename'],
                        duration,
                        f"[{status_color}]{video['status']}[/{status_color}]",
                        video['created_at'][:19]  # Trim microseconds
                    )
                
                console.print(table)
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command()
@click.argument('video_id')
def summary(video_id):
    """Get a summary of a video."""
    async def run():
        async with VideoAnalysisClient() as client:
            try:
                with console.status("Generating summary..."):
                    summary_text = await client.get_summary(video_id)
                
                console.print(Panel(
                    Markdown(f"**Video Summary**\n\n{summary_text}"),
                    title=f"Video {video_id}",
                    border_style="green"
                ))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command()
@click.option('--location', help='Location to query (e.g., shed, driveway)')
@click.option('--time', 'time_query', help='Time query (e.g., today, yesterday, last week)')
@click.option('--content', help='Search for specific content (e.g., car, person)')
@click.option('--limit', default=20, help='Maximum results')
def query(location, time_query, content, limit):
    """Query videos by location and time.
    
    Examples:
        video_client query --location shed --time today
        video_client query --location driveway --time "last week" --content car
        video_client query --time yesterday  # All locations
    """
    async def run():
        async with VideoAnalysisClient() as client:
            try:
                from src.utils.date_parser import DateParser
                
                # Build query description
                query_parts = []
                if location:
                    query_parts.append(f"at [yellow]{location}[/yellow]")
                if time_query:
                    query_parts.append(f"[cyan]{time_query}[/cyan]")
                if content:
                    query_parts.append(f"containing [green]{content}[/green]")
                
                query_desc = " ".join(query_parts) if query_parts else "all videos"
                
                with console.status(f"Searching for videos {query_desc}..."):
                    # Query using storage directly
                    start_time = None
                    end_time = None
                    if time_query:
                        start_time, end_time = DateParser.parse_date_query(time_query)
                    
                    videos = client.storage.query_videos_by_location_and_time(
                        location=location,
                        start_time=start_time,
                        end_time=end_time,
                        content_query=content,
                        limit=limit
                    )
                
                if not videos:
                    console.print(f"No videos found {query_desc}")
                    return
                
                # Create table
                table = Table(title=f"Videos {query_desc}")
                table.add_column("Time", style="cyan")
                table.add_column("Location", style="yellow")
                table.add_column("Video ID", style="dim")
                table.add_column("Duration", style="green")
                table.add_column("Summary", style="white", max_width=50)
                
                for video in videos:
                    # Get summary
                    result = client.storage.get_processing_result(video.video_id)
                    summary = "Not analyzed"
                    if result and result.timeline:
                        summaries = []
                        for frame in result.timeline[:2]:
                            if frame.description:
                                summaries.append(frame.description)
                        if summaries:
                            summary = "; ".join(summaries)[:50] + "..."
                    
                    # Format time
                    time_str = video.recording_timestamp.strftime("%m/%d %I:%M %p")
                    from datetime import datetime
                    time_ago = client._format_time_ago(video.recording_timestamp)
                    
                    table.add_row(
                        f"{time_str}\n[dim]{time_ago}[/dim]",
                        video.location,
                        video.video_id[:8] + "...",
                        f"{video.duration:.1f}s" if video.duration else "N/A",
                        summary
                    )
                
                console.print(table)
                console.print(f"\n[dim]Found {len(videos)} videos[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run())


@cli.command()
@click.option('--chat-llm', default='llama2:latest', help='Ollama model for chat interactions')
@click.option('--server', default='http://localhost:8000', help='MCP server URL for HTTP mode')
@click.option('--stdio', is_flag=True, help='Use stdio mode instead of HTTP (legacy)')
def chat(chat_llm, server, stdio):
    """Start interactive chat interface."""
    console.print("[bold blue]Starting MCP Video Analysis Chat...[/bold blue]")
    console.print(f"[dim]Chat LLM: {chat_llm}[/dim]")
    
    # Import and run the chat interface
    import subprocess
    import os
    
    if stdio:
        # Use legacy stdio client
        console.print(f"[dim]Mode: stdio (legacy)[/dim]\n")
        chat_script = Path(__file__).parent / "mcp_chat_client.py"
        subprocess.run([sys.executable, str(chat_script), '--chat-llm', chat_llm])
    else:
        # Use HTTP client (default)
        console.print(f"[dim]Server: {server}[/dim]")
        console.print(f"[dim]Mode: HTTP (recommended)[/dim]\n")
        
        # Check if server is reachable
        try:
            import httpx
            import asyncio
            
            async def check_server():
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"{server}/health")
                        return response.status_code == 200
                except:
                    return False
            
            server_available = asyncio.run(check_server())
            
            if not server_available:
                console.print(f"[yellow]Warning: Server at {server} is not responding[/yellow]")
                console.print(f"[yellow]Start the server with: python server.py --http --host 0.0.0.0 --port 8000[/yellow]")
                console.print(f"[yellow]Or use --stdio flag for legacy mode[/yellow]")
                return
                
        except ImportError:
            console.print("[red]httpx not available for HTTP mode[/red]")
            return
        
        # Use HTTP client
        chat_script = Path(__file__).parent / "mcp_http_client.py"
        subprocess.run([
            sys.executable, str(chat_script), 
            '--server', server, 
            '--chat-llm', chat_llm
        ])


@cli.command()
def status():
    """Check system status."""
    async def run():
        async with VideoAnalysisClient() as client:
            try:
                # Check Ollama
                ollama_status = "✅ Running" if await client.llm_client.is_available() else "❌ Not running"
                
                # Check models
                models = await client.llm_client.list_models()
                model_names = [m["name"] for m in models]
                
                required_models = ["llava:latest", "llama2:latest"]
                model_status = {}
                for model in required_models:
                    model_status[model] = "✅" if model in model_names else "❌"
                
                # Get storage stats
                stats = client.storage.get_storage_stats()
                
                console.print(Panel(
                    f"**Ollama Status:** {ollama_status}\n\n"
                    f"**Required Models:**\n"
                    f"  • llava:latest: {model_status.get('llava:latest', '❌')}\n"
                    f"  • llama2:latest: {model_status.get('llama2:latest', '❌')}\n\n"
                    f"**Storage:**\n"
                    f"  • Total videos: {stats['total_videos']}\n"
                    f"  • Processed: {stats['processed_videos']}\n"
                    f"  • Storage used: {stats['total_size_gb']:.2f} GB\n"
                    f"  • Base path: {client.storage.base_path}",
                    title="System Status",
                    border_style="blue"
                ))
                
                if "❌" in model_status.values():
                    console.print("\n[yellow]Missing models! Install with:[/yellow]")
                    for model, status in model_status.items():
                        if status == "❌":
                            console.print(f"  ollama pull {model}")
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run())


if __name__ == "__main__":
    cli()