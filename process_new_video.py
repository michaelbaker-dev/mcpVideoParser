#!/usr/bin/env python3
"""Simple script to process a video through the MCP server."""
import asyncio
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.manager import StorageManager
from src.processors.video import VideoProcessor
from src.llm.ollama_client import OllamaClient


async def process_video(video_path: str, location: str = None):
    """Process a single video file."""
    print(f"Processing video: {video_path}")
    
    # Initialize components
    storage = StorageManager()
    llm_client = OllamaClient()
    processor = VideoProcessor(storage, llm_client)
    
    try:
        # Store video in database
        metadata = await storage.store_video(video_path, location=location)
        print(f"Stored video with ID: {metadata.video_id}")
        
        # Process video
        result = await processor.process_video(metadata.video_id)
        
        print(f"\nProcessing complete!")
        print(f"- Status: {result.status.value}")
        print(f"- Frames extracted: {result.frames_extracted}")
        print(f"- Frames analyzed: {result.frames_analyzed}")
        print(f"- Processing time: {result.processing_time:.2f}s")
        
        if result.transcript:
            print(f"\nTranscript preview: {result.transcript[:200]}...")
            
        return metadata.video_id
        
    finally:
        await llm_client.close()
        processor.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Process a video file")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--location", help="Location name (e.g., 'sample', 'shed')")
    
    args = parser.parse_args()
    
    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Run processing
    video_id = asyncio.run(process_video(args.video_path, args.location))
    print(f"\nVideo processed successfully! ID: {video_id}")


if __name__ == "__main__":
    main()