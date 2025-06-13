#!/usr/bin/env python3
"""Test the LLM integration for video processing."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.manager import StorageManager
from src.processors.video import VideoProcessor
from src.llm.ollama_client import OllamaClient
from src.utils.config import get_config
from src.utils.logging import setup_logging, get_logger


async def test_video_processing():
    """Test video processing with LLM integration."""
    # Setup
    config = get_config()
    setup_logging(config.logging.model_dump())
    logger = get_logger(__name__)
    
    logger.info("Starting LLM integration test...")
    
    # Initialize components
    storage = StorageManager()
    llm_client = OllamaClient()
    processor = VideoProcessor(storage, llm_client)
    
    # Check if Ollama is running
    try:
        models = await llm_client.list_models()
        logger.info(f"Available models: {[m['name'] for m in models]}")
        
        # Check for vision model
        vision_model = config.llm.vision_model
        if not any(m['name'] == vision_model for m in models):
            logger.warning(f"Vision model {vision_model} not found. Please run: ollama pull {vision_model}")
            return
    except Exception as e:
        logger.error(f"Cannot connect to Ollama. Make sure it's running: {e}")
        return
    
    # Process test video
    test_video = Path(config.storage.base_path) / "originals" / "sample_video.mp4"
    if not test_video.exists():
        logger.error(f"Test video not found: {test_video}")
        return
        
    try:
        # Store video
        logger.info(f"Storing video: {test_video}")
        metadata = await storage.store_video(str(test_video), location="test")
        logger.info(f"Video stored with ID: {metadata.video_id}")
        
        # Process video
        logger.info("Processing video...")
        result = await processor.process_video(metadata.video_id)
        
        logger.info(f"Processing completed!")
        logger.info(f"  - Status: {result.status.value}")
        logger.info(f"  - Frames extracted: {result.frames_extracted}")
        logger.info(f"  - Frames analyzed: {result.frames_analyzed}")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Has transcript: {result.transcript is not None}")
        
        # Show sample frame analysis
        if result.timeline:
            logger.info("\nSample frame analyses:")
            for frame in result.timeline[:3]:  # Show first 3 frames
                logger.info(f"  Frame {frame.frame_number} ({frame.timestamp:.1f}s):")
                logger.info(f"    Description: {frame.description}")
                logger.info(f"    Objects: {', '.join(frame.objects_detected) if frame.objects_detected else 'None'}")
                logger.info(f"    Confidence: {frame.confidence:.2f}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
    finally:
        processor.cleanup()


if __name__ == "__main__":
    asyncio.run(test_video_processing())