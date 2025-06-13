"""Video processing module for frame extraction and analysis."""
import asyncio
import subprocess
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image
import ffmpeg
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils.config import get_config
from ..utils.logging import get_logger
from ..storage.schemas import (
    VideoMetadata, ProcessingResult, FrameAnalysis,
    ProcessingStatus
)
from ..storage.manager import StorageManager
from ..llm.ollama_client import OllamaClient


class VideoProcessor:
    """Handles video processing operations."""
    
    def __init__(self, storage_manager: StorageManager, llm_client: Optional[OllamaClient] = None):
        self.storage = storage_manager
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.performance.thread_count
        )
        self.llm_client = llm_client
    
    async def process_video(self, video_id: str) -> ProcessingResult:
        """Process a video file completely."""
        start_time = time.time()
        
        try:
            # Update status
            self.storage.update_video_status(video_id, ProcessingStatus.PROCESSING)
            
            # Get video metadata
            metadata = self.storage.get_video_metadata(video_id)
            if not metadata:
                raise ValueError(f"Video {video_id} not found")
            
            # Get video file path
            video_path = self._get_video_path(video_id)
            
            # Update metadata with actual video info
            await self._update_video_metadata(video_id, video_path)
            
            # Create processing directory
            process_dir = self.storage.base_path / "processed" / video_id
            process_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            self.logger.info(f"Extracting frames for video {video_id}")
            frame_paths = await self._extract_frames(video_path, process_dir)
            
            # Extract audio and transcribe
            transcript = None
            if self.config.processing.audio.enable_timestamps:
                self.logger.info(f"Extracting audio for video {video_id}")
                transcript = await self._extract_and_transcribe_audio(
                    video_path, process_dir
                )
                if transcript:
                    self.storage.store_transcript(video_id, transcript)
            
            # Analyze frames with LLM
            frame_analyses = await self._analyze_frames(
                frame_paths, video_id
            )
            
            # Store frame analyses
            if frame_analyses:
                self.storage.store_frame_analysis(video_id, frame_analyses)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update status with processing time
            self.storage.update_video_status(
                video_id, 
                ProcessingStatus.COMPLETED,
                processing_time=processing_time
            )
            
            return ProcessingResult(
                video_id=video_id,
                status=ProcessingStatus.COMPLETED,
                frames_extracted=len(frame_paths),
                frames_analyzed=len(frame_analyses),
                transcript=transcript,
                timeline=frame_analyses,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_id}: {e}")
            self.storage.update_video_status(
                video_id, ProcessingStatus.FAILED, str(e)
            )
            raise
    
    def _get_video_path(self, video_id: str) -> Path:
        """Get the path to the original video file."""
        # Get metadata to find location and date
        metadata = self.storage.get_video_metadata(video_id)
        if not metadata:
            raise FileNotFoundError(f"Video metadata not found for {video_id}")
        
        # Build path based on location and date
        from ..utils.date_parser import DateParser
        year, month, day = DateParser.get_date_path_components(metadata.recording_timestamp)
        
        location_dir = self.storage.base_path / "locations" / metadata.location / year / month / day
        
        # Look for the video file
        if location_dir.exists():
            for video_file in location_dir.glob(f"{video_id}*"):
                if video_file.is_file():
                    return video_file
        
        raise FileNotFoundError(f"Video file not found for {video_id}")
    
    async def _update_video_metadata(self, video_id: str, video_path: Path):
        """Update video metadata with actual information from file."""
        try:
            # Get video info using ffprobe
            probe = ffmpeg.probe(str(video_path))
            
            # Extract video stream info
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if video_stream:
                # Update database with actual values
                duration = float(probe['format'].get('duration', 0))
                fps = eval(video_stream.get('r_frame_rate', '0/1'))
                if isinstance(fps, (int, float)):
                    fps = float(fps)
                else:
                    fps = 30.0  # Default
                
                # This is a simplified update - in real implementation,
                # we'd update the database directly
                self.logger.info(
                    f"Video {video_id}: duration={duration:.1f}s, "
                    f"fps={fps}, {video_stream['width']}x{video_stream['height']}"
                )
                
        except Exception as e:
            self.logger.warning(f"Could not probe video {video_id}: {e}")
    
    async def _extract_frames(self, video_path: Path, output_dir: Path) -> List[Path]:
        """Extract frames from video at specified intervals."""
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_rate = self.config.processing.frame_sample_rate
            
            # Calculate which frames to extract
            if self.config.processing.enable_scene_detection:
                frame_indices = await self._detect_scene_changes(
                    video_path, total_frames
                )
            else:
                # Simple interval-based extraction
                frame_indices = list(range(0, total_frames, sample_rate))
            
            # Limit number of frames
            max_frames = self.config.processing.max_frames_per_video
            if len(frame_indices) > max_frames:
                # Evenly sample to get max_frames
                step = len(frame_indices) // max_frames
                frame_indices = frame_indices[::step][:max_frames]
            
            self.logger.info(
                f"Extracting {len(frame_indices)} frames from "
                f"{total_frames} total frames"
            )
            
            # Extract frames
            for i, frame_idx in enumerate(tqdm(frame_indices, desc="Extracting frames")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame
                    timestamp = frame_idx / fps if fps > 0 else 0
                    frame_filename = f"frame_{i:05d}_t{timestamp:.1f}s.jpg"
                    frame_path = frames_dir / frame_filename
                    
                    # Resize if needed
                    height, width = frame.shape[:2]
                    max_dimension = 1920  # Max width/height
                    
                    if width > max_dimension or height > max_dimension:
                        scale = max_dimension / max(width, height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Save with specified quality
                    cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.processing.frame_quality]
                    )
                    
                    frame_paths.append(frame_path)
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
            raise
        
        return frame_paths
    
    async def _detect_scene_changes(self, video_path: Path, total_frames: int) -> List[int]:
        """Detect scene changes in video for intelligent frame extraction."""
        # Simplified scene detection - in production, use more sophisticated methods
        # like comparing histograms or using specialized scene detection libraries
        
        scene_frames = [0]  # Always include first frame
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # Sample frames for scene detection
            sample_interval = max(1, total_frames // 500)  # Check ~500 points
            
            prev_hist = None
            
            for frame_idx in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Calculate histogram
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_hist is not None:
                    # Compare histograms
                    correlation = cv2.compareHist(
                        hist, prev_hist, cv2.HISTCMP_CORREL
                    )
                    
                    # If correlation is low, it's likely a scene change
                    if correlation < (1.0 - self.config.processing.scene_threshold):
                        scene_frames.append(frame_idx)
                
                prev_hist = hist
            
            cap.release()
            
            # Always include last frame
            if scene_frames[-1] != total_frames - 1:
                scene_frames.append(total_frames - 1)
            
        except Exception as e:
            self.logger.warning(f"Scene detection failed, using interval: {e}")
            # Fall back to interval-based extraction
            return list(range(0, total_frames, self.config.processing.frame_sample_rate))
        
        return scene_frames
    
    async def _extract_and_transcribe_audio(
        self, video_path: Path, output_dir: Path
    ) -> Optional[str]:
        """Extract audio from video and transcribe it."""
        audio_path = output_dir / "audio.wav"
        
        try:
            # Extract audio using ffmpeg
            self.logger.info("Extracting audio...")
            
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                str(audio_path),
                '-y',  # Overwrite
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"FFmpeg error: {stderr.decode()}")
                return None
            
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                self.logger.warning("No audio track found in video")
                return None
            
            # Transcribe using whisper
            self.logger.info("Transcribing audio...")
            
            # Import whisper here to avoid loading model if not needed
            import whisper
            
            model = whisper.load_model(self.config.processing.audio.model)
            result = model.transcribe(
                str(audio_path),
                language=self.config.processing.audio.language
            )
            
            # Clean up audio file
            audio_path.unlink()
            
            return result.get('text', '').strip()
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return None
    
    async def _analyze_frames(
        self, frame_paths: List[Path], video_id: str
    ) -> List[FrameAnalysis]:
        """Analyze frames using LLM vision model."""
        # If no LLM client available, create one
        if self.llm_client is None:
            self.llm_client = OllamaClient()
            
        # Check if the vision model is available
        try:
            models = await self.llm_client.list_models()
            vision_model = self.config.llm.vision_model
            model_available = any(m['name'] == vision_model for m in models)
            
            if not model_available:
                self.logger.warning(f"Vision model {vision_model} not available. Attempting to pull...")
                success = await self.llm_client.pull_model(vision_model)
                if not success:
                    self.logger.error(f"Failed to pull vision model {vision_model}")
                    return self._create_fallback_analyses(frame_paths)
        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
            return self._create_fallback_analyses(frame_paths)
            
        # Analyze frames using batch processing
        try:
            self.logger.info(f"Analyzing {len(frame_paths)} frames using {vision_model}")
            analyses = await self.llm_client.batch_analyze_frames(
                frame_paths,
                batch_size=self.config.performance.frame_batch_size
            )
            
            # Update frame paths to be relative to storage base
            for analysis in analyses:
                frame_path = Path(analysis.frame_path)
                if frame_path.is_absolute():
                    analysis.frame_path = str(frame_path.relative_to(self.storage.base_path))
                    
            return analyses
            
        except Exception as e:
            self.logger.error(f"Error analyzing frames: {e}")
            return self._create_fallback_analyses(frame_paths)
    
    def _create_fallback_analyses(self, frame_paths: List[Path]) -> List[FrameAnalysis]:
        """Create fallback analyses when LLM is not available."""
        self.logger.warning("Creating fallback frame analyses")
        analyses = []
        
        for i, frame_path in enumerate(frame_paths):
            # Extract timestamp from filename
            timestamp = 0.0
            if '_t' in frame_path.stem:
                try:
                    timestamp_str = frame_path.stem.split('_t')[1].rstrip('s')
                    timestamp = float(timestamp_str)
                except:
                    pass
            
            analysis = FrameAnalysis(
                frame_number=i,
                timestamp=timestamp,
                frame_path=str(frame_path.relative_to(self.storage.base_path)),
                description=f"Frame {i} at {timestamp:.1f}s (analysis unavailable)",
                objects_detected=[],
                confidence=0.0
            )
            analyses.append(analysis)
        
        return analyses
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)