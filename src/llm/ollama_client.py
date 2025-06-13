"""Ollama integration for LLM interactions."""
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import httpx
import asyncio
from PIL import Image
import io

from ..utils.config import get_config
from ..utils.logging import get_logger
from ..storage.schemas import FrameAnalysis


class OllamaClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self, host: Optional[str] = None):
        self.config = get_config()
        self.host = host or self.config.llm.ollama_host
        self.logger = get_logger(__name__)
        self.timeout = self.config.llm.timeout_seconds
        
        # Create async client
        self.client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(self.timeout, connect=10.0)
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def is_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Ollama not available: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model if not already available."""
        try:
            # Check if model exists
            models = await self.list_models()
            if any(m["name"] == model_name for m in models):
                self.logger.info(f"Model {model_name} already available")
                return True
            
            # Pull model
            self.logger.info(f"Pulling model {model_name}...")
            response = await self.client.post(
                "/api/pull",
                json={"name": model_name}
            )
            response.raise_for_status()
            
            # Wait for pull to complete
            # In real implementation, stream the response to show progress
            return True
            
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def analyze_image(
        self, 
        image_path: Union[str, Path], 
        prompt: str,
        model: Optional[str] = None
    ) -> Optional[str]:
        """Analyze an image using vision model."""
        model = model or self.config.llm.vision_model
        
        try:
            # Read and encode image
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Encode image to base64
            with open(image_path, "rb") as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode()
            
            # Prepare request
            request_data = {
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
            
            # Send request
            response = await self.client.post(
                "/api/generate",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {e}")
            return None
    
    async def analyze_frame(self, frame_path: Path, frame_number: int, timestamp: float) -> FrameAnalysis:
        """Analyze a single video frame."""
        prompt = (
            "Describe what you see in this video frame. "
            "Be concise but include important details like: "
            "- Main subjects or people\n"
            "- Actions or activities\n"
            "- Objects and their positions\n"
            "- Scene setting or location\n"
            "Limit your response to 2-3 sentences."
        )
        
        description = await self.analyze_image(frame_path, prompt)
        
        if description:
            # Extract objects from description (simplified)
            objects = self._extract_objects_from_description(description)
            
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp=timestamp,
                frame_path=str(frame_path),
                description=description,
                objects_detected=objects,
                confidence=0.8  # Default confidence
            )
        else:
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp=timestamp,
                frame_path=str(frame_path),
                description="Failed to analyze frame",
                objects_detected=[],
                confidence=0.0
            )
    
    def _extract_objects_from_description(self, description: str) -> List[str]:
        """Extract object names from description (simplified)."""
        # In a real implementation, use NLP or a separate model
        # For now, look for common object words
        common_objects = [
            "person", "people", "man", "woman", "child",
            "car", "vehicle", "truck", "bus", "bicycle",
            "building", "house", "tree", "road", "street",
            "table", "chair", "computer", "phone", "book",
            "dog", "cat", "animal", "bird"
        ]
        
        description_lower = description.lower()
        found_objects = []
        
        for obj in common_objects:
            if obj in description_lower:
                found_objects.append(obj)
        
        return found_objects
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """Generate text using text model."""
        model = model or self.config.llm.text_model
        temperature = temperature or self.config.llm.temperature
        
        try:
            request_data = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            
            if system_prompt:
                request_data["system"] = system_prompt
            
            response = await self.client.post(
                "/api/generate",
                json=request_data
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return None
    
    async def answer_video_question(
        self,
        question: str,
        context: str,
        model: Optional[str] = None
    ) -> Optional[str]:
        """Answer a question about video content."""
        system_prompt = (
            "You are analyzing video content based on frame descriptions and transcripts. "
            "Answer questions based solely on the provided context. "
            "Be concise and accurate. If the information is not in the context, "
            "say so rather than making assumptions."
        )
        
        prompt = f"""Video Context:
{context}

Question: {question}

Answer:"""
        
        return await self.generate_text(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt
        )
    
    async def generate_video_summary(
        self,
        frame_descriptions: List[str],
        transcript: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """Generate a summary of video content."""
        # Prepare context
        context_parts = []
        
        if frame_descriptions:
            context_parts.append("Visual content:")
            for i, desc in enumerate(frame_descriptions[:20]):  # Limit to 20 frames
                context_parts.append(f"- {desc}")
        
        if transcript:
            context_parts.append("\nAudio transcript:")
            context_parts.append(transcript[:1000])  # Limit transcript length
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following video content, provide a brief summary (3-5 sentences):

{context}

Summary:"""
        
        return await self.generate_text(prompt=prompt, model=model)
    
    async def batch_analyze_frames(
        self,
        frame_paths: List[Path],
        batch_size: int = 5
    ) -> List[FrameAnalysis]:
        """Analyze multiple frames in batches."""
        analyses = []
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(frame_paths), batch_size):
            batch = frame_paths[i:i + batch_size]
            
            # Create tasks for parallel processing
            tasks = []
            for j, frame_path in enumerate(batch):
                frame_number = i + j
                # Extract timestamp from filename if available
                timestamp = self._extract_timestamp_from_path(frame_path)
                
                task = self.analyze_frame(frame_path, frame_number, timestamp)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks)
            analyses.extend(batch_results)
            
            # Small delay between batches
            if i + batch_size < len(frame_paths):
                await asyncio.sleep(0.5)
        
        return analyses
    
    def _extract_timestamp_from_path(self, frame_path: Path) -> float:
        """Extract timestamp from frame filename."""
        # Expected format: frame_00001_t1.5s.jpg
        try:
            if '_t' in frame_path.stem and 's' in frame_path.stem:
                timestamp_str = frame_path.stem.split('_t')[1].rstrip('s')
                return float(timestamp_str)
        except:
            pass
        return 0.0
    
    async def check_model_capabilities(self, model_name: str) -> Dict[str, bool]:
        """Check what capabilities a model has."""
        capabilities = {
            "vision": False,
            "text": True,  # Most models support text
            "available": False
        }
        
        try:
            models = await self.list_models()
            for model in models:
                if model["name"] == model_name:
                    capabilities["available"] = True
                    # Check if it's a vision model
                    if "vision" in model_name.lower() or "llava" in model_name.lower():
                        capabilities["vision"] = True
                    break
            
            return capabilities
            
        except Exception as e:
            self.logger.error(f"Error checking model capabilities: {e}")
            return capabilities