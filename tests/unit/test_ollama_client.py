"""Unit tests for Ollama client."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import base64
import httpx

from src.llm.ollama_client import OllamaClient
from src.storage.schemas import FrameAnalysis


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.llm.ollama_host = "http://localhost:11434"
    config.llm.vision_model = "llava:latest"
    config.llm.text_model = "llama2:latest"
    config.llm.temperature = 0.7
    config.llm.timeout_seconds = 60
    return config


@pytest.fixture
def ollama_client(mock_config):
    """Create Ollama client instance."""
    with patch('src.llm.ollama_client.get_config', return_value=mock_config):
        client = OllamaClient()
        yield client


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image file."""
    image_path = tmp_path / "test_frame.jpg"
    # Create a simple 1x1 pixel image
    image_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd0\x07\xff\xd9'
    image_path.write_bytes(image_data)
    return image_path


class TestOllamaClient:
    """Test cases for OllamaClient."""
    
    @pytest.mark.asyncio
    async def test_init(self, mock_config):
        """Test client initialization."""
        with patch('src.llm.ollama_client.get_config', return_value=mock_config):
            async with OllamaClient() as client:
                assert client.host == "http://localhost:11434"
                assert client.timeout == 60
    
    @pytest.mark.asyncio
    async def test_is_available_success(self, ollama_client):
        """Test checking if Ollama is available."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(ollama_client.client, 'get', return_value=mock_response) as mock_get:
            available = await ollama_client.is_available()
            assert available is True
            mock_get.assert_called_once_with("/api/tags")
    
    @pytest.mark.asyncio
    async def test_is_available_failure(self, ollama_client):
        """Test checking Ollama availability when it's not running."""
        with patch.object(ollama_client.client, 'get', side_effect=httpx.ConnectError("Connection refused")):
            available = await ollama_client.is_available()
            assert available is False
    
    @pytest.mark.asyncio
    async def test_list_models(self, ollama_client):
        """Test listing available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:latest", "size": 1000000},
                {"name": "llava:latest", "size": 2000000}
            ]
        }
        mock_response.raise_for_status = Mock()
        
        with patch.object(ollama_client.client, 'get', return_value=mock_response):
            models = await ollama_client.list_models()
            assert len(models) == 2
            assert models[0]["name"] == "llama2:latest"
            assert models[1]["name"] == "llava:latest"
    
    @pytest.mark.asyncio
    async def test_analyze_image(self, ollama_client, sample_image_path):
        """Test image analysis."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "A person walking on a street with buildings in the background."
        }
        mock_response.raise_for_status = Mock()
        
        with patch.object(ollama_client.client, 'post', return_value=mock_response) as mock_post:
            result = await ollama_client.analyze_image(
                sample_image_path,
                "Describe what you see in this image."
            )
            
            assert result == "A person walking on a street with buildings in the background."
            
            # Verify request
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/generate"
            request_data = call_args[1]["json"]
            assert request_data["model"] == "llava:latest"
            assert "images" in request_data
            assert len(request_data["images"]) == 1
    
    @pytest.mark.asyncio
    async def test_analyze_image_file_not_found(self, ollama_client):
        """Test image analysis with non-existent file."""
        result = await ollama_client.analyze_image(
            "/nonexistent/image.jpg",
            "Describe this image"
        )
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_frame(self, ollama_client, sample_image_path):
        """Test analyzing a video frame."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "A person walking down a street. There are cars parked on the side."
        }
        mock_response.raise_for_status = Mock()
        
        with patch.object(ollama_client.client, 'post', return_value=mock_response):
            analysis = await ollama_client.analyze_frame(
                sample_image_path,
                frame_number=10,
                timestamp=5.5
            )
            
            assert isinstance(analysis, FrameAnalysis)
            assert analysis.frame_number == 10
            assert analysis.timestamp == 5.5
            assert "person walking" in analysis.description
            assert "person" in analysis.objects_detected
            assert "car" in analysis.objects_detected
            assert analysis.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_generate_text(self, ollama_client):
        """Test text generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "The video shows a busy street scene with people and vehicles."
        }
        mock_response.raise_for_status = Mock()
        
        with patch.object(ollama_client.client, 'post', return_value=mock_response) as mock_post:
            result = await ollama_client.generate_text(
                "Summarize the video content",
                system_prompt="You are a video analyst"
            )
            
            assert result == "The video shows a busy street scene with people and vehicles."
            
            # Verify request
            call_args = mock_post.call_args[1]["json"]
            assert call_args["model"] == "llama2:latest"
            assert call_args["temperature"] == 0.7
            assert call_args["system"] == "You are a video analyst"
    
    @pytest.mark.asyncio
    async def test_answer_video_question(self, ollama_client):
        """Test answering questions about video content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Yes, there are three people visible in the video."
        }
        mock_response.raise_for_status = Mock()
        
        context = "Frame 1: Two people walking. Frame 2: One person standing."
        question = "How many people are in the video?"
        
        with patch.object(ollama_client.client, 'post', return_value=mock_response):
            answer = await ollama_client.answer_video_question(question, context)
            
            assert "three people" in answer
    
    @pytest.mark.asyncio
    async def test_generate_video_summary(self, ollama_client):
        """Test video summary generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "The video captures a street scene with pedestrians and vehicles."
        }
        mock_response.raise_for_status = Mock()
        
        frame_descriptions = [
            "A person walking on sidewalk",
            "Cars passing by on the street",
            "People crossing at intersection"
        ]
        transcript = "You can hear traffic sounds and people talking."
        
        with patch.object(ollama_client.client, 'post', return_value=mock_response):
            summary = await ollama_client.generate_video_summary(
                frame_descriptions,
                transcript
            )
            
            assert "street scene" in summary
    
    @pytest.mark.asyncio
    async def test_batch_analyze_frames(self, ollama_client, tmp_path):
        """Test batch frame analysis."""
        # Create multiple test frames
        frame_paths = []
        for i in range(3):
            frame_path = tmp_path / f"frame_{i:05d}_t{i}.0s.jpg"
            frame_path.write_bytes(b"fake image data")
            frame_paths.append(frame_path)
        
        # Mock responses
        mock_analyze_frame = AsyncMock(side_effect=[
            FrameAnalysis(
                frame_number=i,
                timestamp=float(i),
                frame_path=str(frame_paths[i]),
                description=f"Frame {i} description",
                objects_detected=["object"],
                confidence=0.8
            )
            for i in range(3)
        ])
        
        with patch.object(ollama_client, 'analyze_frame', mock_analyze_frame):
            results = await ollama_client.batch_analyze_frames(frame_paths, batch_size=2)
            
            assert len(results) == 3
            assert all(isinstance(r, FrameAnalysis) for r in results)
            assert mock_analyze_frame.call_count == 3
    
    @pytest.mark.asyncio
    async def test_check_model_capabilities(self, ollama_client):
        """Test checking model capabilities."""
        mock_models = [
            {"name": "llama2:latest", "size": 1000000},
            {"name": "llava:latest", "size": 2000000}
        ]
        
        with patch.object(ollama_client, 'list_models', return_value=mock_models):
            # Check text model
            caps = await ollama_client.check_model_capabilities("llama2:latest")
            assert caps["available"] is True
            assert caps["text"] is True
            assert caps["vision"] is False
            
            # Check vision model
            caps = await ollama_client.check_model_capabilities("llava:latest")
            assert caps["available"] is True
            assert caps["text"] is True
            assert caps["vision"] is True
            
            # Check non-existent model
            caps = await ollama_client.check_model_capabilities("nonexistent:latest")
            assert caps["available"] is False
    
    def test_extract_objects_from_description(self, ollama_client):
        """Test object extraction from description."""
        description = "A person is walking their dog near a parked car on the street."
        objects = ollama_client._extract_objects_from_description(description)
        
        assert "person" in objects
        assert "dog" in objects
        assert "car" in objects
        assert "street" in objects
    
    def test_extract_timestamp_from_path(self, ollama_client):
        """Test timestamp extraction from frame path."""
        # Test valid timestamp
        path = Path("frame_00001_t5.5s.jpg")
        timestamp = ollama_client._extract_timestamp_from_path(path)
        assert timestamp == 5.5
        
        # Test invalid format
        path = Path("frame_00001.jpg")
        timestamp = ollama_client._extract_timestamp_from_path(path)
        assert timestamp == 0.0