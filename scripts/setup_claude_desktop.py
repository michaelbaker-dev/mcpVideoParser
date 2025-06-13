#!/usr/bin/env python3
"""Setup script for Claude Desktop configuration."""
import json
import os
import sys
from pathlib import Path
import platform
import shutil


def get_claude_config_path():
    """Get the Claude Desktop configuration file path."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        config_path = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    else:  # Linux
        config_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    
    return config_path


def backup_config(config_path):
    """Backup existing configuration."""
    if config_path.exists():
        backup_path = config_path.with_suffix('.json.backup')
        shutil.copy2(config_path, backup_path)
        print(f"✅ Backed up existing config to: {backup_path}")
        return True
    return False


def setup_claude_desktop():
    """Set up Claude Desktop configuration for the MCP video server."""
    print("🎬 MCP Video Analysis Server - Claude Desktop Setup")
    print("=" * 50)
    
    # Get paths
    project_dir = Path(__file__).parent.parent.absolute()
    server_path = project_dir / "server.py"
    
    if not server_path.exists():
        print(f"❌ Error: server.py not found at {server_path}")
        sys.exit(1)
    
    # Get Claude config path
    config_path = get_claude_config_path()
    print(f"📍 Claude config location: {config_path}")
    
    # Create config directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load or create config
    if config_path.exists():
        backup_config(config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {"mcpServers": {}}
        print("📝 Creating new Claude Desktop configuration")
    
    # Add our server configuration
    server_config = {
        "command": sys.executable,  # Use current Python interpreter
        "args": [str(server_path)],
        "env": {
            "PYTHONPATH": str(project_dir),
            "OLLAMA_HOST": "http://localhost:11434",
            "VIDEO_DATA_PATH": str(project_dir / "video_data")
        }
    }
    
    # Check if already configured
    if "mcp-video-analysis" in config.get("mcpServers", {}):
        print("\n⚠️  MCP Video Analysis server is already configured.")
        response = input("Do you want to update the configuration? (y/n): ")
        if response.lower() != 'y':
            print("Configuration unchanged.")
            return
    
    # Update configuration
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    config["mcpServers"]["mcp-video-analysis"] = server_config
    
    # Write configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Claude Desktop configuration updated successfully!")
    
    # Print instructions
    print("\n📋 Next steps:")
    print("1. Make sure Ollama is installed and running:")
    print("   - Install: https://ollama.ai")
    print("   - Start: ollama serve")
    print("   - Pull models: ollama pull llava && ollama pull llama2")
    print("\n2. Install Python dependencies:")
    print(f"   cd {project_dir}")
    print("   pip install -r requirements.txt")
    print("\n3. Restart Claude Desktop to load the new configuration")
    print("\n4. In Claude, you can now use commands like:")
    print("   - 'Process the video at /path/to/video.mp4'")
    print("   - 'What happens in video vid_abc123?'")
    print("   - 'Search for videos mentioning cats'")
    
    # Check Ollama
    print("\n🔍 Checking Ollama status...")
    try:
        import httpx
        client = httpx.Client()
        response = client.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✅ Ollama is running")
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            required_models = ["llava:latest", "llama2:latest"]
            missing_models = [m for m in required_models if m not in model_names]
            
            if missing_models:
                print(f"⚠️  Missing required models: {', '.join(missing_models)}")
                print("   Run: " + " && ".join(f"ollama pull {m}" for m in missing_models))
            else:
                print("✅ All required models are installed")
        else:
            print("❌ Ollama is not responding correctly")
    except Exception as e:
        print(f"❌ Ollama is not running. Start it with: ollama serve")
        print(f"   Error: {e}")


if __name__ == "__main__":
    try:
        setup_claude_desktop()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)