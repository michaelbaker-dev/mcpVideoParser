#!/usr/bin/env python3
"""Setup script for MCP Video Analysis Server."""
import os
import sys
import subprocess
import platform
from pathlib import Path
import json


def print_banner():
    """Print welcome banner."""
    print("""
    ╔══════════════════════════════════════════╗
    ║   🎬 MCP Video Analysis Server Setup 🎬   ║
    ╚══════════════════════════════════════════╝
    """)


def check_python_version():
    """Check Python version."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True


def check_command(command, install_hint):
    """Check if a command is available."""
    print(f"🔍 Checking {command}...")
    try:
        subprocess.run([command, "--version"], capture_output=True, check=False)
        print(f"✅ {command} is installed")
        return True
    except FileNotFoundError:
        print(f"❌ {command} not found")
        print(f"   Install: {install_hint}")
        return False


def check_ollama():
    """Check Ollama installation and models."""
    print("\n🔍 Checking Ollama...")
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print("✅ Ollama is installed")
    except FileNotFoundError:
        print("❌ Ollama not found")
        print("   Install from: https://ollama.ai")
        return False
    
    # Check if Ollama is running
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            # Check models
            data = response.json()
            models = {m["name"] for m in data.get("models", [])}
            
            required = ["llava:latest", "llama2:latest"]
            missing = [m for m in required if m not in models]
            
            if missing:
                print(f"⚠️  Missing models: {', '.join(missing)}")
                print("   Install with:")
                for model in missing:
                    print(f"     ollama pull {model}")
                return False
            else:
                print("✅ All required models installed")
                return True
        else:
            print("❌ Ollama is not responding")
            print("   Start with: ollama serve")
            return False
    except Exception as e:
        print("❌ Ollama is not running")
        print("   Start with: ollama serve")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    dirs = [
        "logs",
        "video_data/originals",
        "video_data/processed",
        "video_data/index"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True


def test_video_processing():
    """Test video processing with sample video."""
    print("\n🧪 Testing video processing...")
    
    sample_video = Path("video_data/originals/sample_video.mp4")
    if not sample_video.exists():
        print("⚠️  Sample video not found")
        print("   The sample video should have been downloaded during setup")
        return False
    
    print("✅ Sample video found")
    
    # Try importing our modules
    try:
        from src.storage.manager import StorageManager
        from src.processors.video import VideoProcessor
        print("✅ Modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def setup_claude_desktop():
    """Run Claude Desktop setup."""
    print("\n🤖 Setting up Claude Desktop integration...")
    try:
        subprocess.run([sys.executable, "scripts/setup_claude_desktop.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Claude Desktop setup failed")
        return False


def main():
    """Main setup function."""
    print_banner()
    
    checks = {
        "Python version": check_python_version(),
        "ffmpeg": check_command("ffmpeg", "brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"),
        "Dependencies": install_dependencies(),
        "Directories": create_directories(),
        "Ollama": check_ollama(),
        "Modules": test_video_processing(),
    }
    
    # Summary
    print("\n" + "="*50)
    print("📊 Setup Summary:")
    print("="*50)
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Setup complete! All checks passed.")
        
        # Ask about Claude Desktop setup
        response = input("\nWould you like to set up Claude Desktop integration? (y/n): ")
        if response.lower() == 'y':
            setup_claude_desktop()
        
        print("\n📚 Next steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. For Claude Desktop: Restart Claude to load the configuration")
        print("3. For CLI usage: ./standalone_client/video_client.py --help")
        print("\n🎬 Try processing the sample video:")
        print("   ./standalone_client/video_client.py process video_data/originals/sample_video.mp4")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above and run again.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)