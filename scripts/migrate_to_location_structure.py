#!/usr/bin/env python3
"""Migrate existing videos to the new location-based structure."""
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import sqlite3

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.date_parser import DateParser
from src.storage.schemas import ProcessingStatus


def migrate_videos(base_path: Path):
    """Migrate videos from old structure to new location-based structure."""
    print("🔄 Migrating videos to location-based structure...")
    
    # Check for old structure
    old_originals = base_path / "originals"
    if not old_originals.exists():
        print("✅ No old structure found. Nothing to migrate.")
        return
    
    # Create new structure
    locations_dir = base_path / "locations"
    locations_dir.mkdir(exist_ok=True)
    
    # Get database connection
    db_path = base_path / "index" / "metadata.db"
    if not db_path.exists():
        print("❌ No database found. Cannot migrate.")
        return
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check if new columns exist
    cursor.execute("PRAGMA table_info(videos)")
    columns = [col['name'] for col in cursor.fetchall()]
    
    if 'location' not in columns:
        print("📝 Adding location and recording_timestamp columns to database...")
        cursor.execute("ALTER TABLE videos ADD COLUMN location TEXT DEFAULT 'unknown'")
        cursor.execute("ALTER TABLE videos ADD COLUMN recording_timestamp TIMESTAMP")
        conn.commit()
        
        # Update existing records with default values
        cursor.execute("UPDATE videos SET recording_timestamp = created_at WHERE recording_timestamp IS NULL")
        conn.commit()
    
    # Get all videos
    cursor.execute("SELECT * FROM videos")
    videos = cursor.fetchall()
    
    migrated = 0
    for video in videos:
        video_id = video['video_id']
        
        # Find the video file
        old_video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            test_path = old_originals / f"{video_id}{ext}"
            if test_path.exists():
                old_video_path = test_path
                break
        
        if not old_video_path:
            print(f"⚠️  Video file not found for {video_id}")
            continue
        
        # Determine location and timestamp
        location = video['location'] if 'location' in columns and video['location'] else 'unknown'
        
        # Try to get timestamp from recording_timestamp or created_at
        if 'recording_timestamp' in columns and video['recording_timestamp']:
            timestamp = datetime.fromisoformat(video['recording_timestamp'])
        else:
            timestamp = datetime.fromisoformat(video['created_at'])
        
        # Create new path
        year, month, day = DateParser.get_date_path_components(timestamp)
        new_dir = locations_dir / location / year / month / day
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Create new filename
        time_str = timestamp.strftime("%H%M%S")
        new_filename = f"{video_id}_{time_str}{old_video_path.suffix}"
        new_path = new_dir / new_filename
        
        # Move file
        print(f"📦 Moving {video_id} to {location}/{year}/{month}/{day}/")
        shutil.move(str(old_video_path), str(new_path))
        
        # Update database if needed
        if location == 'unknown':
            cursor.execute(
                "UPDATE videos SET location = ? WHERE video_id = ?",
                (location, video_id)
            )
        
        migrated += 1
    
    conn.commit()
    conn.close()
    
    # Remove old directories if empty
    if old_originals.exists() and not any(old_originals.iterdir()):
        old_originals.rmdir()
        print("🗑️  Removed empty 'originals' directory")
    
    print(f"\n✅ Migration complete! Migrated {migrated} videos.")
    
    # Create indexes if they don't exist
    print("\n📇 Ensuring database indexes...")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_location ON videos(location)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_timestamp ON videos(recording_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_location_timestamp ON videos(location, recording_timestamp)")
        conn.commit()
        print("✅ Indexes created/verified")
    except Exception as e:
        print(f"⚠️  Index creation warning: {e}")
    finally:
        conn.close()


def main():
    """Main migration function."""
    print("🎬 MCP Video Server - Location Structure Migration")
    print("=" * 50)
    
    # Determine base path
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    else:
        # Try to find video_data directory
        possible_paths = [
            Path.cwd() / "video_data",
            Path(__file__).parent.parent / "video_data",
            Path.home() / ".mcp-video" / "video_data"
        ]
        
        base_path = None
        for path in possible_paths:
            if path.exists():
                base_path = path
                break
        
        if not base_path:
            print("❌ Could not find video_data directory.")
            print("   Please provide the path as an argument:")
            print("   python migrate_to_location_structure.py /path/to/video_data")
            sys.exit(1)
    
    print(f"📍 Using base path: {base_path}")
    
    # Confirm
    response = input("\nThis will reorganize your video files. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    try:
        migrate_videos(base_path)
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()