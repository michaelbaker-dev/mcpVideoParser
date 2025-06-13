#!/usr/bin/env python3
"""Migrate existing database to new schema with location and recording_timestamp."""
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config


def migrate_database():
    """Add missing columns to existing database."""
    config = get_config()
    db_path = Path(config.storage.base_path) / "index" / "metadata.db"
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return False
    
    print(f"Migrating database at {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(videos)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add location column if missing
        if 'location' not in columns:
            print("Adding 'location' column...")
            cursor.execute("""
                ALTER TABLE videos 
                ADD COLUMN location TEXT DEFAULT 'unknown'
            """)
            # Update existing records with a default location
            cursor.execute("""
                UPDATE videos 
                SET location = 'unknown' 
                WHERE location IS NULL
            """)
        
        # Add recording_timestamp column if missing
        if 'recording_timestamp' not in columns:
            print("Adding 'recording_timestamp' column...")
            cursor.execute("""
                ALTER TABLE videos 
                ADD COLUMN recording_timestamp TEXT
            """)
            # Set recording_timestamp to created_at for existing records
            cursor.execute("""
                UPDATE videos 
                SET recording_timestamp = created_at 
                WHERE recording_timestamp IS NULL
            """)
        
        # Create new indexes
        print("Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_location ON videos(location)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_timestamp ON videos(recording_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_location_timestamp ON videos(location, recording_timestamp)")
        
        conn.commit()
        
        # Verify migration
        cursor.execute("PRAGMA table_info(videos)")
        new_columns = [col[1] for col in cursor.fetchall()]
        
        if 'location' in new_columns and 'recording_timestamp' in new_columns:
            print("✅ Migration successful!")
            
            # Show current videos
            cursor.execute("SELECT COUNT(*) FROM videos")
            count = cursor.fetchone()[0]
            print(f"\nDatabase contains {count} videos")
            
            if count > 0:
                print("\nSample migrated records:")
                cursor.execute("""
                    SELECT video_id, filename, location, recording_timestamp 
                    FROM videos 
                    LIMIT 5
                """)
                for row in cursor.fetchall():
                    print(f"  - {row[0]}: {row[1]} | Location: {row[2]} | Time: {row[3]}")
            
            return True
        else:
            print("❌ Migration failed - columns not added")
            return False
            
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False
    finally:
        conn.close()


def backup_database():
    """Create a backup of the database before migration."""
    config = get_config()
    db_path = Path(config.storage.base_path) / "index" / "metadata.db"
    
    if db_path.exists():
        backup_path = db_path.with_suffix('.db.backup')
        print(f"Creating backup at {backup_path}")
        import shutil
        shutil.copy2(db_path, backup_path)
        return True
    return False


if __name__ == "__main__":
    print("=== Database Migration Tool ===\n")
    
    # Create backup first
    if backup_database():
        print("Backup created successfully\n")
    
    # Run migration
    if migrate_database():
        print("\n✅ Database migration completed successfully!")
        print("\nYou can now run the MCP server with: ./server.py")
    else:
        print("\n❌ Migration failed. Check the error messages above.")
        print("Your original database has been backed up as metadata.db.backup")