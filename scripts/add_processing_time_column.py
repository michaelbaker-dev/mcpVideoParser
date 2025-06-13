#!/usr/bin/env python3
"""Add processing_time column to videos table."""
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config

def main():
    """Add processing_time column to videos table."""
    config = get_config()
    db_path = Path(config.storage.base_path) / "index" / "metadata.db"
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return
        
    print(f"Adding processing_time column to {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(videos)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'processing_time' in columns:
            print("Column 'processing_time' already exists")
        else:
            # Add the column
            cursor.execute("ALTER TABLE videos ADD COLUMN processing_time REAL DEFAULT 0.0")
            conn.commit()
            print("Successfully added processing_time column")
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()