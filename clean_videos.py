#!/usr/bin/env python3
"""
Clean all videos from the database and file system.

This script provides a complete reset of the video storage system by:
1. Removing all video entries from the database
2. Deleting all video files from the location-based structure
3. Removing all processed files (frames, transcripts)
4. Optionally backing up the database before cleaning
"""

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.manager import StorageManager
from src.utils.logging import get_logger


class VideoCleaner:
    """Handles complete cleanup of video storage system."""
    
    def __init__(self, backup_db: bool = True):
        """
        Initialize the video cleaner.
        
        Args:
            backup_db: Whether to backup database before cleaning
        """
        self.storage = StorageManager()
        self.logger = get_logger(__name__)
        self.backup_db = backup_db
        self.stats = {
            'videos_removed': 0,
            'files_deleted': 0,
            'space_freed': 0
        }
    
    def backup_database(self) -> Optional[Path]:
        """Create a backup of the database before cleaning."""
        if not self.backup_db:
            return None
            
        db_path = self.storage.base_path / "index" / "metadata.db"
        if not db_path.exists():
            self.logger.warning("No database found to backup")
            return None
            
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.parent / f"metadata_backup_{timestamp}.db"
        
        try:
            shutil.copy2(db_path, backup_path)
            self.logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return None
    
    def get_all_videos(self) -> list:
        """Get all video records from the database."""
        conn = sqlite3.connect(self.storage.base_path / "index" / "metadata.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM videos")
            videos = cursor.fetchall()
            return [dict(row) for row in videos]
        finally:
            conn.close()
    
    def clean_database(self) -> int:
        """Remove all video entries from the database."""
        conn = sqlite3.connect(self.storage.base_path / "index" / "metadata.db")
        cursor = conn.cursor()
        
        try:
            # Get count before deletion
            cursor.execute("SELECT COUNT(*) FROM videos")
            count = cursor.fetchone()[0]
            
            # Delete all related data
            self.logger.info("Cleaning database tables...")
            
            # Delete from all related tables
            tables = ['frame_analyses', 'transcripts', 'summaries', 'videos']
            for table in tables:
                try:
                    cursor.execute(f"DELETE FROM {table}")
                    self.logger.debug(f"Cleared table: {table}")
                except sqlite3.OperationalError:
                    self.logger.warning(f"Table {table} does not exist")
            
            conn.commit()
            self.logger.info(f"Removed {count} video entries from database")
            return count
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database cleanup failed: {e}")
            raise
        finally:
            conn.close()
    
    def clean_file_system(self) -> tuple[int, int]:
        """
        Remove all video files from the file system.
        
        Returns:
            Tuple of (files_deleted, bytes_freed)
        """
        files_deleted = 0
        bytes_freed = 0
        
        # Clean location-based structure
        locations_dir = self.storage.base_path / "locations"
        if locations_dir.exists():
            self.logger.info(f"Cleaning location directories: {locations_dir}")
            
            for location_dir in locations_dir.iterdir():
                if location_dir.is_dir():
                    # Calculate size before deletion
                    size = sum(f.stat().st_size for f in location_dir.rglob('*') if f.is_file())
                    file_count = sum(1 for f in location_dir.rglob('*') if f.is_file())
                    
                    self.logger.info(f"Removing location '{location_dir.name}': {file_count} files, {size / 1024 / 1024:.2f} MB")
                    
                    try:
                        shutil.rmtree(location_dir)
                        files_deleted += file_count
                        bytes_freed += size
                    except Exception as e:
                        self.logger.error(f"Failed to remove {location_dir}: {e}")
        
        # Clean processed files
        processed_dir = self.storage.base_path / "processed"
        if processed_dir.exists():
            self.logger.info(f"Cleaning processed files: {processed_dir}")
            
            for video_dir in processed_dir.iterdir():
                if video_dir.is_dir():
                    # Calculate size
                    size = sum(f.stat().st_size for f in video_dir.rglob('*') if f.is_file())
                    file_count = sum(1 for f in video_dir.rglob('*') if f.is_file())
                    
                    try:
                        shutil.rmtree(video_dir)
                        files_deleted += file_count
                        bytes_freed += size
                    except Exception as e:
                        self.logger.error(f"Failed to remove {video_dir}: {e}")
        
        # Clean temp directory
        temp_dir = self.storage.base_path / "temp"
        if temp_dir.exists():
            self.logger.info("Cleaning temp directory")
            try:
                shutil.rmtree(temp_dir)
                temp_dir.mkdir()  # Recreate empty temp dir
            except Exception as e:
                self.logger.error(f"Failed to clean temp directory: {e}")
        
        return files_deleted, bytes_freed
    
    def clean_originals(self) -> tuple[int, int]:
        """
        Optionally clean original video files.
        
        Returns:
            Tuple of (files_deleted, bytes_freed)
        """
        originals_dir = self.storage.base_path / "originals"
        if not originals_dir.exists():
            return 0, 0
            
        files_deleted = 0
        bytes_freed = 0
        
        self.logger.info(f"Cleaning original files: {originals_dir}")
        
        for file in originals_dir.iterdir():
            if file.is_file() and file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                size = file.stat().st_size
                try:
                    file.unlink()
                    files_deleted += 1
                    bytes_freed += size
                    self.logger.debug(f"Deleted: {file.name}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {file}: {e}")
        
        return files_deleted, bytes_freed
    
    def run(self, clean_originals: bool = False) -> dict:
        """
        Run the complete cleanup process.
        
        Args:
            clean_originals: Whether to also delete original video files
            
        Returns:
            Dictionary with cleanup statistics
        """
        self.logger.info("Starting video cleanup process...")
        
        # Get video count before cleanup
        videos = self.get_all_videos()
        self.stats['videos_removed'] = len(videos)
        
        # Backup database if requested
        backup_path = self.backup_database()
        if backup_path:
            self.stats['backup_path'] = str(backup_path)
        
        # Clean database
        try:
            self.clean_database()
        except Exception as e:
            self.logger.error(f"Database cleanup failed: {e}")
            if backup_path:
                self.logger.info(f"Database backup available at: {backup_path}")
            raise
        
        # Clean file system
        files_deleted, bytes_freed = self.clean_file_system()
        self.stats['files_deleted'] = files_deleted
        self.stats['space_freed'] = bytes_freed
        
        # Optionally clean originals
        if clean_originals:
            orig_files, orig_bytes = self.clean_originals()
            self.stats['files_deleted'] += orig_files
            self.stats['space_freed'] += orig_bytes
            self.stats['originals_deleted'] = orig_files
        
        self.logger.info("Video cleanup completed successfully!")
        return self.stats


def main():
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(
        description="Clean all videos from the database and file system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean all processed files and database (keeps originals)
  python clean_videos.py
  
  # Clean everything including original video files
  python clean_videos.py --clean-originals
  
  # Clean without creating database backup
  python clean_videos.py --no-backup
  
  # Dry run to see what would be deleted
  python clean_videos.py --dry-run
        """
    )
    
    parser.add_argument(
        "--clean-originals",
        action="store_true",
        help="Also delete original video files (default: keep originals)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip database backup before cleaning"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = VideoCleaner(backup_db=not args.no_backup)
    
    # Get current state
    videos = cleaner.get_all_videos()
    
    if not videos:
        print("No videos found in the system. Nothing to clean.")
        return
    
    # Show what will be cleaned
    print(f"\nFound {len(videos)} videos in the database")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("The following would be deleted:")
        
        # Show locations
        locations_dir = cleaner.storage.base_path / "locations"
        if locations_dir.exists():
            print(f"\nLocation directories:")
            for loc in locations_dir.iterdir():
                if loc.is_dir():
                    file_count = sum(1 for f in loc.rglob('*') if f.is_file())
                    size = sum(f.stat().st_size for f in loc.rglob('*') if f.is_file())
                    print(f"  - {loc.name}: {file_count} files, {size / 1024 / 1024:.2f} MB")
        
        # Show processed
        processed_dir = cleaner.storage.base_path / "processed"
        if processed_dir.exists():
            video_count = sum(1 for d in processed_dir.iterdir() if d.is_dir())
            print(f"\nProcessed video directories: {video_count}")
        
        if args.clean_originals:
            originals_dir = cleaner.storage.base_path / "originals"
            if originals_dir.exists():
                orig_videos = [f for f in originals_dir.iterdir() 
                              if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']]
                print(f"\nOriginal videos: {len(orig_videos)} files")
        
        print("\nNo files were deleted (dry run mode)")
        return
    
    # Confirm with user
    if not args.yes:
        print("\nThis will:")
        print("- Remove all video entries from the database")
        print("- Delete all processed frames and transcripts")
        print("- Delete all videos from the location-based structure")
        if args.clean_originals:
            print("- Delete all original video files")
        if not args.no_backup:
            print("- Create a backup of the database first")
        
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return
    
    # Run cleanup
    try:
        stats = cleaner.run(clean_originals=args.clean_originals)
        
        # Display results
        print("\n=== Cleanup Complete ===")
        print(f"Videos removed from database: {stats['videos_removed']}")
        print(f"Files deleted: {stats['files_deleted']}")
        print(f"Space freed: {stats['space_freed'] / 1024 / 1024:.2f} MB")
        
        if 'originals_deleted' in stats:
            print(f"Original videos deleted: {stats['originals_deleted']}")
        
        if 'backup_path' in stats:
            print(f"\nDatabase backup saved to: {stats['backup_path']}")
        
    except Exception as e:
        print(f"\nError during cleanup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()