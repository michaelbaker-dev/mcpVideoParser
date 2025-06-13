"""Tests for the clean_videos.py script."""
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from clean_videos import VideoCleaner


class TestVideoCleaner:
    """Test the VideoCleaner class."""
    
    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage structure."""
        # Create directory structure
        (tmp_path / "index").mkdir()
        (tmp_path / "locations" / "test_location" / "2024" / "01" / "01").mkdir(parents=True)
        (tmp_path / "locations" / "shed" / "2024" / "02" / "15").mkdir(parents=True)
        (tmp_path / "processed" / "vid_123" / "frames").mkdir(parents=True)
        (tmp_path / "processed" / "vid_456" / "frames").mkdir(parents=True)
        (tmp_path / "originals").mkdir()
        (tmp_path / "temp").mkdir()
        
        # Create test files
        (tmp_path / "locations" / "test_location" / "2024" / "01" / "01" / "video1.mp4").touch()
        (tmp_path / "locations" / "shed" / "2024" / "02" / "15" / "video2.mp4").touch()
        (tmp_path / "processed" / "vid_123" / "frames" / "frame_001.jpg").touch()
        (tmp_path / "processed" / "vid_123" / "transcript.txt").touch()
        (tmp_path / "processed" / "vid_456" / "frames" / "frame_001.jpg").touch()
        (tmp_path / "originals" / "original1.mp4").touch()
        (tmp_path / "originals" / "original2.avi").touch()
        
        # Write some data to files to simulate size
        for file in tmp_path.rglob("*.mp4"):
            file.write_bytes(b"x" * 1000)
        for file in tmp_path.rglob("*.jpg"):
            file.write_bytes(b"x" * 500)
            
        # Create test database
        db_path = tmp_path / "index" / "metadata.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE videos (
                video_id TEXT PRIMARY KEY,
                filename TEXT,
                location TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE frame_analyses (
                video_id TEXT,
                frame_number INTEGER,
                description TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE transcripts (
                video_id TEXT,
                transcript TEXT
            )
        """)
        
        # Insert test data
        cursor.execute("INSERT INTO videos VALUES ('vid_123', 'video1.mp4', 'test_location')")
        cursor.execute("INSERT INTO videos VALUES ('vid_456', 'video2.mp4', 'shed')")
        cursor.execute("INSERT INTO frame_analyses VALUES ('vid_123', 1, 'Test frame')")
        cursor.execute("INSERT INTO transcripts VALUES ('vid_123', 'Test transcript')")
        
        conn.commit()
        conn.close()
        
        return tmp_path
    
    @pytest.fixture
    def video_cleaner(self, temp_storage):
        """Create a VideoCleaner instance with temp storage."""
        with patch('clean_videos.StorageManager') as MockStorage:
            mock_storage = MockStorage.return_value
            mock_storage.base_path = temp_storage
            cleaner = VideoCleaner(backup_db=True)
            cleaner.storage = mock_storage
            return cleaner
    
    def test_backup_database(self, video_cleaner, temp_storage):
        """Test database backup functionality."""
        backup_path = video_cleaner.backup_database()
        
        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.name.startswith("metadata_backup_")
        assert backup_path.suffix == ".db"
        
        # Verify backup is valid
        conn = sqlite3.connect(backup_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM videos")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 2
    
    def test_get_all_videos(self, video_cleaner):
        """Test retrieving all videos from database."""
        videos = video_cleaner.get_all_videos()
        
        assert len(videos) == 2
        assert videos[0]['video_id'] in ['vid_123', 'vid_456']
        assert videos[1]['video_id'] in ['vid_123', 'vid_456']
    
    def test_clean_database(self, video_cleaner):
        """Test database cleanup."""
        # Verify initial state
        videos = video_cleaner.get_all_videos()
        assert len(videos) == 2
        
        # Clean database
        count = video_cleaner.clean_database()
        assert count == 2
        
        # Verify cleaned state
        videos = video_cleaner.get_all_videos()
        assert len(videos) == 0
        
        # Check all tables are empty
        conn = sqlite3.connect(video_cleaner.storage.base_path / "index" / "metadata.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM frame_analyses")
        assert cursor.fetchone()[0] == 0
        
        cursor.execute("SELECT COUNT(*) FROM transcripts")
        assert cursor.fetchone()[0] == 0
        
        conn.close()
    
    def test_clean_file_system(self, video_cleaner):
        """Test file system cleanup."""
        # Check initial state
        locations_dir = video_cleaner.storage.base_path / "locations"
        assert len(list(locations_dir.iterdir())) == 2
        
        processed_dir = video_cleaner.storage.base_path / "processed"
        assert len(list(processed_dir.iterdir())) == 2
        
        # Clean file system
        files_deleted, bytes_freed = video_cleaner.clean_file_system()
        
        assert files_deleted > 0
        assert bytes_freed > 0
        
        # Verify cleaned state
        assert len(list(locations_dir.iterdir())) == 0
        assert len(list(processed_dir.iterdir())) == 0
        
        # Temp dir should exist but be empty
        temp_dir = video_cleaner.storage.base_path / "temp"
        assert temp_dir.exists()
        assert len(list(temp_dir.iterdir())) == 0
    
    def test_clean_originals(self, video_cleaner):
        """Test cleaning original video files."""
        originals_dir = video_cleaner.storage.base_path / "originals"
        
        # Check initial state
        video_files = [f for f in originals_dir.iterdir() if f.suffix in ['.mp4', '.avi']]
        assert len(video_files) == 2
        
        # Clean originals
        files_deleted, bytes_freed = video_cleaner.clean_originals()
        
        assert files_deleted == 2
        assert bytes_freed > 0
        
        # Verify cleaned state
        video_files = [f for f in originals_dir.iterdir() if f.suffix in ['.mp4', '.avi']]
        assert len(video_files) == 0
    
    def test_run_without_originals(self, video_cleaner):
        """Test complete cleanup without deleting originals."""
        stats = video_cleaner.run(clean_originals=False)
        
        assert stats['videos_removed'] == 2
        assert stats['files_deleted'] > 0
        assert stats['space_freed'] > 0
        assert 'backup_path' in stats
        assert 'originals_deleted' not in stats
        
        # Verify originals still exist
        originals_dir = video_cleaner.storage.base_path / "originals"
        video_files = [f for f in originals_dir.iterdir() if f.suffix in ['.mp4', '.avi']]
        assert len(video_files) == 2
    
    def test_run_with_originals(self, video_cleaner):
        """Test complete cleanup including originals."""
        stats = video_cleaner.run(clean_originals=True)
        
        assert stats['videos_removed'] == 2
        assert stats['files_deleted'] > 0
        assert stats['space_freed'] > 0
        assert 'backup_path' in stats
        assert stats['originals_deleted'] == 2
        
        # Verify everything is cleaned
        originals_dir = video_cleaner.storage.base_path / "originals"
        video_files = [f for f in originals_dir.iterdir() if f.suffix in ['.mp4', '.avi']]
        assert len(video_files) == 0
    
    def test_run_no_backup(self, temp_storage):
        """Test cleanup without database backup."""
        with patch('clean_videos.StorageManager') as MockStorage:
            mock_storage = MockStorage.return_value
            mock_storage.base_path = temp_storage
            
            cleaner = VideoCleaner(backup_db=False)
            cleaner.storage = mock_storage
            
            stats = cleaner.run(clean_originals=False)
            
            assert 'backup_path' not in stats
            assert stats['videos_removed'] == 2
    
    def test_main_dry_run(self, temp_storage, capsys):
        """Test main function with dry run."""
        from clean_videos import main
        
        with patch('sys.argv', ['clean_videos.py', '--dry-run']):
            with patch('clean_videos.StorageManager') as MockStorage:
                mock_storage = MockStorage.return_value
                mock_storage.base_path = temp_storage
                
                main()
                
                captured = capsys.readouterr()
                assert "DRY RUN MODE" in captured.out
                assert "No files were deleted" in captured.out
                
                # Verify nothing was actually deleted
                locations_dir = temp_storage / "locations"
                assert len(list(locations_dir.iterdir())) == 2
    
    def test_main_with_confirmation(self, temp_storage, capsys):
        """Test main function with user confirmation."""
        from clean_videos import main
        
        with patch('sys.argv', ['clean_videos.py']):
            with patch('clean_videos.StorageManager') as MockStorage:
                mock_storage = MockStorage.return_value
                mock_storage.base_path = temp_storage
                
                # Simulate user saying no
                with patch('builtins.input', return_value='no'):
                    main()
                    
                    captured = capsys.readouterr()
                    assert "Cleanup cancelled" in captured.out
    
    def test_main_skip_confirmation(self, temp_storage, capsys):
        """Test main function skipping confirmation."""
        from clean_videos import main
        
        with patch('sys.argv', ['clean_videos.py', '--yes', '--no-backup']):
            with patch('clean_videos.StorageManager') as MockStorage:
                mock_storage = MockStorage.return_value
                mock_storage.base_path = temp_storage
                
                main()
                
                captured = capsys.readouterr()
                assert "Cleanup Complete" in captured.out
                assert "Videos removed from database: 2" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])