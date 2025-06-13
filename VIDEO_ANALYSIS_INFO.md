# Video Analysis System - How It Works

## Video Processing Coverage

The video analysis system captures frames throughout the entire video using a combination of:

### 1. **Frame Sampling** (Default: Every 30 frames)
- At 30 fps, this means capturing approximately 1 frame per second
- Configurable via `frame_sample_rate` in config.json

### 2. **Scene Detection** (Enabled by default)
- Automatically detects significant visual changes
- Captures frames when scenes change (e.g., when a car enters the frame)
- Uses a configurable threshold (default: 0.3)

### 3. **Maximum Frames Limit**
- Up to 1000 frames per video can be analyzed
- Evenly distributed if video has more potential frames

## Example: Car Appearing Later in Video

If a car appears at the 3-minute mark in a 5-minute video:

1. **Regular Sampling**: Would capture it if it appears at a sampled frame (every ~1 second)
2. **Scene Detection**: Would likely capture it as a significant visual change
3. **Frame Description**: Each frame gets a detailed description from the vision LLM

## Configuration Options

Edit `config/default_config.json` to adjust:

```json
{
  "processing": {
    "frame_sample_rate": 30,      // Lower = more frames captured
    "enable_scene_detection": true,
    "scene_threshold": 0.3,        // Lower = more sensitive
    "max_frames_per_video": 1000
  }
}
```

### To Capture More Detail:

1. **Reduce frame_sample_rate**: 
   - 15 = capture every 0.5 seconds
   - 10 = capture every 0.33 seconds
   
2. **Lower scene_threshold**:
   - 0.2 = more sensitive to changes
   - 0.1 = very sensitive (might capture too many frames)

3. **Increase max_frames_per_video**:
   - 2000 = allow more frames per video
   - Note: This increases processing time

## Processing a Video with Different Settings

```bash
# Process with more frequent frame capture
python standalone_client/video_client.py process /path/to/video.mp4 --sample-rate 15

# Or use the MCP tool with custom settings
process_video video_path="/path/to/video.mp4" sample_rate=15
```

## Current Limitations

1. **Audio Transcription**: Captures spoken words but doesn't correlate with specific video moments
2. **Real-time Analysis**: Videos must be pre-processed before querying
3. **Storage**: More frames = more storage space required

## Recommended Settings for Different Use Cases

### Security/Surveillance
```json
"frame_sample_rate": 15,
"enable_scene_detection": true,
"scene_threshold": 0.2
```

### General Content Review  
```json
"frame_sample_rate": 30,
"enable_scene_detection": true,
"scene_threshold": 0.3
```

### Detailed Analysis
```json
"frame_sample_rate": 10,
"enable_scene_detection": true,
"scene_threshold": 0.15,
"max_frames_per_video": 2000
```