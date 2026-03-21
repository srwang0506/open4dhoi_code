# 4DHOI Data Preparer

Web-based tool for preparing session data for the 4DHOI preprocessing pipeline. Combines video upload, scene splitting, and point prompt annotation into a single application.

## Features

- **Upload Tab**: Upload video files, auto-detect scenes with PySceneDetect, select and save segments
- **Annotate Tab**: Browse saved sessions, capture frames, annotate human/object click points for SAM2 mask generation

## Quick Start

```bash
pip install flask scenedetect[opencv]

# Run with default data directory (./data)
python app.py

# Custom data directory and port
python app.py --data_dir /path/to/sessions --port 5020
```

Open `http://localhost:5020` in your browser.

## Workflow

### 1. Upload Video (Upload Tab)
1. Enter the object category (e.g. "violin", "spear")
2. Drag & drop or select a video file
3. Click "Parse & Split Scenes" - PySceneDetect splits the video into segments (max 20s each)
4. Click segment cards to select up to 3 segments
5. Click "Save Selected" to save to the data directory

### 2. Annotate Points (Annotate Tab)
1. Click a session in the sidebar to load it
2. Play the video and navigate to the starting frame
3. Click "Capture Frame" to set the annotation base image
4. Click on the captured image to place human (cyan) and object (purple) points
5. Navigate to the desired index frame and click "Select Index Frame"
6. Click "Save Annotation" to write `points.json` and `select_id.json`

## Output Files

For each session, the tool creates:

```
data/{category}/{session_name}/
├── video.mp4          # Saved video segment
├── points.json        # {"human_points": [[x,y],...], "object_points": [[x,y],...]}
└── select_id.json     # {"select_id": N, "start_id": M, "object_name": "category"}
```

These are the required inputs for the preprocessing pipeline (`run_pipeline.sh`).

## Dependencies

- Python 3.8+
- Flask
- PySceneDetect (with OpenCV backend)
- ffmpeg / ffprobe (system)
