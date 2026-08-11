# VALORANT VOD Post-Match Analyzer

Runs your trained YOLO11n model over a **recorded VOD** to find enemy-sighting
moments, cut highlight clips, capture annotated frames, plot on-screen
positioning, and show it all in a dashboard. Fully offline — reads a video
file you already recorded and writes files to disk. Nothing here attaches
to a running game process.

Your model's classes: `enemy`, `enemy head`, `crosshair`, `teammate`.

Two ways to use it:
- **`app.py`** — a local web UI: drag-and-drop a VOD in your browser, watch
  progress, and browse results (frame gallery, clips, charts).
- **`analyze_vod.py`** — the original command-line version, useful for
  scripting or batch-processing multiple VODs.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need `ffmpeg` on your PATH (used to cut highlight clips):
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: download from ffmpeg.org and add it to PATH

## Folder layout

```
vod_analyzer/
├── app.py             # web UI (recommended)
├── analyze_vod.py     # CLI
├── vod_analysis.py    # shared detection/analysis logic
├── templates/index.html
├── model/best.pt       # your trained weights
├── requirements.txt
└── web_data/            # created automatically (uploads + results)
```

## Web UI

```bash
python3 app.py
```

Then open **http://127.0.0.1:5050** in your browser. It only listens on
localhost — nothing is exposed to your network.

1. Drag in a VOD (mp4/mov/mkv/webm/avi), tune the settings if you want
   (sample rate, confidence threshold, clip padding, etc.), click **Run
   analysis**.
2. Watch the progress panel while it samples frames and runs detection.
3. Browse the results:
   - **Match summary** — highlight count, total enemy detections, avg confidence, VOD length
   - **Recording** — the source video, playable inline
   - **Timing** — enemy sightings over time
   - **Positioning** — heatmap of where enemies appeared on screen
   - **Frame captures** — gallery of annotated frames (bounding boxes drawn), click to enlarge
   - **Highlight clips** — auto-cut clips around each enemy-sighting cluster, playable inline
   - **Engagement windows** — raw table of every detected engagement

Each analysis run gets its own job folder under `web_data/output/<job_id>/`,
so results from multiple VODs don't overwrite each other.

## CLI

```bash
python3 analyze_vod.py --video path/to/your_recording.mp4 --model model/best.pt
```

Then open `output/dashboard.html` in a browser.

### Useful flags (both CLI and web UI expose these)

| Setting | Default | What it does |
|---|---|---|
| Sample rate (fps) | 4 | How many frames/sec to run detection on. Higher = more accurate timing, slower. |
| Confidence threshold | 0.4 | Minimum detection confidence. |
| Clip padding (s) | 2.0 | Seconds of padding before/after each highlight. |
| Merge gap (s) | 3.0 | Enemy sightings within this many seconds get merged into one clip/engagement. |
| Min detections / clip | 2 | Minimum detections needed for a moment to count as a highlight. |
| Max frame captures | 80 | Cap on how many annotated frames get saved to the gallery. |
| Extract clips | on | Turn off for a faster, stats-only pass (`--no-clips` on the CLI). |

## Notes / tuning tips

- Runtime scales with sample rate and video length; a 20-minute VOD at 4 fps
  sampling is a few thousand inference calls — expect it to take a while on
  CPU. `ultralytics` will use a GPU automatically if one's available.
- If clips feel choppy or cut off too early/late, raise merge gap and clip padding.
- If you're getting highlights on every stray detection, raise min detections
  or the confidence threshold.
- The heatmap and frame gallery only consider the `enemy` / `enemy head`
  classes — useful for spotting common angles/positions you had to react to.
- Each web UI run's files live under `web_data/`; delete that folder any
  time to clear out old uploads and results.
