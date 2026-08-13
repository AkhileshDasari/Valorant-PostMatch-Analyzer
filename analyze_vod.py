#!/usr/bin/env python3
"""
VALORANT VOD Post-Match Analyzer (CLI)
---------------------------------------
Runs a trained YOLO detection model over a recorded VOD (gameplay video)
to find enemy-sighting moments, cut highlight clips, plot where enemies
tended to appear on screen, and generate an HTML dashboard.

Offline only: reads a video file you already recorded and writes files
to disk. Does not attach to or interact with a running game.

Usage:
    python3 analyze_vod.py --video path/to/vod.mp4 --model model/best.pt

For a browser-based version with an upload UI and a frame-capture
gallery, run app.py instead: `python3 app.py`
"""

import argparse
import json
import os
import sys

import numpy as np

import vod_analysis as va


def parse_args():
    p = argparse.ArgumentParser(description="Analyze a VALORANT VOD with a trained YOLO model")
    p.add_argument("--video", required=True, help="Path to the recorded VOD file")
    p.add_argument("--model", default="model/best.pt", help="Path to YOLO .pt weights")
    p.add_argument("--sample-fps", type=float, default=4.0, help="Detection sampling rate (frames/sec)")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    p.add_argument("--clip-pad", type=float, default=2.0, help="Seconds of padding around each clip")
    p.add_argument("--merge-gap", type=float, default=3.0, help="Merge engagements within N seconds")
    p.add_argument("--min-clip-detections", type=int, default=2, help="Min detections to count as an engagement")
    p.add_argument("--output", default="output", help="Output directory")
    p.add_argument("--no-clips", action="store_true", help="Skip ffmpeg clip extraction")
    p.add_argument("--no-frames", action="store_true", help="Skip saving annotated frame captures")
    return p.parse_args()


def build_dashboard(detections, meta, engagements, clips, frame_captures, out_dir):
    enemy_dets = [d for d in detections if d["class_name"] in va.ENEMY_CLASSES]
    avg_conf = np.mean([d["conf"] for d in enemy_dets]) if enemy_dets else 0
    class_counts = {}
    for d in detections:
        class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

    clip_rows = "".join(
        f"<tr><td>#{c['index']}</td><td>{c['raw_start']:.1f}s-{c['raw_end']:.1f}s</td>"
        f"<td>{c['detections']}</td><td>clips/{c['file']}</td></tr>"
        for c in clips
    ) or '<tr><td colspan="4">No engagements met the threshold.</td></tr>'

    class_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(class_counts.items(), key=lambda x: -x[1])
    )

    frame_thumbs = "".join(
        f'<div class="thumb"><img src="frames/{f["path"]}"><div class="cap">{f["t"]:.1f}s &middot; {f["n_detections"]} det</div></div>'
        for f in frame_captures
    ) or "<p>No frame captures.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>VOD Analysis Dashboard</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f0d; color:#e4efe8; margin:0; padding:2rem; }}
h1 {{ font-size:1.6rem; margin-bottom:0.2rem; }}
.sub {{ color:#7f9184; margin-bottom:2rem; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap:1rem; margin-bottom:2rem; }}
.card {{ background:#121815; border:1px solid #223026; border-radius:10px; padding:1rem 1.2rem; }}
.card .num {{ font-size:1.8rem; font-weight:700; color:#4dff9e; }}
.card .lbl {{ color:#7f9184; font-size:0.85rem; }}
img {{ max-width:100%; border-radius:8px; margin-bottom:1.5rem; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:2rem; }}
th, td {{ text-align:left; padding:0.5rem 0.8rem; border-bottom:1px solid #223026; }}
th {{ color:#7f9184; font-weight:600; font-size:0.85rem; text-transform:uppercase; }}
section {{ margin-bottom:2.5rem; }}
h2 {{ font-size:1.1rem; color:#4dff9e; border-bottom:1px solid #223026; padding-bottom:0.4rem; }}
.thumbs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:0.6rem; }}
.thumb img {{ margin-bottom:0.2rem; border:1px solid #223026; }}
.thumb .cap {{ font-size:0.75rem; color:#7f9184; }}
</style></head><body>
<h1>VOD Analysis Dashboard</h1>
<div class="sub">{os.path.basename(meta['video_path'])} &middot; {meta['duration']:.0f}s &middot; {meta['width']}x{meta['height']}</div>
<div class="grid">
<div class="card"><div class="num">{len(engagements)}</div><div class="lbl">Highlight moments</div></div>
<div class="card"><div class="num">{len(enemy_dets)}</div><div class="lbl">Enemy detections</div></div>
<div class="card"><div class="num">{avg_conf*100:.0f}%</div><div class="lbl">Avg confidence</div></div>
<div class="card"><div class="num">{meta['duration']/60:.1f}m</div><div class="lbl">VOD length</div></div>
</div>
<section><h2>Enemy sightings over time</h2><img src="timeline.png"></section>
<section><h2>On-screen positioning heatmap</h2><img src="heatmap.png"></section>
<section><h2>Frame captures</h2><div class="thumbs">{frame_thumbs}</div></section>
<section><h2>Highlight clips</h2><table><tr><th>#</th><th>Time range</th><th>Detections</th><th>File</th></tr>{clip_rows}</table></section>
<section><h2>Detections by class</h2><table><tr><th>Class</th><th>Count</th></tr>{class_rows}</table></section>
</body></html>"""
    path = os.path.join(out_dir, "dashboard.html")
    with open(path, "w") as f:
        f.write(html)
    return path


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    frames_dir = None if args.no_frames else os.path.join(args.output, "frames")

    def progress(frac, msg):
        print(f"      [{frac*100:5.1f}%] {msg}")

    print("[1/4] Running detection...")
    detections, meta, frame_captures = va.run_detection(
        args.video, args.model, args.sample_fps, args.conf,
        frames_dir=frames_dir, progress_cb=progress,
    )
    print(f"      {len(detections)} raw detections, {len(frame_captures)} frame captures saved")

    with open(os.path.join(args.output, "detections.json"), "w") as f:
        json.dump({"meta": {k: v for k, v in meta.items() if k != "class_names"}, "detections": detections}, f, indent=2)

    print("[2/4] Finding engagements...")
    engagements = va.find_engagements(detections, args.merge_gap, args.min_clip_detections)
    print(f"      Found {len(engagements)} engagement window(s)")

    clips = []
    if not args.no_clips and engagements:
        print("[3/4] Extracting highlight clips...")
        clips = va.extract_clips(args.video, engagements, args.clip_pad, args.output, progress_cb=progress)

    print("[4/4] Building charts + dashboard...")
    va.build_heatmap(detections, os.path.join(args.output, "heatmap.png"))
    va.build_timeline(detections, meta["duration"], os.path.join(args.output, "timeline.png"))
    dashboard_path = build_dashboard(detections, meta, engagements, clips, frame_captures, args.output)

    print(f"\nDone. Dashboard: {dashboard_path}")


if __name__ == "__main__":
    main()
