#!/usr/bin/env python3
"""
app.py — VALORANT VOD Analyzer, local web UI
----------------------------------------------
Runs a small local web server (Flask) so you can upload a recorded VOD
in a browser, watch it get analyzed, and browse the results: frame
captures, highlight clips, positioning heatmap, and a sightings timeline.

This is a LOCAL, offline tool:
  - It only listens on localhost (127.0.0.1) by default.
  - It reads a video file you upload and writes results to disk.
  - It never attaches to, reads memory from, or interacts with a running
    game process in any way.

Run:
    python3 app.py
Then open:
    http://127.0.0.1:5050
"""

import json
import os
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory

import vod_analysis as va

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "web_data", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "web_data", "output")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi",".flv"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024  # 8GB

JOBS = {}
JOBS_LOCK = threading.Lock()


def set_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def get_job(job_id):
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {}))


def run_job(job_id, video_path, cfg):
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)

    def progress(frac, msg):
        set_job(job_id, progress=frac, message=msg)

    try:
        set_job(job_id, status="running", progress=0.0, message="Starting...")
        detections, meta, frame_captures = va.run_detection(
            video_path, MODEL_PATH,
            sample_fps=cfg["sample_fps"], conf=cfg["conf"],
            frames_dir=frames_dir, max_captures=cfg["max_captures"],
            progress_cb=progress,
        )

        progress(0.85, "Finding engagements...")
        engagements = va.find_engagements(detections, cfg["merge_gap"], cfg["min_detections"])

        clips = []
        if engagements and cfg["make_clips"]:
            clips = va.extract_clips(video_path, engagements, cfg["clip_pad"], out_dir, progress_cb=progress)

        # Generate the full annotated detection video (optional)
        annotated_video_url = None
        if cfg.get("make_annotated_video", True):
            def annotated_progress(frac, msg):
                # map 0..1 -> 0.96..1.0 progress slice
                set_job(job_id, progress=0.96 + frac * 0.03,
                        message=f"[Annotated Video] {msg}")

            progress(0.96, "Generating annotated detection video...")
            ann_fname = va.generate_annotated_video(
                video_path, MODEL_PATH, out_dir,
                conf=cfg["conf"], progress_cb=annotated_progress,
            )
            if ann_fname:
                annotated_video_url = f"/media/{job_id}/{ann_fname}"

        progress(0.99, "Building charts...")
        va.build_heatmap(detections, os.path.join(out_dir, "heatmap.png"))
        va.build_timeline(detections, meta["duration"], os.path.join(out_dir, "timeline.png"))

        enemy_dets = [d for d in detections if d["class_name"] in va.ENEMY_CLASSES]
        class_counts = {}
        for d in detections:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
        avg_conf = sum(d["conf"] for d in enemy_dets) / len(enemy_dets) if enemy_dets else 0

        result = {
            "meta": {
                "video_name": os.path.basename(video_path),
                "duration": meta["duration"],
                "width": meta["width"],
                "height": meta["height"],
            },
            "stats": {
                "n_engagements": len(engagements),
                "n_enemy_detections": len(enemy_dets),
                "avg_confidence": avg_conf,
                "class_counts": class_counts,
            },
            "engagements": [
                {"index": i + 1, "start": s, "end": e, "detections": c}
                for i, (s, e, c) in enumerate(engagements)
            ],
            "clips": clips,
            "frame_captures": sorted(frame_captures, key=lambda f: f["t"]),
            "heatmap_url": f"/media/{job_id}/heatmap.png",
            "timeline_url": f"/media/{job_id}/timeline.png",
            "video_url": f"/media/{job_id}/source{os.path.splitext(video_path)[1]}",
            "annotated_video_url": annotated_video_url,
        }
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)

        set_job(job_id, status="done", progress=1.0, message="Complete", result=result)
    except Exception as e:
        set_job(job_id, status="error", message=str(e))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type {ext}. Use mp4/mov/mkv/webm/avi."}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": f"Model not found at {MODEL_PATH}"}), 500

    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    video_path = os.path.join(job_upload_dir, f"source{ext}")
    file.save(video_path)

    cfg = {
        "sample_fps": float(request.form.get("sample_fps", 4.0)),
        "conf": float(request.form.get("conf", 0.4)),
        "clip_pad": float(request.form.get("clip_pad", 2.0)),
        "merge_gap": float(request.form.get("merge_gap", 3.0)),
        "min_detections": int(request.form.get("min_detections", 2)),
        "max_captures": int(request.form.get("max_captures", 80)),
        "make_clips": request.form.get("make_clips", "true") == "true",
        "make_annotated_video": request.form.get("make_annotated_video", "true") == "true",
    }

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "progress": 0.0, "message": "Queued", "created": time.time()}

    # Also expose the source video for the <video> player via /media
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    t = threading.Thread(target=run_job, args=(job_id, video_path, cfg), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({k: v for k, v in job.items() if k != "result"})


@app.route("/api/result/<job_id>")
def result(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    if job.get("status") != "done":
        return jsonify({"error": "Job not finished"}), 409
    return jsonify(job["result"])


def _safe_job_dir(job_id):
    # job_id comes from uuid4().hex[:12] server-side, but validate defensively
    # against path traversal before joining.
    if not job_id.isalnum():
        return None
    return os.path.join(OUTPUT_DIR, job_id)


@app.route("/media/<job_id>/<path:filename>")
def media(job_id, filename):
    job_dir = _safe_job_dir(job_id)
    if not job_dir or not os.path.isdir(job_dir):
        return "Not found", 404
    if filename.startswith("source"):
        # source video lives in the upload dir, not the output dir
        upload_dir = os.path.join(UPLOAD_DIR, job_id)
        return send_from_directory(upload_dir, filename)
    if filename.startswith("clips/"):
        return send_from_directory(os.path.join(job_dir, "clips"), filename[len("clips/"):])
    if filename.startswith("frames/"):
        return send_from_directory(os.path.join(job_dir, "frames"), filename[len("frames/"):])
    return send_from_directory(job_dir, filename)


if __name__ == "__main__":
    print("running at: http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
