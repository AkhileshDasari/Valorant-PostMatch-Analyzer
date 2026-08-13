"""
vod_analysis.py
----------------
Core, reusable VOD analysis logic: run a YOLO model over a video file,
cluster enemy sightings into highlight-worthy "engagements", extract
clips, save annotated frame captures, and build chart images.

This module does file I/O only (reads a video file, writes images/clips/
json to an output directory). It never touches a running game process.

Used by both analyze_vod.py (CLI) and app.py (local web UI).
"""

import os
import subprocess
from collections import defaultdict

import cv2
import numpy as np

# BGR colors (OpenCV) per class, used when drawing boxes on frame captures
CLASS_COLORS = {
    "enemy": (0, 60, 255),        # red
    "enemy head": (0, 165, 255),  # orange
    "teammate": (255, 210, 60),   # cyan-ish
    "crosshair": (200, 200, 200), # light gray
}
ENEMY_CLASSES = ("enemy", "enemy head")


def run_detection(video_path, model_path, sample_fps=4.0, conf=0.4,
                   frames_dir=None, max_captures=80, capture_min_gap=1.5,
                   progress_cb=None):
    """Sample the video at sample_fps, run YOLO detection on each sampled frame,
    and (optionally) save annotated JPEG captures of frames that contain an
    enemy detection, throttled to at most one every `capture_min_gap` seconds.

    progress_cb(fraction: float, message: str) is called periodically if given.

    Returns (detections, meta, frame_captures)
    """
    from ultralytics import YOLO

    if progress_cb:
        progress_cb(0.0, "Loading model...")
    model = YOLO(model_path)
    class_names = model.names

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / native_fps if native_fps else 0

    frame_step = max(1, round(native_fps / sample_fps))

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    detections = []
    frame_captures = []
    last_capture_t = -1e9
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            t = frame_idx / native_fps
            results = model.predict(frame, conf=conf, verbose=False)[0]

            frame_dets = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = {
                    "t": t,
                    "class_id": cls_id,
                    "class_name": class_names[cls_id],
                    "conf": confidence,
                    "cx": cx / width,
                    "cy": cy / height,
                    "bbox": [x1, y1, x2, y2],
                }
                detections.append(d)
                frame_dets.append(d)

            has_enemy = any(d["class_name"] in ENEMY_CLASSES for d in frame_dets)
            if (frames_dir and has_enemy and len(frame_captures) < max_captures
                    and (t - last_capture_t) >= capture_min_gap):
                capture_path = _save_annotated_frame(frame, frame_dets, frames_dir, frame_idx, t)
                frame_captures.append({
                    "path": os.path.basename(capture_path),
                    "t": t,
                    "n_detections": len(frame_dets),
                    "classes": sorted(set(d["class_name"] for d in frame_dets)),
                    "max_conf": max((d["conf"] for d in frame_dets), default=0),
                })
                last_capture_t = t

            if progress_cb and duration > 0:
                progress_cb(min(0.85, 0.05 + 0.80 * (t / duration)), f"Analyzing {t:.0f}s / {duration:.0f}s")

        frame_idx += 1

    cap.release()

    meta = {
        "video_path": video_path,
        "duration": duration,
        "width": width,
        "height": height,
        "native_fps": native_fps,
        "sample_fps": sample_fps,
        "class_names": class_names,
    }
    return detections, meta, frame_captures


def _save_annotated_frame(frame, frame_dets, frames_dir, frame_idx, t):
    img = frame.copy()
    for d in frame_dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        color = CLASS_COLORS.get(d["class_name"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class_name']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

    # Downscale large frames for a lighter gallery
    h, w = img.shape[:2]
    if w > 960:
        scale = 960 / w
        img = cv2.resize(img, (960, int(h * scale)))

    fname = f"frame_{frame_idx:07d}_{t:.1f}s.jpg"
    path = os.path.join(frames_dir, fname)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def find_engagements(detections, merge_gap=3.0, min_detections=2, enemy_classes=ENEMY_CLASSES):
    """Group enemy detections in time into 'engagement' windows (highlight-worthy moments)."""
    enemy_times = sorted(d["t"] for d in detections if d["class_name"] in enemy_classes)
    if not enemy_times:
        return []

    windows = []
    start = enemy_times[0]
    prev = enemy_times[0]
    count = 1
    for t in enemy_times[1:]:
        if t - prev <= merge_gap:
            prev = t
            count += 1
        else:
            windows.append((start, prev, count))
            start, prev, count = t, t, 1
    windows.append((start, prev, count))

    return [w for w in windows if w[2] >= min_detections]


def extract_clips(video_path, engagements, pad, out_dir, progress_cb=None):
    """Use ffmpeg to cut a clip for each engagement window, with padding."""
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    clips = []

    n = len(engagements)
    for i, (start, end, count) in enumerate(engagements, 1):
        cs = max(0, start - pad)
        dur = (end - start) + 2 * pad
        fname = f"highlight_{i:02d}_{int(cs)}s.mp4"
        out_path = os.path.join(clips_dir, fname)
        cmd = [
            "ffmpeg", "-y", "-ss", f"{cs:.2f}", "-i", video_path,
            "-t", f"{dur:.2f}", "-c:v", "libx264", "-c:a", "aac",
            "-preset", "veryfast", "-loglevel", "error", out_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
        except FileNotFoundError:
            import warnings
            warnings.warn("ffmpeg not found — highlight clip extraction skipped.")
            success = False
        if success:
            clips.append({
                "file": fname, "index": i, "start": cs, "end": cs + dur,
                "detections": count, "raw_start": start, "raw_end": end,
            })
        if progress_cb:
            progress_cb(0.85 + 0.10 * (i / max(1, n)), f"Cutting clip {i}/{n}")
    return clips


def generate_annotated_video(video_path, model_path, out_dir, conf=0.4,
                              progress_cb=None):
    """
    Process the full VOD with YOLO detection and render an annotated video
    with a tactical HUD overlay (matching the reference image style):
      - Top banner: Total Enemy Detections
      - Top-left SYSTEM STATS glass panel (Live FPS, Active Enemies, Resolution)
      - Bounding boxes with class labels and per-detection IDs
      - Top-right inset: 'Last Detected' enemy crop thumbnail
      - Horizontal sci-fi scanline
    Writes a raw OpenCV-annotated video, then re-muxes with original audio via ffmpeg.
    Returns the filename of the final annotated mp4, or None on failure.
    """
    from ultralytics import YOLO

    if progress_cb:
        progress_cb(0.0, "Loading model for annotated video...")

    model = YOLO(model_path)
    class_names = model.names

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "_annotated_raw.mp4")
    final_fname = "annotated_detection.mp4"
    final_path = os.path.join(out_dir, final_fname)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, native_fps, (width, height))

    # HUD configuration
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    MONO = cv2.FONT_HERSHEY_DUPLEX
    SCAN_Y_FRAC = 0.72          # scanline vertical position (fraction of height)
    INSET_W, INSET_H = 200, 112  # top-right crop inset dimensions

    total_enemy_count = 0
    last_detected_crop = None   # BGR crop of last detected enemy
    det_id_counter = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / native_fps
        results = model.predict(frame, conf=conf, verbose=False)[0]

        frame_dets = []
        active_enemies = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_names[cls_id]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            det_id_counter += 1
            det_id = det_id_counter
            frame_dets.append({
                "class_name": class_name, "conf": confidence,
                "bbox": (x1, y1, x2, y2), "id": det_id,
            })

            is_enemy = class_name in ENEMY_CLASSES
            if is_enemy:
                total_enemy_count += 1
                active_enemies += 1
                # Capture crop for inset thumbnail
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size > 0:
                    last_detected_crop = crop.copy()

        # ── Draw bounding boxes & labels ──────────────────────────────────
        for d in frame_dets:
            x1, y1, x2, y2 = d["bbox"]
            color = CLASS_COLORS.get(d["class_name"], (200, 200, 200))
            is_enemy = d["class_name"] in ENEMY_CLASSES

            # Box (thicker for enemies)
            thickness = 2 if is_enemy else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # ID badge above box
            id_label = f"ID: {d['id']}"
            (tw, th), _ = cv2.getTextSize(id_label, MONO, 0.45, 1)
            badge_x, badge_y = x1, max(th + 6, y1 - 2)
            cv2.rectangle(frame, (badge_x, badge_y - th - 6),
                          (badge_x + tw + 8, badge_y), color, -1)
            cv2.putText(frame, id_label, (badge_x + 4, badge_y - 3),
                        MONO, 0.45, (8, 8, 8), 1, cv2.LINE_AA)

            # Confidence label at bottom of box
            conf_label = f"{d['class_name']} {d['conf']:.2f}"
            cv2.putText(frame, conf_label,
                        (x1 + 3, min(height - 4, y2 + 14)),
                        FONT, 0.42, color, 1, cv2.LINE_AA)

        # ── Total Enemy Count banner ───────────────────────────────────────
        banner_text = f"Total Enemy Count: {total_enemy_count}"
        (bw, bh), _ = cv2.getTextSize(banner_text, MONO, 0.8, 2)
        # Dark semi-transparent background strip
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (bw + 24, bh + 20), (5, 10, 8), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, banner_text, (12, bh + 8),
                    MONO, 0.8, (77, 255, 158), 2, cv2.LINE_AA)

        # ── SYSTEM STATS glass panel (top-left below banner) ───────────────
        stats_lines = [
            "SYSTEM STATS",
            f"Live FPS: {int(native_fps)}",
            f"Active Enemies: {active_enemies}",
            f"Resolution: {width}x{height}",
        ]
        sx, sy = 12, bh + 30
        panel_h = len(stats_lines) * 22 + 10
        panel_w = 200
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (sx - 4, sy - 6),
                      (sx + panel_w, sy + panel_h), (10, 18, 14), -1)
        cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (sx - 4, sy - 6),
                      (sx + panel_w, sy + panel_h), (50, 90, 60), 1)
        for li, line in enumerate(stats_lines):
            color_s = (77, 255, 158) if li == 0 else (  # header green
                       (255, 255, 80) if li == 2 else   # active enemies amber
                       (180, 220, 190))
            scale_s = 0.50 if li == 0 else 0.42
            thick_s = 1
            cv2.putText(frame, line, (sx, sy + li * 22 + 14),
                        FONT, scale_s, color_s, thick_s, cv2.LINE_AA)

        # ── Horizontal sci-fi scanline ─────────────────────────────────────
        scan_y = int(height * SCAN_Y_FRAC)
        scan_color = (60, 230, 120)
        cv2.line(frame, (0, scan_y), (width, scan_y), scan_color, 1)
        # Faint scanline glow
        overlay3 = frame.copy()
        cv2.line(overlay3, (0, scan_y - 2), (width, scan_y + 2),
                 scan_color, 4)
        cv2.addWeighted(overlay3, 0.12, frame, 0.88, 0, frame)

        # ── Top-right "Last Detected" inset ───────────────────────────────
        if last_detected_crop is not None:
            inset_x = width - INSET_W - 10
            inset_y = 10
            try:
                inset = cv2.resize(last_detected_crop, (INSET_W, INSET_H))
                # Border + label background
                cv2.rectangle(frame,
                               (inset_x - 2, inset_y - 20),
                               (inset_x + INSET_W + 2, inset_y + INSET_H + 2),
                               (0, 200, 80), 2)
                label_bg_overlay = frame.copy()
                cv2.rectangle(label_bg_overlay,
                               (inset_x - 2, inset_y - 20),
                               (inset_x + INSET_W + 2, inset_y),
                               (0, 80, 30), -1)
                cv2.addWeighted(label_bg_overlay, 0.8, frame, 0.2, 0, frame)
                cv2.putText(frame, "Last Counted",
                            (inset_x + 4, inset_y - 5),
                            FONT, 0.44, (77, 255, 158), 1, cv2.LINE_AA)
                # Paste inset
                frame[inset_y:inset_y + INSET_H,
                      inset_x:inset_x + INSET_W] = inset
            except Exception:
                pass

        writer.write(frame)
        frame_idx += 1

        if progress_cb and total_frames > 0 and frame_idx % 30 == 0:
            frac = frame_idx / total_frames
            progress_cb(frac,
                        f"Annotating frame {frame_idx}/{total_frames} "
                        f"({t:.0f}s)")

    writer.release()
    cap.release()

    # Re-mux with original audio (copy streams, no re-encode)
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_path,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", "-shortest",
        "-loglevel", "error",
        final_path,
    ]
    try:
        mux_result = subprocess.run(cmd, capture_output=True, text=True)
        mux_ok = mux_result.returncode == 0
    except FileNotFoundError:
        import warnings
        warnings.warn("ffmpeg not found — annotated video will have no audio.")
        mux_ok = False

    # Clean up raw file
    try:
        if mux_ok:
            os.remove(raw_path)
    except OSError:
        pass

    if mux_ok:
        return final_fname
    # Fallback: rename raw OpenCV video (no audio) as final output
    import shutil
    try:
        shutil.move(raw_path, final_path)
        return final_fname
    except Exception:
        return None


def build_heatmap(detections, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enemy_pts = [(d["cx"], d["cy"]) for d in detections if d["class_name"] in ENEMY_CLASSES]

    plt.rcParams.update({
        "figure.facecolor": "#10140f", "axes.facecolor": "#10140f",
        "text.color": "#cfe8d8", "axes.edgecolor": "#2a3b2e",
        "xtick.color": "#7f9184", "ytick.color": "#7f9184",
    })
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if enemy_pts:
        xs, ys = zip(*enemy_pts)
        h = ax.hist2d(xs, ys, bins=40, range=[[0, 1], [0, 1]], cmap="YlOrRd")
        cb = fig.colorbar(h[3], ax=ax)
        cb.ax.yaxis.set_tick_params(color="#7f9184")
        cb.outline.set_edgecolor("#2a3b2e")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title("Enemy on-screen position density", color="#e4efe8")
    ax.set_xlabel("normalized x")
    ax.set_ylabel("normalized y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_timeline(detections, duration, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_size = max(2, int(duration / 80)) if duration else 10
    n_bins = max(1, int(duration // bin_size) + 1) if duration else 1
    counts = defaultdict(int)
    for d in detections:
        if d["class_name"] in ENEMY_CLASSES:
            counts[int(d["t"] // bin_size)] += 1

    xs = [i * bin_size for i in range(n_bins)]
    ys = [counts.get(i, 0) for i in range(n_bins)]

    plt.rcParams.update({
        "figure.facecolor": "#10140f", "axes.facecolor": "#10140f",
        "text.color": "#cfe8d8", "axes.edgecolor": "#2a3b2e",
        "xtick.color": "#7f9184", "ytick.color": "#7f9184",
    })
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(xs, ys, width=bin_size * 0.9, color="#4dff9e")
    ax.set_title("Enemy sightings over time", color="#e4efe8")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"detections / {bin_size}s")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
