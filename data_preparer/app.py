"""
4DHOI Data Preparer - Upload videos, split scenes, annotate point prompts.

Combines video upload + scene splitting + point annotation into a single app.
No authentication, no file locks, no upload_records.json.
Works directly with a local data directory.
"""
import base64
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = APP_DIR / "data"
MAX_SEGMENT_SECONDS = 20
MAX_SELECT_COUNT = 3
ALLOWED_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "webm", "flv"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 500  # 500MB
app.process_tokens = {}

# Configurable data directory
DATA_DIR = DEFAULT_DATA_DIR

# Session store for upload tokens
SESSION_STORE_DIR = Path(tempfile.gettempdir()) / "data_preparer_sessions"
SESSION_STORE_DIR.mkdir(parents=True, exist_ok=True)


# ========== Utility Functions ==========

def _safe_folder_name(name, max_len=60):
    name = re.sub(r"[^\w\-_.]", "_", name.strip())
    return name[:max_len] if name else "video"


def _sec_to_str(s):
    h = int(s) // 3600
    m = (int(s) % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def _get_video_duration(path):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "json", str(path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            return float(d.get("format", {}).get("duration", 0) or 0)
    except Exception:
        pass
    return 0


def _probe_video_fps(video_path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "json", str(video_path)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            info = json.loads(r.stdout)
            streams = info.get("streams", [])
            if streams:
                for key in ("avg_frame_rate", "r_frame_rate"):
                    val = streams[0].get(key, "")
                    if val and "/" in val:
                        num, den = val.split("/")
                        if float(den) > 0:
                            fps = float(num) / float(den)
                            if 1 < fps < 240:
                                return fps
    except Exception:
        pass
    return 30.0


def _decode_data_url(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def _extract_segment(src, start_sec, duration_sec, out_path, for_preview=False):
    """Re-encode video segment with ffmpeg."""
    try:
        if duration_sec <= 0:
            return False
        preset = "veryfast" if for_preview else "fast"
        crf = "28" if for_preview else "23"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{float(start_sec):.3f}",
            "-i", str(src),
            "-t", f"{float(duration_sec):.3f}",
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            "-reset_timestamps", "1",
            str(out_path),
        ]
        timeout = 180 if for_preview else 600
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024
    except Exception as e:
        print(f"[FFmpeg Error] {e}")
        return False


def _detect_and_split_scenes(video_path, max_sec=MAX_SEGMENT_SECONDS):
    """Detect scenes with PySceneDetect, split long segments."""
    try:
        from scenedetect import detect, ContentDetector
    except ImportError:
        return [{"error": "scenedetect not installed: pip install scenedetect[opencv]"}]

    scene_list = detect(str(video_path), ContentDetector(threshold=27.0))
    segments = []

    def _to_sec(tc):
        try:
            return float(tc.get_seconds())
        except (AttributeError, TypeError):
            return float(tc)

    for start_tc, end_tc in scene_list:
        start_sec = _to_sec(start_tc)
        end_sec = _to_sec(end_tc)
        dur = end_sec - start_sec
        if dur <= 0:
            continue
        if dur <= max_sec:
            segments.append({
                "start": round(start_sec, 2), "end": round(end_sec, 2),
                "duration": round(dur, 2),
                "start_str": _sec_to_str(start_sec), "end_str": _sec_to_str(end_sec),
            })
        else:
            t = start_sec
            while t < end_sec:
                e = min(t + max_sec, end_sec)
                segments.append({
                    "start": round(t, 2), "end": round(e, 2),
                    "duration": round(e - t, 2),
                    "start_str": _sec_to_str(t), "end_str": _sec_to_str(e),
                })
                t = e

    if not segments:
        total_dur = _get_video_duration(video_path)
        if total_dur <= 0:
            total_dur = max_sec
        t = 0.0
        while t < total_dur:
            e = min(t + max_sec, total_dur)
            segments.append({
                "start": round(t, 2), "end": round(e, 2),
                "duration": round(e - t, 2),
                "start_str": _sec_to_str(t), "end_str": _sec_to_str(e),
            })
            t = e

    return segments


def _scan_sessions(data_dir):
    """Scan data directory for annotation sessions."""
    sessions = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return sessions
    for entry in sorted(data_path.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "video.mp4").exists():
            sessions.append(_build_session_info(entry, ""))
            continue
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and (sub / "video.mp4").exists():
                sessions.append(_build_session_info(sub, entry.name))
    return sessions


def _build_session_info(session_dir, category):
    """Build session info dict."""
    has_points = (session_dir / "points.json").exists()
    has_select = (session_dir / "select_id.json").exists()
    return {
        "name": session_dir.name,
        "path": str(session_dir),
        "category": category,
        "has_video": (session_dir / "video.mp4").exists(),
        "has_points": has_points,
        "has_select_id": has_select,
        "annotated": has_points and has_select,
    }


def _token_info_path(token):
    safe = re.sub(r"[^a-fA-F0-9]", "", str(token or ""))
    return SESSION_STORE_DIR / f"{safe}.json"


def _persist_token_info(token, info):
    try:
        p = _token_info_path(token)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        tmp.replace(p)
    except Exception:
        pass


def _load_token_info(token):
    info = app.process_tokens.get(token)
    if info:
        return info
    try:
        p = _token_info_path(token)
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            info = json.load(f)
        if isinstance(info, dict) and Path(info.get("work_dir", "")).exists():
            app.process_tokens[token] = info
            return info
    except Exception:
        pass
    return None


def _delete_token_info(token):
    app.process_tokens.pop(token, None)
    try:
        p = _token_info_path(token)
        if p.exists():
            p.unlink()
    except Exception:
        pass


# ========== Routes: Pages ==========

@app.route("/")
def index():
    return render_template("index.html")


# ========== Routes: Upload & Scene Split ==========

@app.get("/api/object_categories")
def api_object_categories():
    """List existing object categories from data directory."""
    cats = set()
    if DATA_DIR.exists():
        for entry in DATA_DIR.iterdir():
            if entry.is_dir():
                cats.add(entry.name)
    return jsonify({"categories": sorted(cats)})


@app.post("/api/process")
def api_process():
    """Upload video, detect scenes, return segments for preview."""
    object_category = (request.form.get("object_category") or "").strip()
    if not object_category:
        return jsonify({"ok": False, "error": "Object category is required"}), 400

    video_file = request.files.get("video_file")
    if not video_file or not video_file.filename:
        return jsonify({"ok": False, "error": "Please upload a video file"}), 400

    ext = video_file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": f"Unsupported format: .{ext}"}), 400

    work_dir = Path(tempfile.mkdtemp(prefix="data_preparer_"))
    try:
        video_path = work_dir / f"upload.{ext}"
        video_file.save(video_path)

        segments = _detect_and_split_scenes(video_path)
        if segments and "error" in segments[0]:
            return jsonify({"ok": False, "error": segments[0]["error"]}), 500

        token = secrets.token_hex(16)
        info = {
            "work_dir": str(work_dir),
            "video_path": str(video_path),
            "object_category": object_category,
        }
        app.process_tokens[token] = info
        _persist_token_info(token, info)

        for i, seg in enumerate(segments):
            seg_path = work_dir / f"segment_{i}.mp4"
            _extract_segment(video_path, float(seg["start"]), float(seg["duration"]),
                             seg_path, for_preview=True)
            seg["url"] = f"api/segment/{token}/{i}"

        return jsonify({"ok": True, "token": token, "segments": segments})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/segment/<token>/<int:idx>")
def api_serve_segment(token, idx):
    """Serve a preview segment video."""
    info = _load_token_info(token)
    if not info:
        return jsonify({"error": "Invalid or expired token"}), 404
    seg_path = Path(info["work_dir"]) / f"segment_{idx}.mp4"
    if not seg_path.exists() or seg_path.stat().st_size == 0:
        return jsonify({"error": "Segment not found"}), 404
    return send_file(seg_path, mimetype="video/mp4", conditional=True, max_age=0)


@app.post("/api/save_segments")
def api_save_segments():
    """Save selected segments to the data directory."""
    data = request.get_json(silent=True) or {}
    token = data.get("token")
    selected = data.get("selected", [])
    segments_data = data.get("segments", [])

    if not token or not isinstance(selected, list) or len(selected) == 0:
        return jsonify({"ok": False, "error": "Missing token or selected segments"}), 400
    if len(selected) > MAX_SELECT_COUNT:
        return jsonify({"ok": False, "error": f"Maximum {MAX_SELECT_COUNT} segments"}), 400

    info = _load_token_info(token)
    if not info:
        return jsonify({"ok": False, "error": "Token expired, please re-upload"}), 400

    video_path = Path(info["video_path"])
    object_category = info["object_category"]

    if not video_path.exists():
        return jsonify({"ok": False, "error": "Temporary video expired"}), 400

    saved = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cat_dir = DATA_DIR / _safe_folder_name(object_category, 30)

    for idx_in_selection, seg_idx in enumerate(selected):
        if not isinstance(seg_idx, int) or seg_idx < 0 or seg_idx >= len(segments_data):
            continue
        seg = segments_data[seg_idx]
        if not seg or "start" not in seg:
            continue

        folder_name = f"{ts}_{secrets.token_hex(4)}"
        if idx_in_selection > 0:
            folder_name += f"_part{idx_in_selection + 1:02d}"

        session_dir = cat_dir / folder_name
        session_dir.mkdir(parents=True, exist_ok=True)

        out_video = session_dir / "video.mp4"
        start_sec = float(seg["start"])
        end_sec = float(seg["end"])
        duration_sec = end_sec - start_sec

        if not _extract_segment(video_path, start_sec, duration_sec, out_video):
            return jsonify({"ok": False, "error": f"Segment {idx_in_selection + 1} export failed"}), 500

        saved.append({"path": str(session_dir), "name": folder_name, "category": object_category})

    # Cleanup temp files
    _delete_token_info(token)
    try:
        work_dir = Path(info["work_dir"])
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    return jsonify({"ok": True, "saved": saved, "count": len(saved)})


# ========== Routes: Session Browsing ==========

@app.get("/api/sessions")
def api_sessions():
    """List all sessions in data directory."""
    q = (request.args.get("q") or "").strip().lower()
    sessions = _scan_sessions(DATA_DIR)
    if q:
        sessions = [s for s in sessions if q in s["name"].lower() or q in s["category"].lower()]
    return jsonify({"sessions": sessions, "total": len(sessions)})


# ========== Routes: Video Streaming ==========

@app.get("/api/video/<path:session_path>")
def api_serve_video(session_path):
    """Stream a session video with Range support."""
    # session_path comes from Flask <path:> converter, slashes are preserved
    video_path = Path("/") / session_path / "video.mp4"
    if not video_path.exists():
        # Try without leading slash (relative path)
        video_path = Path(session_path) / "video.mp4"
    if not video_path.exists():
        return "Video not found", 404

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        # Parse range
        match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else min(start + 1024 * 1024, file_size - 1)
            end = min(end, file_size - 1)
            length = end - start + 1

            with open(video_path, "rb") as f:
                f.seek(start)
                data = f.read(length)

            resp = app.response_class(data, status=206, mimetype="video/mp4")
            resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Content-Length"] = length
            return resp

    return send_file(video_path, mimetype="video/mp4", conditional=True)


# ========== Routes: Point Annotation ==========

@app.post("/api/save_annotation")
def api_save_annotation():
    """Save point annotation (points.json + select_id.json) for a session."""
    data = request.get_json(silent=True) or {}
    session_path = data.get("session_path")
    if not session_path:
        return jsonify({"ok": False, "error": "session_path is required"}), 400

    session_dir = Path(session_path)
    if not session_dir.exists() or not (session_dir / "video.mp4").exists():
        return jsonify({"ok": False, "error": "Invalid session path"}), 400

    human_points = data.get("human_points", [])
    object_points = data.get("object_points", [])
    start_timestamp = data.get("start_timestamp")
    selected_timestamp = data.get("selected_timestamp")
    object_category = data.get("object_category", session_dir.parent.name)

    if start_timestamp is None or selected_timestamp is None:
        return jsonify({"ok": False, "error": "start_timestamp and selected_timestamp are required"}), 400

    start_ts = float(start_timestamp)
    sel_ts = float(selected_timestamp)

    if sel_ts < start_ts:
        return jsonify({"ok": False, "error": "selected_timestamp must be >= start_timestamp"}), 400

    # Probe FPS
    fps = _probe_video_fps(session_dir / "video.mp4")
    start_id = int(start_ts * fps)
    select_id = int(sel_ts * fps) - start_id

    # Save points.json
    points_data = {"human_points": human_points, "object_points": object_points}
    with open(session_dir / "points.json", "w", encoding="utf-8") as f:
        json.dump(points_data, f, indent=2)

    # Save select_id.json
    select_data = {
        "select_id": max(0, select_id),
        "start_id": start_id,
        "object_name": object_category,
    }
    with open(session_dir / "select_id.json", "w", encoding="utf-8") as f:
        json.dump(select_data, f, indent=2)

    return jsonify({
        "ok": True,
        "points_path": str(session_dir / "points.json"),
        "select_path": str(session_dir / "select_id.json"),
        "select_id": select_data["select_id"],
        "start_id": start_id,
        "fps": fps,
    })


@app.post("/api/capture_frame")
def api_capture_frame():
    """Save a captured frame image."""
    data = request.get_json(silent=True) or {}
    session_path = data.get("session_path")
    image_data = data.get("image")
    timestamp = data.get("timestamp", 0)

    if not session_path or not image_data:
        return jsonify({"ok": False, "error": "session_path and image are required"}), 400

    session_dir = Path(session_path)
    captures_dir = session_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    img_bytes = _decode_data_url(image_data)
    fname = f"frame_{float(timestamp):.3f}.png"
    out_path = captures_dir / fname
    with open(out_path, "wb") as f:
        f.write(img_bytes)

    return jsonify({"ok": True, "path": str(out_path)})


# ========== Main ==========

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4DHOI Data Preparer")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory for session data (default: ./data)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5020)
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir).resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")

    app.run(host=args.host, port=args.port, debug=True)
