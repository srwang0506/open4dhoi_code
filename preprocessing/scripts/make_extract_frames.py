# -*- coding: utf-8 -*-
"""
从 upload_records.json 读取待处理数据，用 ffmpeg 提取 video.mp4 的所有帧到 frames/ 目录。

前提条件：
- video.mp4 已经从原始视频的 start_id 帧开始截取
- 所以 video.mp4 的第 0 帧 = 原始视频的第 start_id 帧

输出规则：
- frames/00000.jpg 对应 video.mp4 的第 0 帧
- frames/00001.jpg 对应 video.mp4 的第 1 帧
- frames/ 帧数 = video.mp4 帧数

注意：
- 不使用 start_id 过滤，因为 video.mp4 已经是截取后的
- select_id 是相对于 start_id 的偏移（即 frames/ 中的索引）
"""

import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path

from tqdm import tqdm

from task_status import TaskProgress

PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_RECORDS_PATH = PROJECT_DIR / "upload_records.json"
SELECT_ID_FILENAME = "select_id.json"


class ExtractFramesError(Exception):
    """提取帧过程中发生错误，立即终止"""


def _resolve_path(p: str) -> Path:
    if not isinstance(p, str) or not p:
        return Path("")
    if p.startswith("./"):
        return (PROJECT_DIR / p[2:]).resolve()
    if p.startswith("tiktok_data/"):
        return (PROJECT_DIR / p).resolve()
    return Path(p).expanduser().resolve()


def _load_records() -> list[dict]:
    if not UPLOAD_RECORDS_PATH.exists():
        raise ExtractFramesError(f"upload_records.json 不存在: {UPLOAD_RECORDS_PATH}")
    with UPLOAD_RECORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ExtractFramesError("upload_records.json 应为 list")
    return data


def _load_select_json(session_dir: Path) -> dict | None:
    p = session_dir / SELECT_ID_FILENAME
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _load_select_id(session_dir: Path) -> int | None:
    """是否有有效的 select_id（用于判断是否跳过）。"""
    data = _load_select_json(session_dir)
    if not data:
        return None
    sid = data.get("select_id")
    if sid is None:
        return None
    try:
        return int(sid)
    except (TypeError, ValueError):
        return None


def _start_frame_id_for_extract(session_dir: Path) -> int:
    """从哪一帧开始抽帧：优先 start_id，否则 0。"""
    data = _load_select_json(session_dir)
    if not data:
        return 0
    v = data.get("start_id")
    if v is None:
        return 0
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 0


def _get_video_frame_count(video_path: Path) -> int | None:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        if not streams:
            return None
        nb_frames = streams[0].get("nb_frames")
        if nb_frames is None:
            return None
        try:
            return int(nb_frames)
        except (TypeError, ValueError):
            return None
    except Exception:
        return None


def _get_video_fps(video_path: Path) -> float:
    """获取视频帧率"""
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=r_frame_rate", "-of", "json", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return 30.0
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        if not streams:
            return 30.0
        r_frame_rate = streams[0].get("r_frame_rate", "30/1")
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            return float(num) / float(den)
        return float(r_frame_rate)
    except:
        return 30.0


def _create_video_from_frames(frames_dir: Path, video_path: Path, fps: float) -> bool:
    """从 frames/ 合成视频，替换原 video.mp4"""
    # 检查帧文件格式
    jpg_files = list(frames_dir.glob("*.jpg"))
    png_files = list(frames_dir.glob("*.png"))

    if jpg_files:
        pattern = str(frames_dir / "%05d.jpg")
    elif png_files:
        pattern = str(frames_dir / "%05d.png")
    else:
        return False

    # 备份原视频（如果还没备份）
    backup_path = video_path.with_suffix(".mp4.backup")
    if not backup_path.exists() and video_path.exists():
        shutil.copy2(video_path, backup_path)

    # 临时输出文件
    temp_path = video_path.parent / f"{video_path.stem}_tmp.mp4"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "mpeg4",
        "-q:v", "5",
        "-pix_fmt", "yuv420p",
        str(temp_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if temp_path.exists():
            temp_path.unlink()
        return False

    # 替换原文件
    temp_path.replace(video_path)
    return True


def _update_record_progress(records: list[dict], session_folder: str, progress: float | int) -> None:
    for rec in records:
        if str(rec.get("session_folder", "")) == session_folder:
            rec["annotation_progress"] = progress
            return


def _write_records(records: list[dict]) -> None:
    tmp_path = UPLOAD_RECORDS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    tmp_path.replace(UPLOAD_RECORDS_PATH)


def _extract_frames(session_dir: Path, start_frame_id: int, video_path: Path, frames_dir: Path, force_recreate: bool = True, frame_count: int = None) -> None:
    """
    从 video.mp4 提取帧到 frames/ 目录。

    使用 -vf select 过滤器从 start_frame_id 开始提取帧。
    输出帧文件名从 00000.jpg 开始，对应原视频的 start_frame_id 帧。

    Args:
        start_frame_id: 从第几帧开始提取（0-indexed）
        frame_count: 视频总帧数，用于计算输出帧数
    """
    if force_recreate and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
    ]

    # 使用 select 过滤器从 start_frame_id 开始提取
    if start_frame_id > 0:
        cmd.extend(["-vf", f"select='gte(n\\,{start_frame_id})'", "-vsync", "vfr"])

    cmd.extend(["-q:v", "2"])

    # 限制输出帧数
    if frame_count and frame_count > 0 and start_frame_id > 0:
        output_frames = frame_count - start_frame_id
        if output_frames > 0:
            cmd.extend(["-frames:v", str(output_frames)])
    elif frame_count and frame_count > 0:
        cmd.extend(["-frames:v", str(frame_count)])

    cmd.extend(["-start_number", "0", str(frames_dir / "%05d.jpg")])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise ExtractFramesError("未找到 ffmpeg，请先安装")
    except subprocess.CalledProcessError as e:
        raise ExtractFramesError(f"ffmpeg 提取帧失败: {e.stderr or str(e)}")


def _process_one(
    session_path: Path,
    session_folder: str,
    object_category: str,
    video_name: str,
    records: list[dict],
    skip_existing: bool,
) -> tuple[bool, bool]:
    """
    处理单个 session。
    Returns:
        (是否成功, 是否被跳过)
    """
    frames_dir = session_path / "frames"
    if skip_existing and frames_dir.exists():
        n = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))
        if n > 0:
            return True, True

    video_path = session_path / "video.mp4"
    if not video_path.exists():
        video_path = session_path / "video0.mp4"
    if not video_path.exists():
        _update_record_progress(records, session_folder, -1)
        _write_records(records)
        tqdm.write(f"[跳过] 视频不存在: {video_name} ({session_path}) -> progress=-1")
        return False, False

    if _load_select_id(session_path) is None:
        _update_record_progress(records, session_folder, 0)
        _write_records(records)
        tqdm.write(f"[跳过] select_id.json 不存在或无效: {video_name} ({session_path}) -> progress=0")
        return False, False

    # 获取 start_id，从该帧开始提取
    start_frame_id = _start_frame_id_for_extract(session_path)

    frame_count = _get_video_frame_count(video_path)
    output_frames = (frame_count - start_frame_id) if frame_count and start_frame_id > 0 else frame_count
    frame_count_str = f"{frame_count} 帧 (start_id={start_frame_id}, 输出 {output_frames} 帧)" if frame_count else "未知帧数"
    tqdm.write(f"视频: {video_name} | 物体: {object_category} | {frame_count_str}")

    # 获取原视频帧率（在截取前获取）
    fps = _get_video_fps(video_path)

    # 使用 start_id 过滤，从 start_frame_id 开始提取帧
    _extract_frames(session_path, start_frame_id, video_path, frames_dir, force_recreate=not skip_existing, frame_count=frame_count)

    # 如果 start_id > 0，需要从 frames/ 合成新的 video.mp4（截取后的）
    if start_frame_id > 0:
        if _create_video_from_frames(frames_dir, video_path, fps):
            new_frame_count = _get_video_frame_count(video_path)
            tqdm.write(f"  -> video.mp4 已截取: {frame_count} -> {new_frame_count} 帧")
        else:
            tqdm.write(f"  -> [警告] video.mp4 截取失败，保留原视频")

    _update_record_progress(records, session_folder, 1.1)
    _write_records(records)
    return True, False


def _process_single_video_dir(session_dir: Path, skip_existing: bool) -> bool:
    """
    处理单个视频目录（--video_dir 模式），跳过 upload_records 逻辑。
    Returns: True if successful.
    """
    session_dir = session_dir.resolve()
    if not session_dir.exists():
        raise ExtractFramesError(f"目录不存在: {session_dir}")

    frames_dir = session_dir / "frames"
    if skip_existing and frames_dir.exists():
        n = len(list(frames_dir.glob("*.jpg"))) + len(list(frames_dir.glob("*.png")))
        if n > 0:
            print(f"[跳过] 已有 {n} 帧: {frames_dir}")
            return True

    video_path = session_dir / "video.mp4"
    if not video_path.exists():
        video_path = session_dir / "video0.mp4"
    if not video_path.exists():
        raise ExtractFramesError(f"视频不存在: {session_dir}")

    start_frame_id = _start_frame_id_for_extract(session_dir)
    frame_count = _get_video_frame_count(video_path)
    output_frames = (frame_count - start_frame_id) if frame_count and start_frame_id > 0 else frame_count
    frame_count_str = f"{frame_count} 帧 (start_id={start_frame_id}, 输出 {output_frames} 帧)" if frame_count else "未知帧数"
    print(f"视频: {session_dir.name} | {frame_count_str}")

    fps = _get_video_fps(video_path)
    _extract_frames(session_dir, start_frame_id, video_path, frames_dir, force_recreate=not skip_existing, frame_count=frame_count)

    if start_frame_id > 0:
        if _create_video_from_frames(frames_dir, video_path, fps):
            new_frame_count = _get_video_frame_count(video_path)
            print(f"  -> video.mp4 已截取: {frame_count} -> {new_frame_count} 帧")
        else:
            print(f"  -> [警告] video.mp4 截取失败，保留原视频")

    print(f"[完成] 帧提取完成: {frames_dir}")
    return True


def _parse_args():
    parser = argparse.ArgumentParser(description="按 select_id 提取视频帧到 frames/")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Process a single video directory (bypass upload_records.json)",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=1,
        help="只处理 annotation_progress 等于此值的记录 (默认 1)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若 frames/ 已有图片则跳过",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="守护进程模式：持续轮询等待新任务",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="轮询间隔（秒），默认 120",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # 互斥检查
    if args.daemon and args.video_dir:
        print("错误: --daemon 和 --video_dir 不能同时使用")
        sys.exit(1)

    # 单目录模式
    if args.video_dir:
        video_dir = Path(args.video_dir).expanduser().resolve()
        _process_single_video_dir(video_dir, skip_existing=args.skip_existing)
        return

    # 守护模式
    if args.daemon:
        from daemon_runner import daemon_loop, resolve_path, atomic_update_progress

        TARGET_PROGRESS = 1
        NEXT_PROGRESS = 1.1

        def process_batch(records, target_records, context, args, progress_reporter=None):
            from tqdm import tqdm
            for i, rec in enumerate(tqdm(target_records, desc="extract_frames", unit="vid")):
                sf = rec.get("session_folder", "")
                if not sf:
                    tqdm.write(f"[跳过] 记录缺少 session_folder")
                    continue
                session_path = resolve_path(sf)
                if not session_path.exists():
                    tqdm.write(f"[跳过] 目录不存在: {session_path}")
                    continue
                video_name = rec.get("file_name", Path(sf).name)
                object_category = rec.get("object_category", "未知")

                if progress_reporter:
                    progress_reporter.update(i + 1, item_name=video_name)
                try:
                    success, skipped = _process_one(
                        session_path, sf, object_category, video_name,
                        records, args.skip_existing
                    )
                    if success and not skipped:
                        atomic_update_progress(sf, NEXT_PROGRESS)
                        tqdm.write(f"[完成] {video_name} -> progress={NEXT_PROGRESS}")
                except Exception as e:
                    tqdm.write(f"[错误] {video_name}: {e}")

        daemon_loop(
            task_name="extract_frames",
            target_progress=TARGET_PROGRESS,
            next_progress=NEXT_PROGRESS,
            process_batch_fn=process_batch,
            poll_interval=args.poll_interval,
            args=args,
        )
        return

    # 原有批量模式
    records = _load_records()
    # 只处理 progress == 1 的记录（不包括 1.1）
    target = [r for r in records if r.get("annotation_progress", 0) == args.progress]
    if not target:
        return

    # 初始化进度上报
    progress = TaskProgress("extract_frames")
    progress.start(total=len(target), message=f"开始处理 {len(target)} 个视频")

    try:
        for i, rec in enumerate(tqdm(target, desc="extract_frames", unit="vid")):
            sf = rec.get("session_folder", "")
            if not sf:
                raise ExtractFramesError(f"第 {i + 1} 条记录缺少 session_folder")
            session_path = _resolve_path(sf)
            if not session_path.exists():
                raise ExtractFramesError(f"目录不存在: {session_path}")
            video_name = rec.get("file_name", Path(sf).name)
            object_category = rec.get("object_category", "未知")

            # 上报当前进度
            progress.update(i + 1, item_name=video_name, message=f"正在处理: {video_name}")

            _process_one(session_path, sf, object_category, video_name, records, args.skip_existing)

        # 完成
        progress.complete(f"完成！共处理 {len(target)} 个视频")
    except Exception as e:
        progress.fail(str(e))
        raise


if __name__ == "__main__":
    try:
        main()
    except ExtractFramesError as e:
        print(f"extract_frames 错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"extract_frames 异常: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
