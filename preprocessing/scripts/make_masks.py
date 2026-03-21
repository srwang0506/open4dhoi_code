# -*- coding: utf-8 -*-
import sys
import json
import argparse
import subprocess
from pathlib import Path
from contextlib import nullcontext
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Resolve SAM2 root from ./sam2 (script dir or cwd)
_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

SAM2_ROOT = Path(os.environ.get("SAM2_ROOT", "/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/reconstruction/sam2"))
SAM2_PKG_DIR = SAM2_ROOT / "sam2"

sys.path.insert(0, str(SAM2_ROOT))

# 项目根目录
PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_RECORDS_PATH = PROJECT_DIR / "upload_records.json"
DATA_DIR = PROJECT_DIR / "tiktok_data"

# 导入进度上报模块
from task_status import TaskProgress


class MakeMasksError(Exception):
    """处理过程中发生错误，立即终止"""


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


def _resolve_path(p: str) -> Path:
    """解析路径，支持相对路径和绝对路径"""
    if not isinstance(p, str) or not p:
        return Path("")
    if p.startswith("./"):
        return (PROJECT_DIR / p[2:]).resolve()
    # allow already relative like "tiktok_data/..."
    if p.startswith("tiktok_data/"):
        return (PROJECT_DIR / p).resolve()
    # absolute
    return Path(p).expanduser().resolve()


def _load_records() -> list[dict]:
    """读取 upload_records.json"""
    if not UPLOAD_RECORDS_PATH.exists():
        raise FileNotFoundError(f"upload_records.json not found: {UPLOAD_RECORDS_PATH}")
    with UPLOAD_RECORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("upload_records.json must be a list")
    return data


def _get_video_frame_count(video_path: Path) -> int | None:
    """使用 ffprobe 获取视频帧数"""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "json", str(video_path)
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
        return int(nb_frames)
    except Exception:
        return None


def _extract_frames_if_needed(video_path: Path, frames_dir: Path) -> bool:
    """
    若 frames 目录不存在或为空则从视频提取所有帧。已有帧则直接返回 True。
    失败则抛出 MakeMasksError，立即终止。

    注意：提取视频的所有帧，不使用 start_id 过滤。
    frames/00000.jpg 对应 video.mp4 的第 0 帧。
    """
    if frames_dir.exists():
        n = (
            len(list(frames_dir.glob("*.jpg")))
            + len(list(frames_dir.glob("*.jpeg")))
            + len(list(frames_dir.glob("*.png")))
        )
        if n > 0:
            return True

    if not video_path.exists():
        raise MakeMasksError(f"视频不存在: {video_path}")

    # 获取视频帧数用于限制输出
    frame_count = _get_video_frame_count(video_path)

    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-q:v", "2",
    ]

    # 使用 -frames:v 限制输出帧数，避免音频轨道更长导致复制帧
    if frame_count and frame_count > 0:
        cmd.extend(["-frames:v", str(frame_count)])

    cmd.extend(["-start_number", "0", str(frames_dir / "%05d.jpg")])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except FileNotFoundError:
        raise MakeMasksError("未找到 ffmpeg，请安装后再运行")
    except subprocess.CalledProcessError as e:
        raise MakeMasksError(f"ffmpeg 提取帧失败: {e.stderr or str(e)}")


def _save_mask(mask_any: np.ndarray, out_path: Path):
    """
    Save mask as 8-bit PNG.

    SAM2 may output masks with an extra singleton dimension, e.g. (1, H, W).
    We squeeze it to (H, W) before saving.
    """
    m = np.asarray(mask_any)

    # Common SAM2 shapes: (H,W) or (1,H,W) or (H,W,1)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        raise RuntimeError(f"Unexpected mask shape {m.shape} for {out_path}")

    img = (m.astype(np.uint8) * 255)
    Image.fromarray(img, mode="L").save(str(out_path), compress_level=0)


def _resolve_cfg_and_ckpt():
    """
    Hydra search path is pkg://sam2, so config_name must be relative to SAM2_PKG_DIR.
    That means config_name must start with "configs/..."
    """
    cfg_abs = SAM2_PKG_DIR / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    if not cfg_abs.exists():
        raise RuntimeError(f"SAM2 config file not found: {cfg_abs}")

    cfg_name = "configs/sam2.1/sam2.1_hiera_l.yaml"

    ckpt_abs = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
    if not ckpt_abs.exists():
        raise RuntimeError(f"SAM2 checkpoint not found: {ckpt_abs}")

    return cfg_name, ckpt_abs


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate masks for videos with annotation_progress=1")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Optional: Process a single video directory (for backward compatibility). If not provided, will process all videos with annotation_progress=1 from upload_records.json",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have mask_dir and human_mask_dir",
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


def process_single_video(
    demo_dir: Path,
    predictor=None,
    cfg_name: str = None,
    ckpt_path: Path = None,
    device: str = "cuda",
    skip_if_exists: bool = False,
) -> Tuple[bool, bool]:
    """
    处理单个视频目录，生成 mask。任一失败即抛出 MakeMasksError 并终止。
    Returns:
        (成功, 是否被跳过)
    """
    demo_dir = demo_dir.resolve()
    if not demo_dir.exists():
        raise MakeMasksError(f"目录不存在: {demo_dir}")

    out_obj_dir = demo_dir / "mask_dir"
    out_human_dir = demo_dir / "human_mask_dir"
    if skip_if_exists and out_obj_dir.exists() and out_human_dir.exists():
        no = len(list(out_obj_dir.glob("*.png")))
        nh = len(list(out_human_dir.glob("*.png")))
        if no > 0 and nh > 0:
            return True, True

    frames_dir = demo_dir / "frames"
    video_path = demo_dir / "video.mp4"
    if not video_path.exists():
        video_path = demo_dir / "video0.mp4"

    _extract_frames_if_needed(video_path, frames_dir)

    if not frames_dir.exists():
        raise MakeMasksError(f"帧目录不存在: {frames_dir}")
    nf = (
        len(list(frames_dir.glob("*.jpg")))
        + len(list(frames_dir.glob("*.jpeg")))
        + len(list(frames_dir.glob("*.png")))
    )
    if nf == 0:
        raise MakeMasksError(f"无帧文件: {frames_dir}")

    points_path = demo_dir / "points.json"
    if not points_path.exists():
        raise MakeMasksError(f"points.json 不存在: {points_path}")

    out_obj_dir.mkdir(parents=True, exist_ok=True)
    out_human_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(points_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        human_pts = np.asarray(data["human_points"], dtype=np.float32)
        object_pts = np.asarray(data["object_points"], dtype=np.float32)
    except json.JSONDecodeError as e:
        raise MakeMasksError(f"points.json 解析失败 {points_path}: {e}")
    except Exception as e:
        raise MakeMasksError(f"读取 points.json 失败 {points_path}: {e}")

    if len(human_pts) == 0 or len(object_pts) == 0:
        raise MakeMasksError(f"points.json 中 human/object 点为空: {points_path}")

    human_lbl = np.ones((len(human_pts),), dtype=np.int32)
    object_lbl = np.ones((len(object_pts),), dtype=np.int32)
    prompt_frame = 0

    if predictor is None:
        if cfg_name is None or ckpt_path is None:
            cfg_name, ckpt_path = _resolve_cfg_and_ckpt()
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(cfg_name, str(ckpt_path))

    obj_id_object, obj_id_human = 1, 2
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(video_path=str(frames_dir))
        predictor.add_new_points_or_box(
            state, frame_idx=prompt_frame, obj_id=obj_id_object,
            points=object_pts, labels=object_lbl,
        )
        predictor.add_new_points_or_box(
            state, frame_idx=prompt_frame, obj_id=obj_id_human,
            points=human_pts, labels=human_lbl,
        )
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            obj_ids = list(obj_ids)
            for i, oid in enumerate(obj_ids):
                m = (mask_logits[i] > 0).detach().cpu().numpy()
                if oid == obj_id_object:
                    _save_mask(m, out_obj_dir / f"{frame_idx:05d}.png")
                elif oid == obj_id_human:
                    _save_mask(m, out_human_dir / f"{frame_idx:05d}.png")

    return True, False


def main():
    args = _parse_args()

    # 互斥检查
    if args.daemon and args.video_dir:
        print("错误: --daemon 和 --video_dir 不能同时使用")
        sys.exit(1)

    if args.video_dir:
        demo_dir = Path(args.video_dir).expanduser().resolve()
        cfg_name, ckpt_path = _resolve_cfg_and_ckpt()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        process_single_video(
            demo_dir,
            predictor=None,
            cfg_name=cfg_name,
            ckpt_path=ckpt_path,
            device=device,
            skip_if_exists=args.skip_existing,
        )
        return

    # 守护模式
    if args.daemon:
        from daemon_runner import daemon_loop, resolve_path, atomic_update_progress

        TARGET_PROGRESS = 1.1
        NEXT_PROGRESS = 1.2

        def init_fn(args):
            cfg_name, ckpt_path = _resolve_cfg_and_ckpt()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from sam2.build_sam import build_sam2_video_predictor
            predictor = build_sam2_video_predictor(cfg_name, str(ckpt_path))
            return {"predictor": predictor, "cfg_name": cfg_name, "ckpt_path": ckpt_path, "device": device}

        def process_batch(records, target_records, context, args, progress_reporter=None):
            predictor = context["predictor"]
            cfg_name = context["cfg_name"]
            ckpt_path = context["ckpt_path"]
            device = context["device"]

            for i, rec in enumerate(tqdm(target_records, desc="make_masks", unit="vid")):
                sf = rec.get("session_folder", "")
                if not sf:
                    tqdm.write(f"[跳过] 记录缺少 session_folder")
                    continue
                session_path = resolve_path(sf)
                video_name = rec.get("file_name", Path(sf).name)
                object_category = rec.get("object_category", "未知")
                tqdm.write(f"视频: {video_name} | 物体: {object_category}")

                if progress_reporter:
                    progress_reporter.update(i + 1, item_name=video_name)

                try:
                    process_single_video(
                        session_path,
                        predictor=predictor,
                        cfg_name=cfg_name,
                        ckpt_path=ckpt_path,
                        device=device,
                        skip_if_exists=args.skip_existing,
                    )
                    atomic_update_progress(sf, NEXT_PROGRESS)
                    tqdm.write(f"[完成] {video_name} -> progress={NEXT_PROGRESS}")
                except Exception as e:
                    tqdm.write(f"[错误] {video_name}: {e}")
                    raise

        daemon_loop(
            task_name="make_masks",
            target_progress=TARGET_PROGRESS,
            next_progress=NEXT_PROGRESS,
            process_batch_fn=process_batch,
            init_fn=init_fn,
            poll_interval=args.poll_interval,
            args=args,
        )
        return

    # 原有批量模式
    records = _load_records()
    target_records = [r for r in records if r.get("annotation_progress", 0) == 1.1]
    print(len(target_records))
    if len(target_records) == 0:
        return

    # 初始化进度上报
    progress = TaskProgress("make_masks")
    progress.start(total=len(target_records), message=f"开始处理 {len(target_records)} 个视频")

    cfg_name, ckpt_path = _resolve_cfg_and_ckpt()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(cfg_name, str(ckpt_path))

    try:
        for i, record in enumerate(tqdm(target_records, desc="make_masks", unit="vid")):
            session_folder = record.get("session_folder", "")
            if not session_folder:
                raise MakeMasksError(f"第 {i + 1} 条记录缺少 session_folder")
            session_path = _resolve_path(session_folder)
            video_name = record.get("file_name", Path(session_folder).name)
            object_category = record.get("object_category", "未知")
            tqdm.write(f"视频: {video_name} | 物体: {object_category}")

            # 上报当前进度
            progress.update(i + 1, item_name=video_name, message=f"正在处理: {video_name}")

            process_single_video(
                session_path,
                predictor=predictor,
                cfg_name=cfg_name,
                ckpt_path=ckpt_path,
                device=device,
                skip_if_exists=args.skip_existing,
            )
            _update_record_progress(records, session_folder, 1.2)
            _write_records(records)

        # 完成
        progress.complete(f"完成！共处理 {len(target_records)} 个视频")
    except Exception as e:
        progress.fail(str(e))
        raise


if __name__ == "__main__":
    try:
        main()
    except MakeMasksError as e:
        print(f"make_masks 错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"make_masks 异常: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
