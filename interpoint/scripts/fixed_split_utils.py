from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def default_split_json_path() -> Path:
    """Default split file shipped/checked-in next to these scripts."""
    return Path(__file__).with_name("split_progress4_seed42.json")


def _as_abs_session_folder(session_folder: str, data_root: Path) -> Path:
    p = Path(session_folder)
    if not p.is_absolute():
        p = data_root / p
    # strict=False keeps non-existing paths from raising
    return p.resolve(strict=False)


def load_split_session_records(
    split_json_path: Path,
    data_root: Path,
    split: str,
) -> List[Dict[str, object]]:
    """Load session records for a given split from a fixed split JSON.

    Expected JSON schema:
      - splits.{split}: list of record ids
      - records.{id}.session_folder: relative or absolute path

    Returns a list of dicts with at least:
      - record_id: str
      - session_folder: Path (absolute)
      - object_category: str
      - file_name: str
    """
    with open(split_json_path, "r") as f:
        data = json.load(f)

    split_ids = list((data.get("splits") or {}).get(split) or [])
    records = data.get("records") or {}

    out: List[Dict[str, object]] = []
    seen = set()

    for rid in split_ids:
        rec = records.get(rid)
        if not isinstance(rec, dict):
            continue

        sf = rec.get("session_folder")
        if not isinstance(sf, str) or not sf:
            continue

        abs_sf = _as_abs_session_folder(sf, data_root)
        key = str(abs_sf)
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "record_id": str(rid),
                "session_folder": abs_sf,
                "object_category": str(rec.get("object_category") or "object"),
                "file_name": str(rec.get("file_name") or "unknown"),
            }
        )

    return out


def iter_samples_from_session_records(
    session_records: Iterable[Dict[str, object]],
    *,
    frame_interval: int,
) -> Iterable[Dict[str, object]]:
    """Yield per-frame samples from split-defined session folders."""
    for rec in session_records:
        session_folder = rec.get("session_folder")
        if not isinstance(session_folder, Path):
            continue

        kp_file = session_folder / "kp_record_new.json"
        obj_file = session_folder / "obj_init.obj"
        frames_dir = session_folder / "frames"

        if not (kp_file.exists() and obj_file.exists() and frames_dir.exists()):
            continue

        frames = sorted([f.name for f in frames_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if not frames:
            continue

        for frame_name in frames[:: max(1, int(frame_interval))]:
            frame_id = Path(frame_name).stem
            yield {
                "session_folder": session_folder,
                "kp_file": kp_file,
                "obj_file": obj_file,
                "frame_name": frame_name,
                "frame_id": frame_id,
                "file_name": str(rec.get("file_name") or "unknown"),
                "object_category": str(rec.get("object_category") or "object"),
                "record_id": str(rec.get("record_id") or ""),
            }
