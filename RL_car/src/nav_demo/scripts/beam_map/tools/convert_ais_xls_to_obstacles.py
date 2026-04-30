#!/usr/bin/env python3
"""Convert raw AIS XLS tracks into a Gazebo dynamic-obstacle scenario."""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BEAM_MAP_DIR = SCRIPT_DIR.parent
if str(BEAM_MAP_DIR) not in sys.path:
    sys.path.insert(0, str(BEAM_MAP_DIR))

from ais_coordinate_transform import AisMapTransform, WorldPoint, yaw_from_points


COLUMN_ALIASES = {
    "mmsi": "mmsi",
    "船舶mmsi": "mmsi",
    "经度": "lon",
    "longitude": "lon",
    "lon": "lon",
    "lng": "lon",
    "纬度": "lat",
    "latitude": "lat",
    "lat": "lat",
    "时间": "time",
    "更新时间": "time",
    "time": "time",
    "timestamp": "time",
    "速度": "speed",
    "航速": "speed",
    "speed": "speed",
    "sog": "speed",
    "航向": "cog",
    "对地航向": "cog",
    "cog": "cog",
    "heading": "heading",
}


def detect_header_row(df: pd.DataFrame) -> int:
    keywords = ("mmsi", "经度", "longitude", "lon", "纬度", "latitude", "lat", "时间", "time")
    for idx, row in df.iterrows():
        row_text = " ".join(str(v).strip().lower() for v in row.values if pd.notna(v))
        if sum(1 for key in keywords if key in row_text) >= 2:
            return int(idx)
    return 1


def standardize_columns(df: pd.DataFrame, header_row: int) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.iloc[header_row].values
    df = df.iloc[header_row + 1 :].reset_index(drop=True)

    rename = {}
    for col in df.columns:
        normalized = str(col).strip().lower()
        if normalized in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[normalized]
            continue
        for key, value in COLUMN_ALIASES.items():
            if key in normalized:
                rename[col] = value
                break
    return df.rename(columns=rename)


def parse_time(value: Any) -> Optional[datetime]:
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip().replace("(UTC+8)", "").replace("(UTC)", "").strip()
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%m/%d %H:%M:%S",
        "%m/%d %H:%M",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.year == 1900:
                parsed = parsed.replace(year=2024)
            return parsed
        except ValueError:
            pass
    return None


def extract_mmsi_from_filename(path: Path) -> str:
    head = path.stem.split("_", 1)[0]
    return head if head else path.stem


def load_track(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(str(path), header=None)
    df = standardize_columns(raw, detect_header_row(raw))
    required = {"lon", "lat", "time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(sorted(missing)))

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["time_dt"] = df["time"].apply(parse_time)
    df = df.dropna(subset=["lon", "lat", "time_dt"])
    df = df[(df["lon"] >= 0) & (df["lon"] <= 180) & (df["lat"] >= 0) & (df["lat"] <= 90)]
    df = df.sort_values("time_dt").drop_duplicates(subset=["time_dt"]).reset_index(drop=True)
    if "mmsi" not in df.columns or df["mmsi"].isna().all():
        df["mmsi"] = extract_mmsi_from_filename(path)
    return df


def resample_track(df: pd.DataFrame, dt: float) -> List[Dict[str, float]]:
    if len(df) < 2:
        return []

    start = df["time_dt"].iloc[0]
    end = df["time_dt"].iloc[-1]
    duration = (end - start).total_seconds()
    if duration <= 0:
        return []

    samples = []
    count = int(math.floor(duration / dt)) + 1
    for idx in range(count + 1):
        stamp = min(start + timedelta(seconds=idx * dt), end)
        after = int(df["time_dt"].searchsorted(stamp))
        if after <= 0:
            row = df.iloc[0]
            lon = float(row["lon"])
            lat = float(row["lat"])
        elif after >= len(df):
            row = df.iloc[-1]
            lon = float(row["lon"])
            lat = float(row["lat"])
        else:
            row0 = df.iloc[after - 1]
            row1 = df.iloc[after]
            t0 = row0["time_dt"]
            t1 = row1["time_dt"]
            alpha = (stamp - t0).total_seconds() / max((t1 - t0).total_seconds(), 1e-6)
            lon = float(row0["lon"] + alpha * (row1["lon"] - row0["lon"]))
            lat = float(row0["lat"] + alpha * (row1["lat"] - row0["lat"]))
        samples.append({"t": (stamp - start).total_seconds(), "lon": lon, "lat": lat})
        if stamp == end:
            break
    return samples


def build_obstacle(
    path: Path,
    transform: AisMapTransform,
    dt: float,
    radius: float,
    min_points: int,
    max_speed: float,
) -> Optional[Dict]:
    df = load_track(path)
    before_bounds = len(df)
    df = df[df.apply(lambda r: transform.lonlat_in_meta_bounds(float(r["lon"]), float(r["lat"])), axis=1)]
    if len(df) < min_points:
        return None

    samples = resample_track(df, dt)
    points = []
    for sample in samples:
        world, pixel = transform.lonlat_to_world(sample["lon"], sample["lat"])
        if transform.pixel_in_bounds(pixel, margin=2.0):
            points.append(
                {
                    "t": float(sample["t"]),
                    "x": float(world.x),
                    "y": float(world.y),
                    "col": float(pixel.col),
                    "row": float(pixel.row),
                    "lon": float(sample["lon"]),
                    "lat": float(sample["lat"]),
                }
            )

    if len(points) < min_points:
        return None

    velocities = []
    for idx, point in enumerate(points):
        if idx == 0:
            p0 = WorldPoint(points[idx]["x"], points[idx]["y"])
            p1 = WorldPoint(points[min(idx + 1, len(points) - 1)]["x"], points[min(idx + 1, len(points) - 1)]["y"])
            delta_t = max(points[min(idx + 1, len(points) - 1)]["t"] - point["t"], 1e-6)
        else:
            p0 = WorldPoint(points[idx - 1]["x"], points[idx - 1]["y"])
            p1 = WorldPoint(point["x"], point["y"])
            delta_t = max(point["t"] - points[idx - 1]["t"], 1e-6)

        vx = (p1.x - p0.x) / delta_t
        vy = (p1.y - p0.y) / delta_t
        speed = math.hypot(vx, vy)
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale
        yaw = yaw_from_points(p0, p1) if speed > 1e-6 else 0.0
        point["vx"] = round(vx, 4)
        point["vy"] = round(vy, 4)
        point["yaw"] = round(yaw, 4)
        point["t"] = round(point["t"], 3)
        point["x"] = round(point["x"], 3)
        point["y"] = round(point["y"], 3)
        point["col"] = round(point["col"], 3)
        point["row"] = round(point["row"], 3)
        velocities.append(math.hypot(vx, vy))

    return {
        "id": str(df["mmsi"].iloc[0]),
        "source_file": path.name,
        "radius": radius,
        "start_time": points[0]["t"],
        "end_time": points[-1]["t"],
        "trajectory": points,
        "stats": {
            "raw_points": int(before_bounds),
            "in_bounds_points": int(len(df)),
            "resampled_points": int(len(points)),
            "max_speed": round(max(velocities) if velocities else 0.0, 4),
        },
    }


def choose_dt(input_dir: Path) -> float:
    intervals = []
    for path in sorted(input_dir.glob("*.xls"))[:5]:
        try:
            df = load_track(path)
        except Exception:
            continue
        if len(df) > 1:
            values = df["time_dt"].diff().dt.total_seconds().dropna()
            intervals.extend([float(v) for v in values if v > 0])
    if not intervals:
        return 5.0
    median = float(np.median(np.array(intervals)))
    if median <= 3.0:
        return 5.0
    if median <= 8.0:
        return 10.0
    return max(10.0, median)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/raw/trajectories")
    parser.add_argument("--map-meta", required=True)
    parser.add_argument("--affine", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--max-active-obstacles", type=int, default=5)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--world-origin-x", type=float, default=None)
    parser.add_argument("--world-origin-y", type=float, default=None)
    parser.add_argument("--world-scale", type=float, default=0.01)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--max-speed", type=float, default=8.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    dt = args.dt if args.dt is not None else choose_dt(input_dir)
    # The AIS map metadata is in real-world meters. Gazebo training is much
    # easier to keep stable when the map is centered and scaled down.
    import yaml
    with open(args.map_meta, "r", encoding="utf-8") as f:
        map_meta_for_origin = yaml.safe_load(f)
    res_x = float(map_meta_for_origin.get("resolution_x", map_meta_for_origin.get("resolution", 1.0)))
    res_y = float(map_meta_for_origin.get("resolution_y", map_meta_for_origin.get("resolution", 1.0)))
    width = int(map_meta_for_origin["width"])
    height = int(map_meta_for_origin["height"])
    origin_x = args.world_origin_x
    origin_y = args.world_origin_y
    if origin_x is None:
        origin_x = -0.5 * width * res_x * args.world_scale
    if origin_y is None:
        origin_y = -0.5 * height * res_y * args.world_scale

    transform = AisMapTransform.from_files(
        args.map_meta,
        affine_path=args.affine,
        world_origin=(origin_x, origin_y),
        world_scale=args.world_scale,
    )

    obstacles = []
    failures = []
    for path in sorted(input_dir.glob("*.xls")):
        try:
            obstacle = build_obstacle(path, transform, dt, args.radius, args.min_points, args.max_speed)
            if obstacle is None:
                failures.append({"file": path.name, "reason": "not enough in-bounds trajectory points"})
            else:
                obstacles.append(obstacle)
        except Exception as exc:
            failures.append({"file": path.name, "reason": str(exc)})

    scenario = {
        "schema_version": 1,
        "map_frame": "map",
        "gazebo_frame": "world",
        "time_step": dt,
        "time_scale": args.time_scale,
        "max_active_obstacles": args.max_active_obstacles,
        "obstacles": obstacles,
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_dir": str(input_dir),
            "source_files": len(list(input_dir.glob("*.xls"))),
            "successful_files": len(obstacles),
            "failed_files": len(failures),
            "failures": failures,
            "map_meta": args.map_meta,
            "affine": args.affine,
            "world_origin": [origin_x, origin_y],
            "world_scale": args.world_scale,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2)

    print("Wrote %s with %d obstacles (%d failures)" % (out_path, len(obstacles), len(failures)))


if __name__ == "__main__":
    main()
