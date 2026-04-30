#!/usr/bin/env python3
"""Coordinate transforms for AIS trajectories and the navigation map."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class PixelPoint:
    col: float
    row: float


@dataclass(frozen=True)
class WorldPoint:
    x: float
    y: float


class AisMapTransform:
    """Convert lon/lat samples into map pixels and Gazebo world coordinates.

    The project uses a y-down map convention: increasing rows become
    increasing Gazebo y. This keeps generated obstacle tracks aligned with the
    occupancy array indexing used by the map utilities.
    """

    def __init__(
        self,
        map_meta: Dict,
        affine_params: Optional[Dict] = None,
        world_origin: Tuple[float, float] = (0.0, 0.0),
        world_scale: float = 1.0,
    ):
        self.map_meta = map_meta
        self.affine_params = affine_params
        self.world_origin = world_origin
        self.world_scale = world_scale
        self.width = int(map_meta["width"])
        self.height = int(map_meta["height"])
        self.resolution_x = float(map_meta.get("resolution_x", map_meta.get("resolution", 1.0)))
        self.resolution_y = float(map_meta.get("resolution_y", map_meta.get("resolution", 1.0)))

    @classmethod
    def from_files(
        cls,
        map_meta_path: str,
        affine_path: Optional[str] = None,
        world_origin: Tuple[float, float] = (0.0, 0.0),
        world_scale: float = 1.0,
    ) -> "AisMapTransform":
        with open(map_meta_path, "r", encoding="utf-8") as f:
            map_meta = yaml.safe_load(f)

        affine_params = None
        if affine_path:
            try:
                with open(affine_path, "r", encoding="utf-8") as f:
                    affine_doc = yaml.load(f, Loader=yaml.FullLoader)
                affine_params = (affine_doc or {}).get("affine_params")
            except FileNotFoundError:
                affine_params = None

        return cls(
            map_meta=map_meta,
            affine_params=affine_params,
            world_origin=world_origin,
            world_scale=world_scale,
        )

    def lonlat_to_pixel(self, lon: float, lat: float) -> PixelPoint:
        if self.affine_params:
            a1 = float(self.affine_params["a1"])
            b1 = float(self.affine_params["b1"])
            c1 = float(self.affine_params["c1"])
            a2 = float(self.affine_params["a2"])
            b2 = float(self.affine_params["b2"])
            c2 = float(self.affine_params["c2"])
            col = a1 * lon + b1 * lat + c1
            row = a2 * lon + b2 * lat + c2
        else:
            lon_min = float(self.map_meta["lon_min"])
            lon_max = float(self.map_meta["lon_max"])
            lat_min = float(self.map_meta["lat_min"])
            lat_max = float(self.map_meta["lat_max"])
            col = (lon - lon_min) / (lon_max - lon_min) * (self.width - 1)
            row = (lat_max - lat) / (lat_max - lat_min) * (self.height - 1)

        return PixelPoint(col=col, row=row)

    def pixel_to_world(self, point: PixelPoint) -> WorldPoint:
        x = self.world_origin[0] + point.col * self.resolution_x * self.world_scale
        y = self.world_origin[1] + point.row * self.resolution_y * self.world_scale
        return WorldPoint(x=x, y=y)

    def lonlat_to_world(self, lon: float, lat: float) -> Tuple[WorldPoint, PixelPoint]:
        pixel = self.lonlat_to_pixel(lon, lat)
        return self.pixel_to_world(pixel), pixel

    def pixel_in_bounds(self, point: PixelPoint, margin: float = 0.0) -> bool:
        return (
            -margin <= point.col <= self.width - 1 + margin
            and -margin <= point.row <= self.height - 1 + margin
        )

    def lonlat_in_meta_bounds(self, lon: float, lat: float, margin_ratio: float = 0.05) -> bool:
        lon_min = float(self.map_meta["lon_min"])
        lon_max = float(self.map_meta["lon_max"])
        lat_min = float(self.map_meta["lat_min"])
        lat_max = float(self.map_meta["lat_max"])
        lon_margin = (lon_max - lon_min) * margin_ratio
        lat_margin = (lat_max - lat_min) * margin_ratio
        return (
            lon_min - lon_margin <= lon <= lon_max + lon_margin
            and lat_min - lat_margin <= lat <= lat_max + lat_margin
        )


def yaw_from_points(p0: WorldPoint, p1: WorldPoint) -> float:
    return math.atan2(p1.y - p0.y, p1.x - p0.x)
