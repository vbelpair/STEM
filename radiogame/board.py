from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional, Callable
import numpy as np
from collections import deque

from .hexgrid import Axial, AXIAL_DIRS, hex_to_pixel_flat, pixel_to_hex_flat, axial_round

@dataclass
class Tile:
    """
    One hex tile in axial coordinates (q,r).
    Stores accumulated dose and an optional tissue label.
    """
    q: int
    r: int
    dose: float = 0.0
    tissue: int = 0          # 0 non-tumor, 1 tumor
    score_dose: bool = True  # False for "air" tiles
    mean_intensity: float = 0.0
    can_start_beam: bool = False  # NEW

    def coords(self) -> Axial:
        return (self.q, self.r)

    def neighbours(self) -> Iterable[Axial]:
        for dq, dr in AXIAL_DIRS:
            yield (self.q + dq, self.r + dr)

class GameBoard:
    """
    Hex grid over a 2D image.

    - Coordinates are axial (q,r).
    - Orientation is FLAT-TOP (matching hex_to_pixel_flat/pixel_to_hex_flat).
    - Tiles are generated so their centers lie inside the image bounds.
    """

    def __init__(
        self,
        image,
        hex_size_px: float,
        origin_xy: Optional[np.ndarray] = None,
        image_array_2d: Optional[np.ndarray] = None,
        air_threshold: float = 100.,
    ):
        self.image = image
        self.hex_size = float(hex_size_px)

        W, H = self.image.GetSize()  # SimpleITK: (x,y)
        self.W, self.H = int(W), int(H)

        if origin_xy is None:
            self.origin_xy = np.array([self.W / 2.0, self.H / 2.0], dtype=float)
        else:
            self.origin_xy = np.array(origin_xy, dtype=float)

        self.tiles: Dict[Axial, Tile] = {}
        self.image_array_2d = image_array_2d  # numpy (H,W) in display units (0..255 typically)
        self.air_threshold = float(air_threshold)
        self._build_covering_grid()
        self.compute_start_region()

    # --- coordinate helpers ---
    def hex_center_xy(self, qr: Axial) -> np.ndarray:
        ax = np.array([qr[0], qr[1]], dtype=float)
        return hex_to_pixel_flat(ax, self.hex_size, self.origin_xy)

    def pixel_to_hex(self, xy: np.ndarray) -> Axial:
        frac = pixel_to_hex_flat(np.asarray(xy, dtype=float), self.hex_size, self.origin_xy)
        return axial_round(frac)

    def inside_image_xy(self, xy: np.ndarray) -> bool:
        x, y = float(xy[0]), float(xy[1])
        return (0.0 <= x < self.W) and (0.0 <= y < self.H)

    def inside_grid(self, qr: Axial) -> bool:
        return qr in self.tiles

    def get_tile(self, qr: Axial) -> Optional[Tile]:
        return self.tiles.get(qr)

    def _tile_mean_intensity(self, center_xy: np.ndarray) -> float:
        """
        Approximate mean pixel intensity for the tile by sampling a circular ROI around the hex center.
        Uses image_array_2d if provided; otherwise returns 255.
        """
        if self.image_array_2d is None:
            return 255.0

        arr = self.image_array_2d
        H, W = arr.shape[:2]
        cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))

        # ROI radius: a bit smaller than hex radius
        rad = max(2, int(round(self.hex_size * 0.6)))
        x0, x1 = max(0, cx - rad), min(W, cx + rad + 1)
        y0, y1 = max(0, cy - rad), min(H, cy + rad + 1)

        patch = arr[y0:y1, x0:x1].astype(float)
        if patch.size == 0:
            return 0.0

        # circular mask
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2

        vals = patch[mask]
        if vals.size == 0:
            return float(patch.mean())
        return float(vals.mean())

    def max_dose_non_tumor(self) -> float:
        """
        Maximum accumulated dose in non-tumor tiles (excluding air).
        """
        vals = [t.dose for t in self.tiles.values() if t.tissue == 0 and t.score_dose]
        return float(max(vals)) if vals else 0.0
    
    def min_dose_tumor(self) -> float:
        """
        Maximum accumulated dose in non-tumor tiles (excluding air).
        """
        vals = [t.dose for t in self.tiles.values() if t.tissue == 1 and t.score_dose]
        return float(min(vals)) if vals else 0.0
    
    def tumor_dose_variability(self) -> float:
        """
        Dose variability inside tumor as coefficient of variation (std/mean),
        excluding air tiles. Lower = more uniform.
        """
        vals = np.array([t.dose for t in self.tiles.values() if t.tissue == 1 and t.score_dose], dtype=float)
        if vals.size == 0:
            return 0.0
        mu = float(vals.mean())
        if mu <= 1e-12:
            return 0.0
        sigma = float(vals.std(ddof=0))
        return float(sigma / mu)

    # --- grid construction ---
    def _build_covering_grid(self, tissue_label_fn: Optional[Callable[[np.ndarray], int]] = None) -> None:
        corners = np.array(
            [[0.0, 0.0], [self.W - 1.0, 0.0], [0.0, self.H - 1.0], [self.W - 1.0, self.H - 1.0]],
            dtype=float,
        )
        frac = pixel_to_hex_flat(corners, self.hex_size, self.origin_xy)
        qmin = int(np.floor(frac[:, 0].min())) - 3
        qmax = int(np.ceil(frac[:, 0].max())) + 3
        rmin = int(np.floor(frac[:, 1].min())) - 3
        rmax = int(np.ceil(frac[:, 1].max())) + 3

        tiles: Dict[Axial, Tile] = {}
        for q in range(qmin, qmax + 1):
            for r in range(rmin, rmax + 1):
                center = self.hex_center_xy((q, r))
                if self.inside_image_xy(center):
                    mean_I = self._tile_mean_intensity(center)
                    score = mean_I >= self.air_threshold
                    tiles[(q, r)] = Tile(q=q, r=r, dose=0.0, tissue=0, score_dose=score, mean_intensity=mean_I)

        self.tiles = tiles

    def compute_start_region(self, air_threshold: float = 50.0, edge_margin: float = 2.0) -> None:
        """
        Mark tiles where beams are allowed to start.
        Combined criteria:
        - mean_intensity < air_threshold (is air)
        - AND located at the actual image border
        
        Args:
            air_threshold: Intensity threshold for classifying as air
            edge_margin: Distance from edge in hex_size units (default 0.5)
        """
        # Reset all tiles
        for t in self.tiles.values():
            t.can_start_beam = False
        
        # Calculate edge distance in pixels
        edge_distance = self.hex_size * edge_margin
        
        # Mark tiles that meet BOTH criteria
        for qr, tile in self.tiles.items():
            # Criterion 1: Must be air (low intensity)
            if not tile.score_dose:  # This means mean_intensity < air_threshold
                center = self.hex_center_xy(qr)
                x, y = center
                
                # Criterion 2: Must be at image border
                at_left = x < edge_distance
                at_right = x > self.W - edge_distance
                at_top = y < edge_distance
                at_bottom = y > self.H - edge_distance
                
                # ✅ Only allow start if BOTH conditions are met
                if at_left or at_right or at_top or at_bottom:
                    tile.can_start_beam = True
        
        # Optional: Debug info
        start_count = sum(1 for t in self.tiles.values() if t.can_start_beam)
        total_tiles = len(self.tiles)
        print(f"Start region: {start_count} tiles out of {total_tiles} total tiles")


    def set_tumor(self, tumor_hex: Axial) -> None:
        """Mark exactly one tile as tumor (tissue=1), all others tissue=0."""
        for qr, t in self.tiles.items():
            t.tissue = 1 if (qr == tumor_hex) else 0

    def set_tumor_tiles(self, tumor_hexes: set[Axial]) -> None:
        """Mark all hexes in tumor_hexes as tumor (tissue=1), others as non-tumor (tissue=0)."""
        tumor_hexes = set(tumor_hexes)
        for qr, t in self.tiles.items():
            t.tissue = 1 if qr in tumor_hexes else 0

    def tumor_tiles(self) -> set[Axial]:
        """Return set of tumor axial coordinates."""
        return {qr for qr, t in self.tiles.items() if t.tissue == 1}

    def dose_stats(self) -> tuple[float, float]:
        """
        Returns:
        dose_tumor = mean dose over tumor tiles (only score_dose tiles)
        dose_non_tumor = mean dose over non-tumor score_dose tiles
        """
        tumor = [t.dose for t in self.tiles.values() if t.tissue == 1 and t.score_dose]
        normal = [t.dose for t in self.tiles.values() if t.tissue == 0 and t.score_dose]

        dose_tumor = float(np.mean(tumor)) if len(tumor) else 0.0
        dose_non = float(np.mean(normal)) if len(normal) else 0.0
        return dose_tumor, dose_non

    # --- dose bookkeeping ---
    def reset_dose(self) -> None:
        for t in self.tiles.values():
            t.dose = 0.0

    def reset_all(self) -> None:
        """Reset dose and tumor labels to initial state."""
        for t in self.tiles.values():
            t.dose = 0.0
            t.tissue = 0  # no tumor

    def add_dose(self, qr: Axial, amount: float) -> None:
        t = self.tiles.get(qr)
        if t is not None:
            t.dose += float(amount)
