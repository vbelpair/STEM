from __future__ import annotations
from typing import Tuple, List
import numpy as np

Axial = Tuple[int, int]  # (q, r)
SQRT3 = float(np.sqrt(3.0))

# Axial neighbor directions (same for flat/pointy; orientation affects pixel transform)
AXIAL_DIRS: Tuple[Axial, ...] = (
    (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)
)

def axial_add(a: Axial, b: Axial) -> Axial:
    return (a[0] + b[0], a[1] + b[1])

def hex_to_pixel_flat(ax: np.ndarray, size: float, origin_xy: np.ndarray) -> np.ndarray:
    """
    Axial -> pixel for FLAT-TOP hexes.

    Parameters
    ----------
    ax : array-like [...,2] with (q,r)
    size : float
        Hex radius in pixels (center->corner).
    origin_xy : (2,)
        Pixel coordinate corresponding to hex (0,0) center.
    """
    q, r = ax[..., 0], ax[..., 1]
    x = size * (1.5 * q)
    y = size * (SQRT3 * (r + 0.5 * q))
    return np.stack([x, y], axis=-1) + origin_xy

def pixel_to_hex_flat(xy: np.ndarray, size: float, origin_xy: np.ndarray) -> np.ndarray:
    """
    Pixel -> fractional axial (q,r) for FLAT-TOP hexes.
    """
    p = xy - origin_xy
    x, y = p[..., 0], p[..., 1]
    q = (2.0 / 3.0 * x) / size
    r = (-1.0 / 3.0 * x + SQRT3 / 3.0 * y) / size
    return np.stack([q, r], axis=-1)

def axial_round(frac_qr: np.ndarray) -> Axial:
    """
    Round fractional axial coords to nearest hex using cube-rounding.
    """
    q, r = float(frac_qr[0]), float(frac_qr[1])
    x = q
    z = r
    y = -x - z

    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)

    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return (int(rx), int(rz))

# --- Red Blob "line drawing" (cube lerp + round) ---

def axial_to_cube(a: Axial) -> np.ndarray:
    q, r = a
    x = q
    z = r
    y = -x - z
    return np.array([x, y, z], dtype=float)

def cube_to_axial(c: np.ndarray) -> Axial:
    x, y, z = c
    return (int(round(x)), int(round(z)))

def cube_round(c: np.ndarray) -> np.ndarray:
    x, y, z = c
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)

    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return np.array([rx, ry, rz], dtype=float)

def cube_lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t

def hex_line_draw(a: Axial, b: Axial) -> List[Axial]:
    """
    Red Blob line drawing: interpolate in cube space and round.
    Returns inclusive list [a ... b].
    """
    ac = axial_to_cube(a)
    bc = axial_to_cube(b)
    N = int(max(abs(ac - bc)))  # cube distance
    if N == 0:
        return [a]

    results: List[Axial] = []
    for i in range(N + 1):
        t = i / N
        c = cube_round(cube_lerp(ac, bc, t))
        results.append(cube_to_axial(c))
    return results

# --- Segment -> "touched hexes" (UI beam preview) ---

def touched_hexes_by_segment(board, start_xy: np.ndarray, end_xy: np.ndarray) -> List[Axial]:
    """
    Compute which hexes are touched by a straight line segment in pixel space.

    Strategy: dense sampling along the segment; convert each sample to nearest hex;
    keep unique hexes in traversal order.

    This is robust and simple for interactive UI preview.
    """
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)
    v = end_xy - start_xy
    L = float(np.linalg.norm(v))

    if L < 1e-6:
        h = board.pixel_to_hex(start_xy)
        return [h] if board.inside_grid(h) else []

    step = max(1.0, board.hex_size / 3.0)
    n = int(np.ceil(L / step)) + 1
    ts = np.linspace(0.0, 1.0, n)
    pts = start_xy[None, :] + ts[:, None] * v[None, :]

    seen = set()
    out: List[Axial] = []
    for p in pts:
        h = board.pixel_to_hex(p)
        if board.inside_grid(h) and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def ray_to_rect_border(start_xy: np.ndarray, dir_xy: np.ndarray, W: int, H: int, eps: float = 1e-9) -> np.ndarray:
    """
    Intersect ray p(t)=start + t*dir with rectangle [0,W) x [0,H).
    Returns the first intersection point with the border in forward direction.
    """
    dx, dy = float(dir_xy[0]), float(dir_xy[1])
    if abs(dx) < eps and abs(dy) < eps:
        return start_xy.copy()

    ts = []

    # x = 0 and x = W-1
    if abs(dx) > eps:
        for xb in (0.0, W - 1.0):
            t = (xb - start_xy[0]) / dx
            y = start_xy[1] + t * dy
            if t > 0 and 0.0 <= y <= (H - 1):
                ts.append(t)

    # y = 0 and y = H-1
    if abs(dy) > eps:
        for yb in (0.0, H - 1.0):
            t = (yb - start_xy[1]) / dy
            x = start_xy[0] + t * dx
            if t > 0 and 0.0 <= x <= (W - 1):
                ts.append(t)

    if not ts:
        return start_xy.copy()

    tmin = min(ts)
    return start_xy + tmin * np.array([dx, dy], dtype=float)

def ray_path_to_border(board, start_hex: Axial, arrow_end_xy: np.ndarray, max_steps: int = 2000) -> List[Axial]:
    """
    Build a geometric ray path:
      start_hex center -> direction defined by arrow_end_xy -> extend to game border.
    Returns list of axial hexes touched (inclusive), capped by max_steps.
    """
    sxy = board.hex_center_xy(start_hex)
    dir_xy = np.asarray(arrow_end_xy, dtype=float) - sxy
    far_xy = ray_to_rect_border(sxy, dir_xy, board.W, board.H)

    path = touched_hexes_by_segment(board, sxy, far_xy)
    return path[:max_steps]