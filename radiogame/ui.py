from __future__ import annotations
from typing import Optional, Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyBboxPatch
from radiogame.menu import MenuScreen

from .hexgrid import Axial, touched_hexes_by_segment, ray_path_to_border
from .translations import t

# --- UI styling (radiotherapy vibe) ---
STYLE = {
    "grid_edge": (0.65, 1.0, 1.0, 0.55),     # cyan-ish
    "beam_preview": (1.0, 0.8, 0.1, 0.95),   # amber
    "start_dot": (1.0, 0.2, 0.8, 1.0),       # magenta
    #"hit_fill": (0.1, 1.0, 0.8, 0.25),       # mint glow
    "panel_bg": (0.0, 0.0, 0.0, 0.35),       # translucent black
    "panel_text": "white",
}

PARTICLE_COLORS = {
    "gamma":   (0.20, 1.00, 0.20, 1.0),  # green
    "electron":(0.20, 0.55, 1.00, 1.0),  # blue
    "proton":  (1.00, 0.20, 0.20, 1.0),  # red
}

def color_dose_tumor(x: float) -> str:
    if x < 5:
        return "red"
    elif x < 7:
        return "orange"
    elif x < 10:
        return "yellow"
    else:
        return "green"


def color_tumor_variability(x: float) -> str:
    if x < 0.1:
        return "green"
    elif x < 0.2:
        return "yellow"
    elif x < 0.5:
        return "orange"
    else:
        return "red"


def color_dose_non_tumor(x: float) -> str:
    if x < 0.2:
        return "green"
    elif x < 0.4:
        return "yellow"
    elif x < 1.0:
        return "orange"
    else:
        return "red"


def color_max_dose_non_tumor(x: float) -> str:
    if x < 1:
        return "green"
    elif x < 5:
        return "yellow"
    elif x < 10:
        return "orange"
    else:
        return "red"


def _style_axes(ax):
    """Dark, clean axes with no ticks."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("black")

def _with_alpha(rgba, a: float):
    r, g, b, _ = rgba
    return (r, g, b, a)


def hex_corners_flat(center_xy: np.ndarray, size: float) -> np.ndarray:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300], dtype=float))
    x = cx + size * np.cos(angles)
    y = cy + size * np.sin(angles)
    return np.stack([x, y], axis=1)

def snap_to_6_dirs_flat(board, start: Axial, target: Axial) -> Optional[Axial]:
    """
    Snap an intended direction to one of the 6 axial neighbor directions by vector similarity.
    Returns (dq,dr) or None if too small.
    """
    axial_dirs = [(1,0),(0,1),(-1,0),(0,-1),(1,-1),(-1,1)]
    sxy = board.hex_center_xy(start)
    txy = board.hex_center_xy(target)
    v = txy - sxy
    if np.linalg.norm(v) < 1e-6:
        return None

    best = None
    best_cos = -1e9
    for d in axial_dirs:
        n = (start[0] + d[0], start[1] + d[1])
        if not board.inside_grid(n):
            continue
        nxy = board.hex_center_xy(n)
        u = nxy - sxy
        cos = float(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u)))
        if cos > best_cos:
            best_cos = cos
            best = d
    return best

class BeamPickerUI:
    """
    Interactive Matplotlib UI for selecting:
      - start hex (click)
      - direction/segment (drag)
    Modes:
      - "free": line segment; highlight hexes touched by that segment
      - "6dir": snapped to 6 directions; highlight hexes along snapped segment

    Workflow:
      1) Dose map visible
      2) Click start: dose map temporarily hidden
      3) Drag arrow: hit tiles highlighted
      4) Enter: fires -> callback deposits dose
      5) Dose map shown again and autoscaled
    """

    def __init__(self, board, image_2d: np.ndarray, on_fire: Optional[Callable] = None, show_grid: bool = True, particle_models=None, language: str = "en", menu: MenuScreen = None):
        self.board = board
        self.image_2d = image_2d
        self.on_fire = on_fire  # callback(path_hexes, start_hex, mode_str)
        self.language = language
        self.menu = menu

        self.mode = "free"  # or "6dir"
        self.start_hex: Optional[Axial] = None
        self.arrow_end_xy: Optional[np.ndarray] = None
        self.dragging = False
        self.phase = "select_tumor"  # "select_tumor" or "play"
        self.tumor_set: set[Axial] = set()
        self._tumor_drag_mode: Optional[str] = None  # "add" or "erase"


        # Particle selection (GUI-level)
        self.particles = ["gamma", "electron", "proton"]
        self.particle_idx = 2  # default: proton
        self.particle_name = self.particles[self.particle_idx]
        self._particle_models = particle_models or {}

        # Energy scaling: E0 ~ arrow_length_px / (hex_size * energy_scale)
        # smaller scale => more energy for same arrow length
        self.energy_scale = 1.

        # create figure
        self.fig = plt.figure(figsize=(13.5, 7.8), constrained_layout=True)
        self.fig.patch.set_facecolor("black")

        # Grid:
        # row 0: main image (col 0), DVH (col 1)
        # row 1: HUD for image (col 0), stats under DVH (col 1)
        gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[3.6, 1.6],
            height_ratios=[4.0, 1.05],
            wspace=0.04,
            hspace=0.15,
        )

        self.ax = self.fig.add_subplot(gs[0, 0])       # planning image
        self.ax_dvh = self.fig.add_subplot(gs[0, 1])   # DVH
        self.ax_hud_img = self.fig.add_subplot(gs[1, 0])  # HUD under image
        self.ax_stats = self.fig.add_subplot(gs[1, 1])    # stats under DVH

        # Turn the HUD/stats axes into "panels"
        for a in (self.ax_hud_img, self.ax_stats):
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            a.set_facecolor("black")

        self._artists = {}
        self._tile_keys: List[Axial] = list(self.board.tiles.keys())  # stable ordering

        self._draw_base(show_grid=show_grid)
        self._hide_dose_map()

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_hud_panels()
        self._update_stats_panel()
        self._apply_particle_style()
        self._render_start_region()

    def _draw_base(self, show_grid: bool):
        self.ax.imshow(self.image_2d, cmap="gray", origin="upper")
        _style_axes(self.ax)
        self.ax.set_xlim(0, self.board.W)
        self.ax.set_ylim(self.board.H, 0)
        self.ax.set_aspect("equal")

        polys = []
        for qr in self._tile_keys:
            c = self.board.hex_center_xy(qr)
            polys.append(hex_corners_flat(c, self.board.hex_size))

        # Dose fill layer
        doses = np.array([self.board.tiles[qr].dose for qr in self._tile_keys], dtype=float)
        pc_fill = PolyCollection(polys, array=doses, cmap="magma", edgecolors="none", alpha=0.45)
        self.ax.add_collection(pc_fill)
        self._artists["dose_fill"] = pc_fill

        bbox = self.ax.get_position()
        cbar_w = 0.018
        pad = 0.010
        cax = self.fig.add_axes([bbox.x1 + pad, bbox.y0, cbar_w, bbox.height])

        cbar = self.fig.colorbar(pc_fill, cax=cax)
        #cbar.set_label("Dose (a.u.)", color="white", labelpad=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
        cbar.outline.set_edgecolor("white")

        self._artists["dose_cbar"] = cbar
        self._artists["dose_cax"] = cax

        # Hit/highlight layer
        #pc_hit = PolyCollection([], facecolors=[STYLE["hit_fill"]], edgecolors="none", alpha=STYLE["hit_fill"][3])
        #self.ax.add_collection(pc_hit)
        #self._artists["hit_fill"] = pc_hit
        # --- Preview deposit layer (colored by predicted dose per tile) ---
        pc_preview = PolyCollection(
            [],
            array=np.array([], dtype=float),
            cmap="magma",          # radiotherapy vibe; change if you prefer
            edgecolors="none",
            alpha=0.42,
        )
        self.ax.add_collection(pc_preview)
        self._artists["preview_fill"] = pc_preview

        # Grid outlines on top
        if show_grid:
            pc_edges = PolyCollection(
                polys,
                facecolors="none",
                edgecolors=[STYLE["grid_edge"]],
                linewidths=0.1,
                alpha=1.0,
            )
            self.ax.add_collection(pc_edges)


        # Selection markers
        self._artists["start_dot"], = self.ax.plot([], [], marker="o", ms=8,
                                           color=STYLE["start_dot"], markeredgecolor="white",
                                           markeredgewidth=0.8)
        self._artists["arrow"] = self.ax.annotate(
                                    "", xy=(0, 0), xytext=(0, 0),
                                    arrowprops=dict(arrowstyle="->", linewidth=2.4, color=STYLE["beam_preview"])
                                )


        self._autoscale_dose_colormap()

        # Tumor overlay (shown during selection and optionally during play)
        pc_tumor = PolyCollection(
            [],
            facecolors=[(1.0, 0.2, 0.2, 0.18)],   # soft red fill
            edgecolors=[(1.0, 0.2, 0.2, 0.85)],   # red outline
            linewidths=1.6,
        )
        self.ax.add_collection(pc_tumor)
        self._artists["tumor_fill"] = pc_tumor

        # start region
        pc_start = PolyCollection(
                    [],
                    facecolors=[(0.3, 0.9, 1.0, 0.12)],  # cyan glow
                    edgecolors="none",
                )
        self.ax.add_collection(pc_start)
        self._artists["start_region"] = pc_start

        # DVH
        self._init_dvh_plot()
        self._update_dvh()

    # --- panels ---
    def _update_hud_panels(self):
        lang = self.language  # Use instance language
        
        mode_txt = t("mode_free", lang) if self.mode == "free" else t("mode_6dir", lang)
        particle_txt = t(self.particle_name, lang)
        E0_txt = "-"
        if self.phase == "play" and self.start_hex is not None and self.arrow_end_xy is not None:
            E0_txt = str(self._compute_energy_E0())
        
        # --- Block 1 (status) ---
        block1 = "\n".join([
            f"{t('phase', lang)}: {t('phase_tumor_select', lang) if self.phase=='select_tumor' else t('phase_treatment', lang)}",
            #f"{t('mode', lang)}: {mode_txt} (M)" if self.phase == "play" else f"{t('mode', lang)}: -",
            f"{t('particle', lang)}: {particle_txt} (1/2/3)" if self.phase == "play" else f"{t('particle', lang)}: -",
            f"{t('energy', lang)}: {E0_txt}" if self.phase == "play" else f"{t('energy', lang)}: -",
        ])
        
        # --- Block 3 (controls) ---
        if self.phase == "select_tumor":
            block3 = "\n".join([
                t("tumor_painting", lang) + ":",
                t("left_drag_add", lang),
                t("right_drag_erase", lang),
                t("enter_confirm", lang),
                t("t_back_later", lang),
                t("back_to_menu", lang),
                t("esc_quit", lang),
            ])
        else:
            block3 = "\n".join([
                t("controls", lang) + ":",
                t("click_start", lang),
                t("drag_aim", lang),
                t("enter_fire", lang),
                t("r_reset", lang),
                t("n_new_game", lang),
                t("back_to_menu", lang),
                t("esc_quit", lang),
            ])

        # Style color depends on particle
        c = PARTICLE_COLORS.get(self.particle_name, (1, 1, 1, 1))
        edge = (c[0], c[1], c[2], 0.9)

        # Clear panel axis
        axp = self.ax_hud_img
        axp.clear()
        axp.set_xticks([]); axp.set_yticks([])
        for s in axp.spines.values():
            s.set_visible(False)
        axp.set_facecolor("black")

        # One shared box background spanning full panel
        # (use an invisible text bbox by placing two texts with same bbox style)
        box_kwargs = dict(
            boxstyle="round,pad=0.6",
            facecolor=(0, 0, 0, 0.55),
            edgecolor=edge,
            linewidth=1.3
        )

        # Place block1 left, block3 right inside the SAME visual "box feel"
        axp.text(0.25, 0.88, block1, ha="left", va="top", color="white",
                fontsize=11, family="monospace", transform=axp.transAxes, bbox=box_kwargs)

        axp.text(0.6, 0.88, block3, ha="left", va="top", color="white",
                fontsize=11, family="monospace", transform=axp.transAxes, bbox=box_kwargs)

        self.fig.canvas.draw_idle()

    def _update_stats_panel(self):
        lang = self.language
        dt, dn = self.board.dose_stats()
        mnon = self.board.max_dose_non_tumor()
        mtum = self.board.min_dose_tumor()
        cv = self.board.tumor_dose_variability()

        ax = self.ax_stats
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor("black")

        # --- Draw the SAME rounded "glass" box as before, but manually ---
        box_left, box_bottom = 0.03, 0.12
        box_width, box_height = 0.94, 0.78

        box = FancyBboxPatch(
            (box_left, box_bottom),
            box_width, box_height,
            boxstyle="round,pad=0.02",
            transform=ax.transAxes,
            linewidth=1.0,
            facecolor=(1, 1, 1, 0.08),   # subtle white glass
            edgecolor=(1, 1, 1, 0.35),
        )
        ax.add_patch(box)

        # Layout inside the box
        x_label = box_left + 0.04
        x_value = box_left + box_width - 0.04
        y_top = box_bottom + box_height - 0.06
        dy = 0.15  # vertical spacing (axes fraction)

        label_style = dict(
            ha="left", va="top", fontsize=12, family="monospace",
            color="white", transform=ax.transAxes
        )
        value_style = dict(
            ha="right", va="top", fontsize=12, family="monospace",
            fontweight="bold", transform=ax.transAxes
        )

        lines = [
            (t("dose_tumor_avg", lang), dt, color_dose_tumor),
            (t("min_dose_tumor", lang), mtum, color_dose_tumor),
            (t("tumor_variability_cv", lang), cv, color_tumor_variability),
            (t("dose_non_tumor_avg", lang), dn, color_dose_non_tumor),
            (t("max_dose_non_tumor", lang), mnon, color_max_dose_non_tumor),
        ]

        y = y_top
        for label, val, cfn in lines:
            ax.text(x_label, y, f"{label}:", **label_style)
            ax.text(x_value, y, f"{val:.3f}", color=cfn(float(val)), **value_style)
            y -= dy

        self.fig.canvas.draw_idle()


    def _render_start_region(self):
        polys = []
        for qr, t in self.board.tiles.items():
            if t.can_start_beam:
                polys.append(
                    hex_corners_flat(self.board.hex_center_xy(qr), self.board.hex_size)
                )
        self._artists["start_region"].set_verts(polys)

    def _update_dose_layer(self):
        doses = np.array([self.board.tiles[qr].dose for qr in self._tile_keys], dtype=float)
        self._artists["dose_fill"].set_array(doses)

    def _autoscale_dose_colormap(self, eps: float = 1e-6):
        doses = np.array([t.dose for t in self.board.tiles.values()], dtype=float)
        if doses.size == 0:
            return
        dmin = float(doses.min())
        dmax = float(doses.max())
        if dmax - dmin < eps:
            dmin = 0.0
        self._artists["dose_fill"].set_clim(dmin, dmax)

    def _start_from_event(self, event):
        xy = np.array([event.xdata, event.ydata], dtype=float)
        h = self.board.pixel_to_hex(xy)

        if not self.board.inside_grid(h):
            return

        tile = self.board.get_tile(h)
        if tile is None or not tile.can_start_beam:
            return  # ❌ invalid start location

        self.start_hex = h
        center = self.board.hex_center_xy(h)

        self._artists["start_dot"].set_data([center[0]], [center[1]])
        self.arrow_end_xy = center.copy()
        self._artists["arrow"].xy = (center[0], center[1])
        self._artists["arrow"].set_position((center[0], center[1]))

    def _compute_path(self) -> list[Axial]:
        if self.start_hex is None or self.arrow_end_xy is None:
            return []
        return ray_path_to_border(self.board, self.start_hex, self.arrow_end_xy)
    
    def _render_path(self):
        path = self._compute_path()
        if not path:
            self._artists["preview_fill"].set_verts([])
            self._artists["preview_fill"].set_array(np.array([], dtype=float))
            return

        deposits = self._compute_preview_deposits(path)

        polys = [hex_corners_flat(self.board.hex_center_xy(h), self.board.hex_size) for h in path]
        vals = np.asarray(deposits, dtype=float)

        self._artists["preview_fill"].set_verts(polys)
        self._artists["preview_fill"].set_array(vals)

        vmax = float(max(1.0, vals.max()))
        self._artists["preview_fill"].set_clim(0.0, vmax)

    def _hide_dose_map(self):
        self._artists["dose_fill"].set_visible(False)
        self._artists["dose_cbar"].ax.set_visible(False)

    def _show_dose_map(self):
        self._artists["dose_fill"].set_visible(True)
        self._artists["dose_cbar"].ax.set_visible(True)

    def _clear_selection(self):
        self.start_hex = None
        self.arrow_end_xy = None
        self._artists["start_dot"].set_data([], [])
        #self._artists["hit_fill"].set_verts([])
        self._artists["preview_fill"].set_verts([])
        self._artists["preview_fill"].set_array(np.array([], dtype=float))
        self._artists["arrow"].xy = (0, 0)
        self._artists["arrow"].set_position((0, 0))

    def restart_game(self):
        """
        Full restart:
        - clears dose
        - clears tumor selection
        - returns to tumor selection phase
        - clears aiming artifacts
        - restores dose map view & rescales
        """
        # reset underlying board state
        self.board.reset_all()

        # UI state
        self.phase = "select_tumor"
        self.tumor_set = set()
        self._tumor_drag_mode = None

        # clear overlays
        self._clear_selection()
        if "tumor_fill" in self._artists:
            self._artists["tumor_fill"].set_verts([])

        # show dose map again (all zeros)
        self._show_dose_map()
        self._update_dose_layer()
        self._autoscale_dose_colormap()
        self._update_dvh() 

        # refresh HUD
        self._update_hud_panels()
        self._update_stats_panel()
        self.fig.canvas.draw_idle()

    def _apply_particle_style(self):
        """
        Update UI accents to match selected particle type.
        gamma: green, electron: blue, proton: red
        """
        c = PARTICLE_COLORS.get(self.particle_name, (1, 1, 1, 1))

        # start dot
        self._artists["start_dot"].set_color(c)
        self._artists["start_dot"].set_markeredgecolor("white")

        # arrow color
        self._artists["arrow"].arrow_patch.set_color(c)
        self._artists["arrow"].arrow_patch.set_linewidth(2.6)

        # hit highlight fill (transparent colored wash)
        #self._artists["hit_fill"].set_facecolor([_with_alpha(c, 0.22)])

    def _peak_index_from_arrow_end(self, path: list[Axial]) -> int:
        """
        Return index in path of the hex containing arrow_end_xy (or nearest fallback).
        """
        if not path or self.arrow_end_xy is None:
            return 0
        aim_hex = self.board.pixel_to_hex(self.arrow_end_xy)
        if aim_hex in path:
            return path.index(aim_hex)
        return len(path) - 1


    def _compute_energy_E0(self) -> int:
        """
        Convert arrow length (pixels) -> integer energy.
        E0 is proportional to arrow length.
        """
        if self.start_hex is None or self.arrow_end_xy is None:
            return 0
        sxy = self.board.hex_center_xy(self.start_hex)
        arrow_len_px = float(np.linalg.norm(self.arrow_end_xy - sxy))
        denom = max(1e-6, self.board.hex_size * self.energy_scale)
        E0 = int(round(arrow_len_px / denom))
        return max(1, E0)
        
    def _compute_preview_deposits(self, path: list[Axial]) -> list[int]:
        """
        Compute per-tile preview doses along a geometric ray path,
        respecting that air tiles do not receive dose and do not
        consume particle energy.
        """
        if not path:
            return []

        # Arrow length → energy
        E = self._compute_energy_E0()
        particle = self._particle_models[self.particle_name]

        # Identify material tiles (energy-consuming)
        material_tiles = []
        for h in path:
            t = self.board.get_tile(h)
            if t is not None and t.score_dose:
                material_tiles.append(h)

        if not material_tiles:
            # Entire ray is air
            return [0] * len(path)

        # Particle decides how dose is distributed over MATERIAL interactions
        if self.particle_name == "gamma":
            peak_idx_path = self._peak_index_from_arrow_end(path)
            peak_hex = path[peak_idx_path]
            # map peak_hex into material index (so air doesn't shift the peak unpredictably)
            if peak_hex in material_tiles:
                peak_idx_mat = material_tiles.index(peak_hex)
            else:
                peak_idx_mat = len(material_tiles) - 1
            material_deposits = particle.deposits(E, n_steps=len(material_tiles), peak_index=peak_idx_mat)
        else:
            material_deposits = particle.deposits(E, n_steps=len(material_tiles))

        # Map deposits back onto full geometric path
        out = []
        m_idx = 0
        for h in path:
            t = self.board.get_tile(h)
            if t is not None and t.score_dose:
                if m_idx < len(material_deposits):
                    out.append(int(material_deposits[m_idx]))
                    m_idx += 1
                else:
                    out.append(0)
            else:
                out.append(0)  # air

        return out

    # --- tumor volume ---
    def _render_tumor_overlay(self):
        if not self.tumor_set:
            self._artists["tumor_fill"].set_verts([])
            return
        polys = [hex_corners_flat(self.board.hex_center_xy(h), self.board.hex_size) for h in self.tumor_set]
        self._artists["tumor_fill"].set_verts(polys)

    def _tumor_apply_at_event(self, event, mode: str):
        if event.inaxes != self.ax or event.xdata is None:
            return
        xy = np.array([event.xdata, event.ydata], dtype=float)
        h = self.board.pixel_to_hex(xy)
        if not self.board.inside_grid(h):
            return
        if mode == "add":
            self.tumor_set.add(h)
        elif mode == "erase":
            self.tumor_set.discard(h)

    # --- DVH ---
    def _init_dvh_plot(self):
        """Initialize DVH axis style and line artists."""
        lang = self.language
        ax = self.ax_dvh
        ax.set_facecolor("black")
        for s in ax.spines.values():
            s.set_color((1, 1, 1, 0.35))
        ax.tick_params(colors="white", labelsize=9)
        self.ax_dvh.set_aspect("auto")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        self.ax_dvh.margins(x=0.02, y=0.02)

        ax.set_xlabel(t("dvh_dose_label", lang), color="white", fontsize=10)
        ax.set_ylabel(t("dvh_volume_label", lang), color="white", fontsize=10)
        ax.set_title(t("dvh_title", lang), color="white", fontsize=11, pad=8)
        
        # Lines (cumulative DVH curves)
        (self._artists["dvh_tumor"],) = ax.plot([], [], lw=2.2, color=(1.0, 0.2, 0.2, 1.0), 
                                                label=t("dvh_tumor", lang))
        (self._artists["dvh_normal"],) = ax.plot([], [], lw=2.2, color=(0.65, 1.0, 1.0, 0.95), 
                                                label=t("dvh_non_tumor", lang))

        leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
        leg.get_frame().set_facecolor((0, 0, 0, 0.6))
        leg.get_frame().set_edgecolor((1, 1, 1, 0.25))
        for text in leg.get_texts():
            text.set_color("white")

    def _compute_cumulative_dvh(self, doses: np.ndarray, maxdose: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Cumulative DVH: y(x) = % volume receiving dose >= threshold.
        Returns x_percent (0..100), y_percent (0..100).
        """
        x = np.linspace(0.0, 100.0, 101)  # 0..100%
        if doses.size == 0 or maxdose <= 0:
            return x, np.zeros_like(x)

        # Convert x% -> dose threshold in absolute dose units
        thr = (x / 100.0) * maxdose

        # For each threshold, fraction >= thr
        # Vectorized: compare (dose[:,None] >= thr[None,:])
        frac = (doses[:, None] >= thr[None, :]).mean(axis=0)
        y = 100.0 * frac
        return x, y

    def _update_dvh(self):
        """Recompute and redraw DVH curves (tumor vs non-tumor)."""
        # Use only material tiles (score_dose=True) for DVH
        tumor = np.array([t.dose for t in self.board.tiles.values() if t.tissue == 1 and t.score_dose], dtype=float)
        normal = np.array([t.dose for t in self.board.tiles.values() if t.tissue == 0 and t.score_dose], dtype=float)

        all_doses = np.array([t.dose for t in self.board.tiles.values() if t.score_dose], dtype=float)
        maxdose = float(all_doses.max()) if all_doses.size else 0.0
        if maxdose <= 0:
            maxdose = 1.0  # avoid divide-by-zero; x-axis still 0..100%

        x_t, y_t = self._compute_cumulative_dvh(tumor, maxdose)
        x_n, y_n = self._compute_cumulative_dvh(normal, maxdose)

        self._artists["dvh_tumor"].set_data(x_t, y_t)
        self._artists["dvh_normal"].set_data(x_n, y_n)

        self.ax_dvh.set_aspect("auto")
        self.ax_dvh.set_xlim(0, 100)
        self.ax_dvh.set_ylim(0, 100)
        self.ax_dvh.margins(x=0.02, y=0.02)

    def _go_back_to_menu(self):
        """Handle returning to menu."""
        if self.menu is not None:
            # Close current figure and call back to menu
            plt.close(self.fig)
            self.menu._create_ui()
            self.menu.show()
        else:
            # No callback, just show message
            print("Back to menu functionality not configured")


    # --- event handlers ---
    def _on_press(self, event):
        if self.phase == "select_tumor":
            if event.inaxes != self.ax or event.xdata is None:
                return
            if event.button == 1:
                self._tumor_drag_mode = "add"
            elif event.button == 3:
                self._tumor_drag_mode = "erase"
            else:
                return

            self._tumor_apply_at_event(event, self._tumor_drag_mode)
            self._render_tumor_overlay()
            self._update_hud_panels()
            self._update_stats_panel()
            self.fig.canvas.draw_idle()
            return
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None:
            return
        self._hide_dose_map()
        self.dragging = True
        self._start_from_event(event)
        self.fig.canvas.draw_idle()

    def _on_move(self, event):
        if self.phase == "select_tumor":
            if self._tumor_drag_mode is None:
                return
            self._tumor_apply_at_event(event, self._tumor_drag_mode)
            self._render_tumor_overlay()
            self._update_hud_panels()
            self._update_stats_panel()
            self.fig.canvas.draw_idle()
            return
        if not self.dragging or self.start_hex is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return

        self.arrow_end_xy = np.array([event.xdata, event.ydata], dtype=float)

        s = self.board.hex_center_xy(self.start_hex)
        self._artists["arrow"].xy = (self.arrow_end_xy[0], self.arrow_end_xy[1])
        self._artists["arrow"].set_position((s[0], s[1]))

        self._render_path()
        self.fig.canvas.draw_idle()
        self._update_hud_panels()
        self._update_stats_panel()

    def _on_release(self, event):
        if self.phase == "select_tumor":
            self._tumor_drag_mode = None
            return
        self.dragging = False

    def _on_key(self, event):

        if event.key in ("1", "2", "3"):
            self.particle_idx = int(event.key) - 1
            self.particle_name = self.particles[self.particle_idx]
            self._apply_particle_style()
            self._update_hud_panels()
            self._update_stats_panel()
            return

        elif event.key.lower() == "m":
            self._go_back_to_menu()
            #self.mode = "free" if self.mode == "6dir" else "6dir"
            #self._render_path()
            #self._update_hud_panels()
            #self._update_stats_panel()

        elif event.key.lower() == "n":
            self.restart_game()
            return
        
        elif event.key.lower() == "r":
            self._clear_selection()
            self._show_dose_map()
            self.fig.canvas.draw_idle()

        elif event.key == "enter":
            if self.phase == "select_tumor":
                if len(self.tumor_set) == 0:
                    return  # require at least 1 tile
                self.board.set_tumor_tiles(self.tumor_set)
                self.phase = "play"
                self._tumor_drag_mode = None

                # show dose map now that we're in play mode
                self._show_dose_map()
                self._update_hud_panels()
                self._update_stats_panel()
                self.fig.canvas.draw_idle()
                self._update_dvh() 
                return
            path = self._compute_path()
            if self.on_fire is not None and path and self.start_hex is not None:
                E0 = self._compute_energy_E0()
                deposits = self._compute_preview_deposits(path)
                self.on_fire(path, deposits)

                self._update_dose_layer()
                self._autoscale_dose_colormap()
                self._update_dvh() 

                self._clear_selection()
                self._show_dose_map()
                self.fig.canvas.draw_idle()
                self._update_hud_panels()
                self._update_stats_panel()

        elif event.key == "escape":
            plt.close(self.fig)
        
        elif event.key.lower() == "t":
            # toggle tumor edit mode
            if self.phase == "play":
                self.phase = "select_tumor"
                # initialize tumor_set from board labels (in case shots happened)
                self.tumor_set = self.board.tumor_tiles()
                self._clear_selection()
                self._hide_dose_map()
            else:
                # leaving tumor edit without confirming changes:
                # keep selection phase, or you can decide to go back to play only via Enter
                pass
            self._render_tumor_overlay()
            self._update_hud_panels()
            self._update_stats_panel()
            self.fig.canvas.draw_idle()



    def show(self):
        plt.show()
