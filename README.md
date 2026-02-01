# Radiotherapy Game Board (STEM Project)

A small educational game prototype that overlays a hexagonal grid on a medical image (e.g., brain CT/MRI).  
Players choose a beam start location and direction to deliver “dose” to tiles, aiming to maximize dose in tumor tissue and minimize dose in healthy tissue.

## Features (current)
- Loads a 2D image using SimpleITK
- Generates a flat-top hex grid covering the image
- Interactive Matplotlib UI:
  - click to select beam start location (snapped to hex)
  - drag to draw an arrow/beam segment
  - highlights hex tiles touched by the segment
  - press Enter to “fire” and deposit dose
  - dose map reappears and auto-rescales

## Installation
Recommended: create a virtual environment.

```bash
pip install numpy matplotlib SimpleITK
````

## Run

Place an image named `CT_scan.jpg` in the project folder (or update `main.py`).

```bash
python main.py
```

## Project structure

* `radiogame/hexgrid.py`
  Hex math: axial coords, pixel transforms (flat-top), RedBlob line utilities, “touched hexes by segment”.
* `radiogame/board.py`
  `Tile` + `GameBoard` (dose storage and grid generation).
* `radiogame/particles.py`
  Particle classes (Gamma/Electron/Proton) — currently basic placeholders.
* `radiogame/ui.py`
  `BeamPickerUI` (Matplotlib-based interactive aiming + dose visualization).
* `radiogame/io_utils.py`
  Image loading, 2D conversion, simple display normalization.
* `main.py`
  Entry point wiring everything together.

## TODO / Next steps

* Create good particle sources and realistic dose deposition models
* Add particle selection controls in the GUI (gamma/electron/proton + energy slider)
* Add grid modifiers:

  * tissue-dependent absorption
  * scattering
  * stochastic stopping / range
  * IDEE MEISJE: meerdere pijlen tegelijkertijd maken en laten afvure: komt op neer meerder paden opslaan en elk pad aflopen (voorzie ook verwijder knop)
  * IDEE MEISJE: pad moet dose profiel tonen inplaats van enkel pad highlighten
* Implement tumor/healthy reward function:

  * reward = dose(tumor) − λ * dose(healthy)
* Support tumor masks / segmentation overlays
* Add multi-player turn logic and scoring

## Notes

This project currently focuses on a clean architecture and interactive aiming mechanics.
Clinical accuracy is NOT a goal at this stage; physics and reward modeling will be introduced later as “game modifiers”.

```

---

## Small extra improvements you get “for free” with this split
- No duplicated `hex_corners_flat` / no duplicated imports
- Utilities don’t depend on UI, UI doesn’t own board logic
- `main.py` is the only executable script
- You can now write tests later (e.g., `tests/test_hexgrid.py`) without dragging Matplotlib into everything

---

If you want, I can also provide a tiny `pyproject.toml` or `setup.cfg` so you can do `pip install -e .` and run it as a package — but the layout above already works with plain `python main.py` as-is.
