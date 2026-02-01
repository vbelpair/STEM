# Radiotherapy Game Board (STEM Project)

An educational game that simulates radiotherapy treatment planning. Players overlay a hexagonal grid on medical images (CT/MRI scans) and strategically deliver radiation beams to maximize dose in tumor tissue while minimizing damage to healthy tissue.

## 🎮 Features

### Core Gameplay
- **Multi-language support**: Play in English or Dutch (Nederlands)
- **Image selection menu**: Choose from multiple medical image scenarios
- **Interactive tumor segmentation**: Paint tumor regions before treatment
- **Three particle types**: 
  - **Gamma rays**: Buildup effect with maximum at target depth
  - **Electrons**: Rapid dose deposition near surface (ideal for skin cancer)
  - **Protons**: Bragg peak targeting for deep-seated tumors
- **Real-time dose visualization**: Color-coded dose map with adjustable colorbar
- **DVH (Dose-Volume Histogram)**: Track tumor vs. healthy tissue dose distribution
- **Intelligent start regions**: Beams can only enter from outside the patient

### Interactive UI
- **Tumor Selection Phase**:
  - Left-click drag to paint tumor regions
  - Right-click drag to erase
  - Visual overlay shows selected tumor area
  
- **Treatment Planning Phase**:
  - Click to select beam start location (air regions only)
  - Drag to aim beam direction and set energy (arrow length = energy)
  - Preview shows predicted dose deposition along beam path
  - Press Enter to fire the beam
  - Real-time statistics: dose metrics, variability, DVH curves

### Controls
- **1/2/3**: Switch particle type (Gamma/Electron/Proton)
- **M**: Toggle aiming mode (Free/6-direction)
- **Enter**: Confirm tumor selection / Fire beam
- **R**: Reset current beam
- **N**: New game (reset all)
- **B**: Back to menu
- **T**: Return to tumor editing
- **Esc**: Quit

## 📦 Installation

### Prerequisites
Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies
```bash
pip install numpy matplotlib SimpleITK pillow
```

### Required Packages
- **numpy**: Numerical computations and array operations
- **matplotlib**: Interactive UI and visualization
- **SimpleITK**: Medical image loading (supports DICOM, NIfTI, JPG, PNG)
- **pillow**: Image processing for menu thumbnails

## 🚀 Quick Start

1. **Add medical images**: Place your medical images (JPG, PNG, DICOM) in the `images/` folder

2. **Run the game**:
```bash
python main.py
```

3. **Select language and image** in the menu screen

4. **Paint the tumor region** using left-click drag

5. **Press Enter** to start treatment planning

6. **Fire beams** by clicking start position, dragging to aim, and pressing Enter

## 📁 Project Structure

```
STEM/
├── images/                    # Medical images folder
│   ├── brain.jpg             # Brain tumor example
│   └── [your images]         # Add more images here
│
├── radiogame/                # Main package
│   ├── __init__.py          # Package initialization
│   ├── menu.py              # Menu screen with language/image selection
│   ├── translations.py      # English/Dutch translations
│   ├── hexgrid.py           # Hexagonal coordinate system & geometry
│   ├── board.py             # Game board, dose tracking, statistics
│   ├── particles.py         # Radiation particle physics models
│   ├── ui.py                # Interactive Matplotlib UI
│   └── io_utils.py          # Image loading and processing
│
└── main.py                  # Entry point
```

## 🎯 Game Objectives

Maximize your score by:
- ✅ **Maximizing** average dose in tumor tissue
- ✅ **Minimizing** maximum dose in healthy tissue  
- ✅ **Minimizing** dose variability within tumor (uniform coverage)
- ✅ Delivering **minimum therapeutic dose** to all tumor regions

## 🔬 Physics Models

### Gamma Rays
- Buildup region with maximum at user-defined depth
- Exponential attenuation after peak
- Good for deep-seated tumors with penetration needed

### Electrons
- Rapid dose deposition near surface
- Short effective range (~ 0.25 × energy)
- Ideal for superficial lesions (skin cancer, melanoma)

### Protons
- Low entrance dose
- Sharp Bragg peak near end of range (~ 0.6 × energy)
- Minimal exit dose beyond target
- Best for sparing tissues beyond tumor

## 🌍 Adding More Images

1. **Place images** in the `images/` folder (JPG, PNG, DICOM, NIfTI supported)

2. **Add descriptions** in `radiogame/translations.py`:
```python
TRANSLATIONS = {
    "en": {
        "your_image.jpg": "Description in English",
        # ...
    },
    "nl": {
        "your_image.jpg": "Beschrijving in het Nederlands",
        # ...
    }
}
```

3. **Recommended image types**:
   - Brain tumors (deep, requires protons/gamma)
   - Skin cancer (surface, ideal for electrons)
   - Lung nodules (various depths)
   - Prostate cancer (deep pelvic region)

## ⚙️ Configuration

Adjust game parameters in `main.py`:

```python
board = GameBoard(
    sitk_image,
    hex_size_px=18.0,        # Hexagon size in pixels
    air_threshold=80.0,      # Intensity threshold for air (0-255)
    edge_margin=0.5          # Start region margin (in hex units)
)
```

## 🎓 Educational Value

This game demonstrates fundamental concepts in radiation oncology:
- **Treatment planning**: Beam angle and energy selection
- **Dose optimization**: Balancing tumor coverage vs. normal tissue sparing
- **Particle selection**: Understanding different radiation modalities
- **DVH analysis**: Clinical tool for evaluating treatment plans

## 📝 Notes

- This is an **educational prototype** focused on interactive mechanics
- Physics models are simplified for gameplay (not clinically accurate)
- Intensity values are normalized for visualization (not absolute dose)
- Multi-player and advanced scoring features may be added in future versions

## 🛠️ Making it Portable

To create a standalone executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --add-data "images:images" --add-data "radiogame:radiogame" main.py
```

The executable will be in the `dist/` folder.

## 📄 License

Educational use - STEM project

## 👥 Contributing

This is an educational project. Suggestions for improvements are welcome!

---

**Made with ❤️ for radiation oncology education**