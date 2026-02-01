"""
Main entry point for radiotherapy game.
Shows menu, then launches game with selected settings.
"""

import sys
from pathlib import Path

# Add radiogame to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from radiogame.menu import MenuScreen
from radiogame.io_utils import load_image, image_to_array_2d, normalize_for_display
from radiogame.board import GameBoard
from radiogame.particles import Proton, Electron, Gamma
from radiogame.ui import BeamPickerUI


def start_game(image_filename: str, language: str, menu_instance: MenuScreen = None):
    """
    Start the game with selected image and language.
    
    Args:
        image_filename: Name of the selected medical image
        language: Selected language code ("en" or "nl")
        menu_instance: Reference to menu to return to
    """
    print(f"Starting game with image: {image_filename}, language: {language}")
    
    # Load the selected image
    image_path = Path("images") / image_filename
    
    try:
        sitk_image = load_image(str(image_path))
        image_array_2d = image_to_array_2d(sitk_image)
        display_array = normalize_for_display(image_array_2d)
        
        # Create game board
        hex_size = 18.0
        board = GameBoard(
            sitk_image,
            hex_size_px=hex_size,
            image_array_2d=image_array_2d,
            air_threshold=50.0
        )
        
        # Create particle models
        particle_models = {
            "proton": Proton(),
            "electron": Electron(),
            "gamma": Gamma()
        }
        
        # Fire callback
        def on_fire(path, deposits):
            """Handle beam firing."""
            for qr, dose_val in zip(path, deposits):
                board.add_dose(qr, dose_val)
        
        # Create and show UI with selected language and back callback
        ui = BeamPickerUI(
            board,
            display_array,
            on_fire=on_fire,
            show_grid=True,
            particle_models=particle_models,
            language=language,
            menu=menu_instance  # ✅ NIEUW
        )
        
        ui.show()
        
    except Exception as e:
        print(f"Error starting game: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point with menu loop."""
    menu = MenuScreen()
    
    def start_with_menu_ref(image_filename: str, language: str):
        """Wrapper to pass menu reference."""
        start_game(image_filename, language, menu)
    
    menu.set_start_callback(start_with_menu_ref)
    menu.show()


if __name__ == "__main__":
    main()
