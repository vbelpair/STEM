import SimpleITK as sitk

from radiogame.io_utils import load_image, image_to_array_2d, normalize_for_display
from radiogame.board import GameBoard
from radiogame.ui import BeamPickerUI
from radiogame.particles import Proton, Electron, Gamma

def main():
    print("Game is starting")

    img = load_image("images/brain.jpg")
    arr_raw = image_to_array_2d(img)          # raw (H,W)
    arr_disp = normalize_for_display(arr_raw) # 0..1 for display

    board = GameBoard(img, hex_size_px=12, image_array_2d=arr_raw, air_threshold=50.0)

    models = {
        "proton": Proton(),
        "electron": Electron(),
        "gamma": Gamma(),
    }

    def on_fire_callback(path, deposits):

        j = 0  # index into deposits (energy consumption happens only in material)
        for h, d in zip(path,deposits):
            tile = board.get_tile(h)
            if tile is None:
                continue

            if not tile.score_dose:
                # air: no dose and no energy loss
                continue

            if j >= len(deposits):
                break

            if d > 0: 
                board.add_dose(h, d)
            j += 1

        # optional: print stats after each shot
        dt, dn = board.dose_stats()
        print(f"dose_tumor={dt:.3f} | dose_non_tumor={dn:.3f}")


    ui = BeamPickerUI(board, arr_disp, on_fire=on_fire_callback, show_grid=True, particle_models=models)
    ui.show()

if __name__ == "__main__":
    main()
