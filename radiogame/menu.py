"""
Menu screen for radiotherapy game with language and image selection.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict
from PIL import Image

from .translations import t


class MenuScreen:
    """Interactive menu for selecting language and medical image."""
    
    def __init__(self, images_folder: str = "images"):
        """
        Initialize menu screen.
        
        Args:
            images_folder: Path to folder containing medical images
        """
        self.images_folder = Path(images_folder)
        self.selected_language = "en"
        self.selected_image = None
        self.on_start_callback: Optional[Callable] = None
        
        # Scrolling state
        self.scroll_offset = 0  # ✅ NEW - tracks scroll position
        self.items_per_page = 5  # ✅ NEW - how many images to show at once
        
        # Scan for available images
        self.available_images = self._scan_images()
        if self.available_images:
            self.selected_image = self.available_images[0]
        
        # Create UI
        self.fig = None
        self.radio_lang = None
        self.image_buttons = []
        self._create_ui()
    
    def _scan_images(self) -> list[str]:
        """Scan images folder for available medical images."""
        if not self.images_folder.exists():
            print(f"Warning: Images folder '{self.images_folder}' not found")
            # Try relative path from radiogame folder
            alt_path = Path(__file__).parent.parent / "images"
            if alt_path.exists():
                self.images_folder = alt_path
            else:
                return []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = []
        
        for file in self.images_folder.iterdir():
            if file.is_file() and file.suffix.lower() in valid_extensions:
                images.append(file.name)
        
        return sorted(images)
    
    def _create_ui(self):
        """Create the menu UI."""
        self.fig = plt.figure(figsize=(10, 7), facecolor='#0a0a0a')
        self.fig.canvas.manager.set_window_title(t("menu_title", self.selected_language))
        
        # Title
        self.title_text = self.fig.text(
            0.5, 0.94,
            t("menu_title", self.selected_language),
            ha='center', va='top', fontsize=26, fontweight='bold',
            color='white', family='sans-serif'
        )
        
        self.subtitle_text = self.fig.text(
            0.5, 0.89,
            t("menu_subtitle", self.selected_language),
            ha='center', va='top', fontsize=12, color='#00d4ff',
            family='sans-serif', style='italic'
        )
        
        # Language selection area
        self.ax_lang = self.fig.add_axes([0.08, 0.70, 0.25, 0.12])
        self.ax_lang.set_facecolor('#1a1a1a')
        for spine in self.ax_lang.spines.values():
            spine.set_color('#00d4ff')
            spine.set_linewidth(2)
        
        self.lang_label = self.ax_lang.text(
            0.5, 0.85, t("language_label", self.selected_language) + ":",
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='white', transform=self.ax_lang.transAxes
        )
        
        # Radio buttons for language
        radio_ax = self.fig.add_axes([0.10, 0.71, 0.20, 0.08])
        radio_ax.set_facecolor('#1a1a1a')
        self.radio_lang = RadioButtons(
            radio_ax, 
            ('English', 'Nederlands'),
            active=0 if self.selected_language == "en" else 1
        )
        
        # Style labels
        for label in self.radio_lang.labels:
            label.set_color('white')
            label.set_fontsize(10)
        
        self.radio_lang.on_clicked(self._on_language_change)
        
        # Image selection area
        self.ax_images = self.fig.add_axes([0.08, 0.20, 0.45, 0.42])
        self.ax_images.set_facecolor('#1a1a1a')
        for spine in self.ax_images.spines.values():
            spine.set_color('#00d4ff')
            spine.set_linewidth(2)
        
        # ✅ NEW - Scroll buttons
        self.ax_scroll_up = self.fig.add_axes([0.515, 0.58, 0.03, 0.03])
        self.btn_scroll_up = Button(self.ax_scroll_up, '▲', color='#00d4ff', hovercolor='#00ffff')
        self.btn_scroll_up.label.set_fontsize(10)
        self.btn_scroll_up.label.set_color('black')
        self.btn_scroll_up.on_clicked(self._scroll_up)
        
        self.ax_scroll_down = self.fig.add_axes([0.515, 0.22, 0.03, 0.03])
        self.btn_scroll_down = Button(self.ax_scroll_down, '▼', color='#00d4ff', hovercolor='#00ffff')
        self.btn_scroll_down.label.set_fontsize(10)
        self.btn_scroll_down.label.set_color('black')
        self.btn_scroll_down.on_clicked(self._scroll_down)
        
        # ✅ NEW - Scroll indicator text
        self.scroll_indicator = self.fig.text(
            0.53, 0.40, "",
            ha='center', va='center', fontsize=8, color='#888888',
            family='monospace'
        )
        
        # Image preview area
        self.ax_preview = self.fig.add_axes([0.58, 0.20, 0.36, 0.42])
        self.ax_preview.set_facecolor('#000000')
        for spine in self.ax_preview.spines.values():
            spine.set_color('#00d4ff')
            spine.set_linewidth(2)
        
        # Description text area
        self.desc_text = self.fig.text(
            0.5, 0.14, "",
            ha='center', va='top', fontsize=10, color='white',
            wrap=True, family='sans-serif'
        )
        
        # Start button
        self.ax_button = self.fig.add_axes([0.38, 0.05, 0.24, 0.06])
        self.start_button = Button(
            self.ax_button,
            t("start_game", self.selected_language),
            color='#00d4ff',
            hovercolor='#00ffff'
        )
        self.start_button.label.set_color('black')
        self.start_button.label.set_fontsize(14)
        self.start_button.label.set_fontweight('bold')
        self.start_button.on_clicked(self._on_start_clicked)
        
        # ✅ NEW - Connect scroll wheel event
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll_wheel)
        
        self._update_image_list()
        self._update_preview()
        self._update_scroll_indicator()
    
    def _scroll_up(self, event):
        """Scroll up in the image list."""
        if self.scroll_offset > 0:
            self.scroll_offset -= 1
            self._update_image_list()
            self._update_scroll_indicator()
    
    def _scroll_down(self, event):
        """Scroll down in the image list."""
        max_offset = max(0, len(self.available_images) - self.items_per_page)
        if self.scroll_offset < max_offset:
            self.scroll_offset += 1
            self._update_image_list()
            self._update_scroll_indicator()
    
    def _on_scroll_wheel(self, event):
        """Handle mouse wheel scrolling."""
        if event.inaxes == self.ax_images:
            if event.button == 'up':
                self._scroll_up(None)
            elif event.button == 'down':
                self._scroll_down(None)
    
    def _update_scroll_indicator(self):
        """Update the scroll position indicator."""
        if len(self.available_images) <= self.items_per_page:
            self.scroll_indicator.set_text("")
            return
        
        start_idx = self.scroll_offset + 1
        end_idx = min(self.scroll_offset + self.items_per_page, len(self.available_images))
        total = len(self.available_images)
        
        self.scroll_indicator.set_text(f"{start_idx}-{end_idx}\nof\n{total}")
        self.fig.canvas.draw_idle()
    
    def _update_image_list(self):
        """Update the list of available images with scrolling."""
        self.ax_images.clear()
        self.ax_images.set_xlim(0, 1)
        self.ax_images.set_ylim(0, 1)
        self.ax_images.set_xticks([])
        self.ax_images.set_yticks([])
        
        # Title
        self.ax_images.text(
            0.5, 0.96,
            t("select_image", self.selected_language),
            ha='center', va='top', fontsize=12, fontweight='bold',
            color='white', transform=self.ax_images.transAxes
        )
        
        if not self.available_images:
            self.ax_images.text(
                0.5, 0.5,
                t("no_images_found", self.selected_language),
                ha='center', va='center', fontsize=10, color='red',
                transform=self.ax_images.transAxes
            )
            return
        
        # ✅ Calculate visible slice
        start_idx = self.scroll_offset
        end_idx = min(start_idx + self.items_per_page, len(self.available_images))
        visible_images = self.available_images[start_idx:end_idx]
        
        # Display visible images
        y_start = 0.85
        y_spacing = 0.15
        
        for i, img_name in enumerate(visible_images):
            y_pos = y_start - i * y_spacing
            
            # Highlight selected image
            if img_name == self.selected_image:
                rect = Rectangle(
                    (0.05, y_pos - 0.06), 0.90, 0.11,
                    transform=self.ax_images.transAxes,
                    facecolor='#00d4ff', edgecolor='none', alpha=0.3
                )
                self.ax_images.add_patch(rect)
            
            # Image name text (clickable)
            text = self.ax_images.text(
                0.08, y_pos,
                f"📄 {img_name}",
                ha='left', va='center', fontsize=10,
                color='white' if img_name == self.selected_image else '#aaaaaa',
                transform=self.ax_images.transAxes,
                picker=5
            )
            text.image_name = img_name
        
        # Connect click event
        self.fig.canvas.mpl_connect('pick_event', self._on_image_pick)
    
    def _update_preview(self):
        """Update the image preview."""
        self.ax_preview.clear()
        self.ax_preview.set_xticks([])
        self.ax_preview.set_yticks([])
        
        if not self.selected_image:
            self.ax_preview.text(
                0.5, 0.5, "No image selected",
                ha='center', va='center', fontsize=10, color='gray'
            )
            self.desc_text.set_text("")
            return
        
        # Load and display image
        img_path = self.images_folder / self.selected_image
        try:
            img = Image.open(img_path)
            self.ax_preview.imshow(img, cmap='gray', aspect='auto')
            
            # Update description
            desc = t(self.selected_image, self.selected_language)
            self.desc_text.set_text(desc)
            
        except Exception as e:
            self.ax_preview.text(
                0.5, 0.5, f"Error loading image:\n{str(e)}",
                ha='center', va='center', fontsize=9, color='red'
            )
            self.desc_text.set_text("")
        
        self.fig.canvas.draw_idle()
    
    def _on_language_change(self, label):
        """Handle language selection change."""
        self.selected_language = "en" if label == "English" else "nl"
        
        # Update all text elements
        self.title_text.set_text(t("menu_title", self.selected_language))
        self.subtitle_text.set_text(t("menu_subtitle", self.selected_language))
        self.lang_label.set_text(t("language_label", self.selected_language) + ":")
        self.start_button.label.set_text(t("start_game", self.selected_language))
        self.fig.canvas.manager.set_window_title(t("menu_title", self.selected_language))
        
        # Refresh image list and preview
        self._update_image_list()
        self._update_preview()
        self.fig.canvas.draw_idle()
    
    def _on_image_pick(self, event):
        """Handle image selection from list."""
        if hasattr(event.artist, 'image_name'):
            self.selected_image = event.artist.image_name
            self._update_image_list()
            self._update_preview()
    
    def _on_start_clicked(self, event):
        """Handle start button click."""
        if self.on_start_callback and self.selected_image:
            plt.close(self.fig)
            self.on_start_callback(self.selected_image, self.selected_language)
    
    def show(self):
        """Display the menu screen."""
        plt.show()
    
    def set_start_callback(self, callback: Callable):
        """
        Set callback function to be called when Start Game is clicked.
        
        Args:
            callback: Function(image_path: str, language: str)
        """
        self.on_start_callback = callback


def show_menu(on_start_callback: Optional[Callable] = None) -> MenuScreen:
    """
    Show menu screen and return MenuScreen instance.
    
    Args:
        on_start_callback: Optional callback(image_path, language)
    
    Returns:
        MenuScreen instance
    """
    menu = MenuScreen()
    if on_start_callback:
        menu.set_start_callback(on_start_callback)
    menu.show()
    return menu

