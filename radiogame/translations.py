"""
Centralized translations for the radiotherapy game.
English and Dutch language support.
"""

TRANSLATIONS = {
    "en": {
        # Menu screen
        "menu_title": "Radiotherapy Game",
        "menu_subtitle": "Educational Treatment Planning Simulator",
        "language_label": "Language",
        "select_image": "Select Medical Image:",
        "start_game": "Start Game",
        "no_images_found": "No images found in images folder",
        
        # Image descriptions (add as you add more images)
        "brain.jpg": "Brain tumor - Deep-seated glioblastoma requiring penetrating radiation",
        
        # Game UI - Phase labels
        "phase_tumor_select": "TUMOR SELECT",
        "phase_treatment": "TREATMENT",
        
        # Main HUD panel
        "phase": "PHASE",
        "mode": "MODE",
        "particle": "PARTICLE",
        "energy": "E0",
        "mode_free": "FREE",
        "mode_6dir": "6-DIR",
        
        # Particle names
        "gamma": "GAMMA",
        "electron": "ELECTRON", 
        "proton": "PROTON",
        
        # Controls section
        "controls": "Controls",
        "tumor_painting": "Tumor painting",
        "left_drag_add": "• Left drag: ADD",
        "right_drag_erase": "• Right drag: ERASE",
        "enter_confirm": "• Enter: confirm",
        "t_back_later": "• T: back later",
        "esc_quit": "• Esc: quit",
        "click_start": "• Click: start",
        "drag_aim": "• Drag: aim",
        "enter_fire": "• Enter: fire",
        "r_reset": "• R: reset",
        "n_new_game": "• N: new game",
        "back_to_menu": "• M: back to menu",
        "confirm_back_menu": "Return to menu? (Unsaved progress will be lost)",
        
        # Stats panel
        "dose_tumor_avg": "Dose tumor (avg)",
        "min_dose_tumor": "Min dose tumor",
        "tumor_variability_cv": "Tumor variability (CV)",
        "dose_non_tumor_avg": "Dose non-tumor (avg)",
        "max_dose_non_tumor": "Max dose non-tumor",
        
        # DVH
        "dvh_title": "DVH",
        "dvh_dose_label": "Dose (% of max)",
        "dvh_volume_label": "Volume (% of structure)",
        "dvh_tumor": "Tumor",
        "dvh_non_tumor": "Non-tumor",
    },
    
    "nl": {
        # Menu screen
        "menu_title": "Radiotherapie Spel",
        "menu_subtitle": "Educatieve Behandelplanningssimulator",
        "language_label": "Taal",
        "select_image": "Selecteer Medische Afbeelding:",
        "start_game": "Start Spel",
        "no_images_found": "Geen afbeeldingen gevonden in images map",
        
        # Image descriptions (add as you add more images)
        "brain.jpg": "Hersentumor - Diepgelegen glioblastoom vereist penetrerende straling",
        
        # Game UI - Phase labels
        "phase_tumor_select": "TUMOR SELECTIE",
        "phase_treatment": "BEHANDELING",
        
        # Main HUD panel
        "phase": "FASE",
        "mode": "MODUS",
        "particle": "DEELTJE",
        "energy": "E0",
        "mode_free": "VRIJ",
        "mode_6dir": "6-RICHT",
        
        # Particle names
        "gamma": "GAMMA",
        "electron": "ELEKTRON",
        "proton": "PROTON",
        
        # Controls section
        "controls": "Bediening",
        "tumor_painting": "Tumor tekenen",
        "left_drag_add": "• Links slepen: TOEVOEGEN",
        "right_drag_erase": "• Rechts slepen: WISSEN",
        "enter_confirm": "• Enter: bevestigen",
        "t_back_later": "• T: later terug",
        "esc_quit": "• Esc: afsluiten",
        "click_start": "• Klik: start",
        "drag_aim": "• Sleep: richt",
        "enter_fire": "• Enter: vuur",
        "r_reset": "• R: reset",
        "n_new_game": "• N: nieuw spel",
        "back_to_menu": "• M: terug naar menu",
        "confirm_back_menu": "Terug naar menu? (Voortgang gaat verloren)",
        
        # Stats panel
        "dose_tumor_avg": "Dosis tumor (gem)",
        "min_dose_tumor": "Min dosis tumor",
        "tumor_variability_cv": "Tumor variabiliteit (CV)",
        "dose_non_tumor_avg": "Dosis niet-tumor (gem)",
        "max_dose_non_tumor": "Max dosis niet-tumor",
        
        # DVH
        "dvh_title": "DVH",
        "dvh_dose_label": "Dosis (% van max)",
        "dvh_volume_label": "Volume (% van structuur)",
        "dvh_tumor": "Tumor",
        "dvh_non_tumor": "Niet-tumor",
    }
}


def t(key: str, lang: str = "en") -> str:
    """
    Get translated text for a given key.
    
    Args:
        key: Translation key
        lang: Language code ("en" or "nl")
    
    Returns:
        Translated string, or the key itself if not found
    """
    if lang not in TRANSLATIONS:
        lang = "en"
    return TRANSLATIONS[lang].get(key, key)
