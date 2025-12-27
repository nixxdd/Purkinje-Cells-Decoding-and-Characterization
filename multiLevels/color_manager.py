import pickle
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
from pathlib import Path

class ColorSchemeManager:
    def __init__(self, base_path='/home/nix/Uni/3Year/Thesis/testStefania/multiLevels'):
        self.base_path = Path(base_path)
    
    def save_color_scheme(self, colors_raw, labels_data, labels_display, 
                         label_colors, mean_fr_unique, name='default'):
        """Save a color scheme with a given name"""
        color_scheme = {
            'colors_raw': colors_raw,
            'labels_data': labels_data,
            'labels_display': labels_display,
            'label_colors': label_colors
        }
        
        filename = f'color_scheme_{name}.pkl'
        save_path = self.base_path / filename
        
        with open(save_path, 'wb') as f:
            pickle.dump(color_scheme, f)
        
        print(f"Color scheme '{name}' saved to {save_path}")
        return save_path
    
    def load_color_scheme(self, name='default'):
        """Load a color scheme by name"""
        filename = f'color_scheme_{name}.pkl'
        load_path = self.base_path / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Color scheme '{name}' not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            color_scheme = pickle.load(f)
        
        print(f"Color scheme '{name}' loaded from {load_path}")
        return color_scheme
    
    def list_available_schemes(self):
        """List all available color schemes"""
        scheme_files = list(self.base_path.glob('color_scheme_*.pkl'))
        schemes = [f.stem.replace('color_scheme_', '') for f in scheme_files]
        return schemes

color_manager = ColorSchemeManager()