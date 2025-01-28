import cv2
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path

def analyze_mask_classes(mask_dir):
    """Analyze all mask images in directory to find unique pixel values"""
    all_values = set()
    mask_path = Path(mask_dir)
    
    # Get list of files first for tqdm
    mask_files = list(mask_path.glob('**/*.png'))
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask = cv2.imread(str(mask_file))
        mask = mask[:,:,0]  # Take first channel
        values = np.unique(mask)
        all_values.update(values)
    
    print(f"Path: {mask_dir}")    
    print(f"Unique pixel values: {sorted(list(all_values))}")
    print(f"Number of classes: {len(all_values)}\n")

for i in ['/workspace/bdd100k/labels/drivable/colormaps', '/workspace/bdd100k/labels/lane/colormaps']:
    # Usage example:
    analyze_mask_classes(i)
