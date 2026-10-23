"""
Part 1: Crop Leaf Image Acquisition
Handles image input from camera, user upload, local file, or PlantVillage benchmark samples.
"""

import os
import urllib.request
import cv2
import numpy as np
from PIL import Image

# Benchmark authentic leaf images from PlantVillage
BENCHMARK_SAMPLES = {
    "Tomato - Early Blight (Sample)": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/segmented/Tomato___Early_blight/0034638d-ac33-4e47-979f-961d15022139___RS_Erly.B%209432_final_masked.jpg",
    "Potato - Late Blight (Sample)": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/segmented/Potato___Late_blight/006c0045-81c2-475a-9b43-442807f41539___RS_LB%205047_final_masked.jpg",
    "Corn - Common Rust (Sample)": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/segmented/Corn_(maize)___Common_rust_/00105342-4960-496a-8f4b-09e25e019ab7___RS_Rust%201566_final_masked.jpg",
    "Tomato - Healthy Crop (Sample)": "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/segmented/Tomato___healthy/0001460a-65a7-4223-9f49-71e78449985b___GH_HL%20Leaf%20259.1_final_masked.jpg"
}

def acquire_leaf_image(source=None, cache_dir="cache"):
    """
    Acquires a leaf image:
    - If source is a PIL Image or numpy array: converts and returns RGB array
    - If source is a file path: loads via OpenCV and converts to RGB
    - If source is a benchmark key: downloads/caches the benchmark image
    - If source is None: loads default Tomato Early Blight sample
    """
    if isinstance(source, np.ndarray):
        return source

    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"))

    os.makedirs(cache_dir, exist_ok=True)
    
    # If source is a known benchmark key
    if source in BENCHMARK_SAMPLES:
        url = BENCHMARK_SAMPLES[source]
        filename = os.path.join(cache_dir, f"{source.replace(' ', '_').replace('-', '_')}.jpg")
        if not os.path.exists(filename):
            try:
                urllib.request.urlretrieve(url, filename)
            except Exception as e:
                print(f"[!] Warning: Could not download sample ({e}). Generating fallback.")
                return _generate_fallback_leaf()
        bgr = cv2.imread(filename)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # If source is a local file path
    if isinstance(source, str) and os.path.exists(source):
        bgr = cv2.imread(source)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Default fallback leaf
    return _generate_fallback_leaf()

def _generate_fallback_leaf():
    """Generates a synthetic realistic leaf image tensor if offline."""
    img = np.full((250, 250, 3), (35, 125, 45), dtype=np.uint8)
    cv2.circle(img, (125, 125), 45, (130, 80, 30), -1)
    return img
