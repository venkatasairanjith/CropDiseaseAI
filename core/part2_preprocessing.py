"""
Part 2: Image Preprocessing (OpenCV / Pillow)
- Resize to 224x224
- Bilateral filtering for edge-preserving lesion denoising
- CLAHE (Contrast Limited Adaptive Histogram Equalization) on Lightness channel
- Normalization pipeline for Deep Learning input
"""

import cv2
import numpy as np

def preprocess_leaf_image(img_rgb, target_size=(224, 224)):
    """
    Executes Part 2 OpenCV preprocessing pipeline:
    Returns:
    - resized_rgb: Direct resize (224x224)
    - enhanced_rgb: Denoised + CLAHE enhanced image for display
    - normalized_tensor: Shape (1, 3, 224, 224) normalized array for model inference
    """
    if img_rgb is None:
        raise ValueError("Input image cannot be None")

    # 1. Resize to target dimension
    resized_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    # 2. Noise Removal: Bilateral filter preserves sharp lesion borders while removing sensor noise
    denoised_rgb = cv2.bilateralFilter(resized_rgb, d=7, sigmaColor=60, sigmaSpace=60)

    # 3. Enhance Contrast & Quality: CLAHE on Lightness (L) channel in LAB color space
    lab = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # 4. Normalization (ImageNet Standard: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225])
    norm_img = enhanced_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (norm_img - mean) / std
    
    # Transpose to (C, H, W) and add batch dim -> (1, 3, 224, 224)
    tensor_input = np.transpose(norm_img, (2, 0, 1))
    tensor_input = np.expand_dims(tensor_input, axis=0)

    return resized_rgb, enhanced_rgb, tensor_input
