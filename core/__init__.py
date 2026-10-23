"""
CropDiseaseAI Core Package
Implements Proposed Architecture Parts 1 to 4:
- Part 1: Crop Leaf Image Acquisition
- Part 2: Image Preprocessing (OpenCV)
- Part 3: AI Disease Classification & Severity Estimation
- Part 4: Crop Health Assessment (Features + Environmental Data)
"""

from .part1_acquisition import acquire_leaf_image, BENCHMARK_SAMPLES
from .part2_preprocessing import preprocess_leaf_image
from .part3_classifier import CropDiseaseClassifier, classify_disease_and_severity, CLASSES
from .part4_assessment import assess_crop_health, AGRONOMIC_PRESCRIPTIONS

__all__ = [
    "acquire_leaf_image",
    "BENCHMARK_SAMPLES",
    "preprocess_leaf_image",
    "CropDiseaseClassifier",
    "classify_disease_and_severity",
    "CLASSES",
    "assess_crop_health",
    "AGRONOMIC_PRESCRIPTIONS",
]
