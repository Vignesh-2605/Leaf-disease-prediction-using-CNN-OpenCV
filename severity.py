# severity.py

import cv2
import numpy as np

def estimate_severity(img_path):
    """
    Improved Severity Estimation:
    - Extracts leaf region first
    - Calculates infection only within leaf area
    """

    # Load image
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"❌ Image not found: {img_path}")

    img = cv2.resize(img, (224, 224))

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 1: Leaf Mask (Green Region Detection)
    leaf_lower = np.array([25, 20, 20])
    leaf_upper = np.array([100, 255, 255])

    leaf_mask = cv2.inRange(hsv, leaf_lower, leaf_upper)

    # Step 2: Disease Mask (Brown/Yellow Infection Detection)
    disease_lower = np.array([5, 50, 50])
    disease_upper = np.array([30, 255, 255])

    disease_mask = cv2.inRange(hsv, disease_lower, disease_upper)

    # Step 3: Severity Calculation Only Inside Leaf Area
    leaf_pixels = np.sum(leaf_mask > 0)

    disease_pixels = np.sum(
        (disease_mask > 0) & (leaf_mask > 0)
    )

    if leaf_pixels == 0:
        severity = 0
    else:
        severity = (disease_pixels / leaf_pixels) * 100                                                                                                                           *10

    return severity, disease_mask