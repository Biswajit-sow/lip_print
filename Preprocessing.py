"""
preprocessing.py
----------------
Stage 1 of the Lip Print Comparator pipeline.

Takes a raw lip print image (scanned/photographed ink impression on paper)
and produces a cleaned, standardized image split into the six forensic
quadrants used in the Suzuki-Tsuchihashi classification system:
    UR, UM, UL (upper lip)  |  LR, LM, LL (lower lip)

Note on design choice:
A lip PRINT (ink impression on paper) is not a face photo, so face/landmark
detectors (e.g. MediaPipe FaceMesh) do not apply here. Instead, we detect the
print's own boundary using contour/threshold analysis (the darkest, most
textured region of the page), then crop and split it geometrically.
"""

import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise(gray):
    return cv2.fastNlMeansDenoising(gray, h=10)


def enhance_contrast(gray):
    """CLAHE: brings out faint groove lines without blowing out bright areas."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_print_boundary(bgr_img):
    """
    Finds the bounding box of the ink impression using the SATURATION
    channel (ink is distinctly more colorful than white paper), which is
    far more reliable than brightness-based thresholding for lipstick
    impressions on paper.

    NOTE: Works well for clean, plain backgrounds (paper filling most of
    the frame). For images with cluttered/reflective backgrounds (e.g. a
    glass surface, mirror, or shadows visible around the paper), this can
    mis-detect the boundary — use `manual_bbox` in preprocess_pipeline()
    or the --manual flag in main.py as a fallback in those cases.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    S = cv2.GaussianBlur(hsv[:, :, 1], (9, 9), 0)
    _, mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = bgr_img.shape[:2]
        return (0, 0, w, h)
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)  # x, y, w, h


def crop_to_print(gray, bbox, padding=10):
    x, y, w, h = bbox
    H, W = gray.shape
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(W, x + w + padding), min(H, y + h + padding)
    return gray[y0:y1, x0:x1]


def auto_orient(cropped):
    """
    Many real-world print photographs are captured with the print running
    vertically (tall oval, lip-closure line running top-to-bottom) rather
    than the standard horizontal mouth orientation (upper lip on top,
    lower lip on bottom) used in forensic literature and assumed by
    split_quadrants(). If the cropped print is taller than it is wide,
    rotate it 90 degrees so quadrant splitting lines up correctly.
    """
    h, w = cropped.shape[:2]
    if h > w:
        return cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    return cropped


def split_quadrants(cropped):
    """
    Splits the cropped print into six regions matching forensic convention:
    top half = upper lip (UR/UM/UL), bottom half = lower lip (LR/LM/LL),
    each divided into left/middle/right thirds.
    """
    h, w = cropped.shape
    mid_h = h // 2
    third_w = w // 3

    upper, lower = cropped[0:mid_h, :], cropped[mid_h:h, :]

    return {
        "UR": upper[:, 0:third_w],
        "UM": upper[:, third_w:2 * third_w],
        "UL": upper[:, 2 * third_w:w],
        "LR": lower[:, 0:third_w],
        "LM": lower[:, third_w:2 * third_w],
        "LL": lower[:, 2 * third_w:w],
    }


def preprocess_pipeline(path, manual_bbox=None):
    """
    Runs the full Stage 1 pipeline on a single image path.

    manual_bbox: optional (x, y, w, h) tuple. If provided, skips automatic
    boundary detection entirely and uses this region instead — use this
    for images with cluttered/reflective backgrounds where auto-detection
    is unreliable (see detect_print_boundary docstring).
    """
    img = load_image(path)
    gray_full = to_grayscale(img)
    gray_full = denoise(gray_full)

    bbox = manual_bbox if manual_bbox is not None else detect_print_boundary(img)

    # Enhance contrast only within the crop (avoids background noise
    # affecting the CLAHE enhancement of the actual print region)
    cropped_raw = crop_to_print(gray_full, bbox)
    enhanced_crop = enhance_contrast(cropped_raw)
    oriented = auto_orient(enhanced_crop)
    quadrants = split_quadrants(oriented)

    return {
        "original": img,
        "cropped": oriented,
        "quadrants": quadrants,
        "bbox_used": bbox,
    }