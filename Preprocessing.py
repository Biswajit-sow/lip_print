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


def detect_print_boundary(gray):
    """
    Finds the bounding box of the ink impression using adaptive thresholding
    and contour detection. Falls back to the full image if no clear region found.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        h, w = gray.shape
        return (0, 0, w, h)
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)  # x, y, w, h


def crop_to_print(gray, bbox, padding=10):
    x, y, w, h = bbox
    H, W = gray.shape
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(W, x + w + padding), min(H, y + h + padding)
    return gray[y0:y1, x0:x1]


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


def preprocess_pipeline(path):
    """Runs the full Stage 1 pipeline on a single image path."""
    img = load_image(path)
    gray = to_grayscale(img)
    gray = denoise(gray)
    enhanced = enhance_contrast(gray)
    bbox = detect_print_boundary(enhanced)
    cropped = crop_to_print(enhanced, bbox)
    quadrants = split_quadrants(cropped)
    return {
        "original": img,
        "enhanced": enhanced,
        "cropped": cropped,
        "quadrants": quadrants,
    }