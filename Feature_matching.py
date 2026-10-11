"""
feature_matching.py
--------------------
Stage 2 + Stage 4 of the Lip Print Comparator pipeline.

1. Enhances groove/ridge structures using a Gabor filter bank (the same
   mathematical tool used to enhance fingerprint ridges).
2. Extracts feature points (ORB keypoints) from the groove-enhanced image.
   These act as a practical, working stand-in for true minutiae points
   (branch points, intersections, ridge endings) until a dedicated
   minutiae-detection model is trained on annotated data.
3. Matches feature points between two prints and computes a similarity score.

This is intentionally a classical computer-vision approach (no training
data required) so it can be validated immediately, before investing in a
trained deep-learning classifier (Suzuki-Tsuchihashi typing is a separate,
later stage — see project roadmap).
"""

import cv2
import numpy as np


def enhance_grooves(gray_img):
    """Applies an 8-orientation Gabor filter bank to highlight groove lines."""
    ksize = 9
    accum = np.zeros_like(gray_img, dtype=np.float32)
    for theta in np.arange(0, np.pi, np.pi / 8):
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5, psi=0
        )
        filtered = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
        accum = np.maximum(accum, filtered)
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
    return accum.astype(np.uint8)


def extract_keypoints(gray_img, max_features=500):
    """
    Extracts feature points using ORB (Oriented FAST + Rotated BRIEF).
    Proxy for minutiae points: robust to rotation/scale, no training needed.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio_thresh=0.75):
    """Brute-force Hamming matching with Lowe's ratio test for reliability."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    return good_matches


def similarity_score(kp1, desc1, kp2, desc2):
    """
    Similarity score (0-100) = reliable matches / smaller keypoint set size.
    This is a simple, explainable metric appropriate for a first prototype.
    """
    good_matches = match_descriptors(desc1, desc2)
    min_kp = min(len(kp1), len(kp2)) if kp1 is not None and kp2 is not None else 0
    if min_kp == 0:
        return 0.0, good_matches
    score = (len(good_matches) / min_kp) * 100
    return round(min(score, 100.0), 2), good_matches


def compare_quadrant(img1, img2):
    """Full per-quadrant comparison: enhance -> extract -> match -> score."""
    enh1, enh2 = enhance_grooves(img1), enhance_grooves(img2)
    kp1, desc1 = extract_keypoints(enh1)
    kp2, desc2 = extract_keypoints(enh2)
    score, matches = similarity_score(kp1, desc1, kp2, desc2)
    return {
        "score": score,
        "num_keypoints_1": len(kp1),
        "num_keypoints_2": len(kp2),
        "num_good_matches": len(matches),
        "kp1": kp1, "kp2": kp2, "matches": matches,
        "enhanced_1": enh1, "enhanced_2": enh2,
    }