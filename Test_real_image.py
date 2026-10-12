"""
test_real_images.py
--------------------
Prototype test script for Image 2, Image 3, and Image 4 ONLY, following
the Suzuki-Tsuchihashi six-quadrant methodology (Das et al., JFMT 2026,
Figure 2): Upper Right, Upper Middle, Upper Left, Lower Right, Lower
Middle, Lower Left.

Image 1 is intentionally excluded: it is a live photo of lips on a face,
not an ink impression on paper. This pipeline's preprocessing (paper
boundary detection) does not apply to that capture method.

IMPORTANT CALIBRATION NOTE:
The qualitative labels below (Very Low / Low / Moderate / High / Very
High) use provisional thresholds for readability only. They are NOT
scientifically calibrated forensic cutoffs -- that requires testing
against at least one confirmed genuine (same-person) pair, which this
prototype does not yet have. Treat all output here as exploratory
pipeline validation, not a forensic conclusion.

HOW TO USE:
1. Edit the IMAGES dictionary below so each "path" points to your actual
   file location.
2. Make sure "Image2" points to the clean, plain-background version.
3. Set "mode" to "auto" for clean backgrounds, or "manual" for cluttered
   ones (opens a window to draw the crop box by hand, then press ENTER).
4. Run:
       python test_real_images.py
"""

import os
import itertools
import cv2

from preprocessing import preprocess_pipeline
from feature_matching import compare_quadrant

# Order matches Figure 2 of the reference study: top row then bottom row,
# left-to-right as the print appears once correctly oriented.
QUADRANT_ORDER = ["UR", "UM", "UL", "LR", "LM", "LL"]
QUADRANT_FULL_NAME = {
    "UR": "Upper Right", "UM": "Upper Middle", "UL": "Upper Left",
    "LR": "Lower Right", "LM": "Lower Middle", "LL": "Lower Left",
}

# ---------------------------------------------------------------------
# EDIT THESE PATHS to match your actual files on your machine.
# Image 1 is deliberately NOT listed here (see note above).
# ---------------------------------------------------------------------
IMAGES = {
    "Image2": {"path": r"F:\lip_print\lip_print\Lip_print_image2.jpg", "mode": "auto"},
    "Image3": {"path": r"F:\lip_print\lip_print\Lip_print_image3.jpg", "mode": "auto"},
    "Image4": {"path": r"F:\lip_print\lip_print\Lip_print_image4.jpg", "mode": "auto"},
}
# ---------------------------------------------------------------------


def interpret_score(score):
    """
    Provisional qualitative band for a similarity score (0-100).
    NOT a calibrated forensic threshold -- see module docstring.
    """
    if score < 5:
        return "Very Low"
    elif score < 15:
        return "Low"
    elif score < 35:
        return "Moderate"
    elif score < 60:
        return "High"
    else:
        return "Very High"


def manual_roi_select(image_path):
    """Opens a window so you can draw a crop box by hand (cluttered backgrounds)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    max_dim = 900
    scale = min(1.0, max_dim / max(h, w))
    display = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img.copy()

    window_name = "Draw a box around the lip print, then press ENTER"
    print(f"\n>>> Draw a box around the lip print in '{os.path.basename(image_path)}', then press ENTER.")
    roi = cv2.selectROI(window_name, display, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w_box, h_box = roi
    if w_box == 0 or h_box == 0:
        print("No region selected -- falling back to automatic detection.")
        return None
    if scale < 1.0:
        x, y, w_box, h_box = [int(v / scale) for v in (x, y, w_box, h_box)]
    return (x, y, w_box, h_box)


def load_print(image_path, mode):
    manual_bbox = manual_roi_select(image_path) if mode == "manual" else None
    return preprocess_pipeline(image_path, manual_bbox=manual_bbox)


def compare_two_prints(result_a, result_b, label_a, label_b, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    total_score = 0.0
    print(f"\n--- {label_a}  vs  {label_b} ---")

    for q in QUADRANT_ORDER:
        img_a = result_a["quadrants"][q]
        img_b = result_b["quadrants"][q]
        full_name = QUADRANT_FULL_NAME[q]

        if img_a.size == 0 or img_b.size == 0:
            print(f"  {q} ({full_name}): skipped (empty crop)")
            continue

        res = compare_quadrant(img_a, img_b)
        total_score += res["score"]
        band = interpret_score(res["score"])
        print(
            f"  {q} ({full_name}): {res['score']}%  [{band}]  "
            f"(matches: {res['num_good_matches']}, "
            f"pts: {res['num_keypoints_1']}/{res['num_keypoints_2']})"
        )

        vis = cv2.drawMatches(
            res["enhanced_1"], res["kp1"],
            res["enhanced_2"], res["kp2"],
            res["matches"], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        safe_name = f"{label_a}_vs_{label_b}_{q}.png".replace(" ", "_")
        cv2.imwrite(os.path.join(output_dir, safe_name), vis)

    overall = round(total_score / len(QUADRANT_ORDER), 2)
    overall_band = interpret_score(overall)
    print(f"  OVERALL: {overall}%  [{overall_band}]")
    return overall


def main():
    print("=" * 65)
    print(" LIP PRINT COMPARATOR -- PROTOTYPE TEST (Image 2, 3, 4)")
    print(" Six-quadrant method per Das et al., JFMT 2026 (Suzuki-Tsuchihashi)")
    print(" Image 1 excluded: live face photo, not an ink print.")
    print("=" * 65)

    results = {}
    for label, cfg in IMAGES.items():
        path, mode = cfg["path"], cfg["mode"]
        if not os.path.isfile(path):
            print(f"\n[ERROR] File not found for {label}: {path}")
            print("Edit the IMAGES dictionary at the top of this script with the correct path.")
            return
        print(f"\nProcessing {label} ({mode}) -> {path}")
        results[label] = load_print(path, mode)

    output_dir = "manual_test_output"
    labels = list(results.keys())
    summary = []
    for label_a, label_b in itertools.combinations(labels, 2):
        score = compare_two_prints(results[label_a], results[label_b], label_a, label_b, output_dir)
        summary.append((label_a, label_b, score))

    print("\n" + "=" * 65)
    print(" SUMMARY -- ALL PAIRWISE COMPARISONS")
    print("=" * 65)
    for a, b, s in sorted(summary, key=lambda x: -x[2]):
        print(f"  {a:10s} vs {b:10s} : {s}%  [{interpret_score(s)}]")
    print("=" * 65)
    print(f"\nMatch visualizations saved to: {output_dir}/")
    print("\nIMPORTANT: The [band] labels above use provisional thresholds for")
    print("readability only -- they are NOT validated forensic cutoffs. This")
    print("requires testing against a confirmed genuine (same-person) pair")
    print("before any result here can be treated as a forensic conclusion.")


if __name__ == "__main__":
    main()