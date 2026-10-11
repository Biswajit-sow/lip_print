"""
test_real_images.py
--------------------
Interactive testing script for comparing REAL lip print images.
Run this in VS Code's terminal (or any terminal with a display) once you
have actual print photos/scans to test.

Usage:
    python test_real_images.py

You will be prompted to:
1. Enter paths to 2 or more real lip print images (type each path, press
   ENTER on an empty line when done — minimum 2 images needed).
2. For each image, choose AUTO or MANUAL cropping:
     - AUTO: automatic boundary detection (works well for plain,
       uncluttered backgrounds).
     - MANUAL: a window pops up — draw a box around just the ink print
       with your mouse, then press ENTER/SPACE to confirm. Use this for
       images with reflections, shadows, or busy backgrounds where AUTO
       picks the wrong region.
3. The script runs every pairwise comparison and prints a summary table,
   plus saves visual match overlays for each quadrant pair.

Note: MANUAL mode opens a GUI window, so it needs to run on a machine
with a display (your local PC) — it will not work over SSH/headless.
"""

import os
import itertools
import cv2

from preprocessing import preprocess_pipeline
from feature_matching import compare_quadrant

QUADRANT_ORDER = ["UR", "UM", "UL", "LR", "LM", "LL"]


def manual_roi_select(image_path):
    """Opens an interactive window so the user can draw a crop box by hand."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Resize only for display/selection convenience; coordinates are
    # scaled back up to the original resolution afterward.
    h, w = img.shape[:2]
    max_dim = 900
    scale = min(1.0, max_dim / max(h, w))
    display = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img.copy()

    window_name = "Draw a box around the lip print, then press ENTER"
    print(f"\n>>> A window will open for '{os.path.basename(image_path)}'.")
    print(">>> Draw a rectangle around JUST the ink print, then press ENTER or SPACE.")
    print(">>> Press 'c' to cancel and use automatic detection instead.")
    roi = cv2.selectROI(window_name, display, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w_box, h_box = roi
    if w_box == 0 or h_box == 0:
        print("No region selected — falling back to automatic detection.")
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
        if img_a.size == 0 or img_b.size == 0:
            print(f"  Quadrant {q}: skipped (empty crop)")
            continue

        res = compare_quadrant(img_a, img_b)
        total_score += res["score"]
        print(
            f"  Quadrant {q}: {res['score']}%  "
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
    print(f"  OVERALL: {overall}%")
    return overall


def main():
    print("=" * 60)
    print(" LIP PRINT COMPARATOR — MANUAL TESTING (REAL IMAGES)")
    print("=" * 60)

    image_paths = []
    print("\nEnter image file paths one at a time (quotes are fine).")
    print("Press ENTER on an empty line when done — need at least 2 images.\n")
    while True:
        p = input(f"Image {len(image_paths) + 1} path (or ENTER to finish): ").strip().strip('"')
        if p == "":
            if len(image_paths) >= 2:
                break
            print("Need at least 2 images before finishing.")
            continue
        if not os.path.isfile(p):
            print(f"File not found: {p}")
            continue
        image_paths.append(p)

    results = {}
    for p in image_paths:
        label = os.path.splitext(os.path.basename(p))[0]
        mode_in = input(f"\nFor '{label}', use (a)uto or (m)anual crop? [a/m]: ").strip().lower()
        mode = "manual" if mode_in.startswith("m") else "auto"
        print(f"Processing {label} ({mode})...")
        results[label] = load_print(p, mode)

    output_dir = "manual_test_output"
    labels = list(results.keys())
    summary = []
    for label_a, label_b in itertools.combinations(labels, 2):
        score = compare_two_prints(results[label_a], results[label_b], label_a, label_b, output_dir)
        summary.append((label_a, label_b, score))

    print("\n" + "=" * 60)
    print(" SUMMARY — ALL PAIRWISE COMPARISONS")
    print("=" * 60)
    for a, b, s in sorted(summary, key=lambda x: -x[2]):
        print(f"  {a:20s} vs {b:20s} : {s}%")
    print("=" * 60)
    print(f"\nMatch visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()