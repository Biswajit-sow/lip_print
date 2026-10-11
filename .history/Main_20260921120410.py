"""
main.py
-------
Command-line entry point for the Lip Print Comparator prototype.

Usage:
    python main.py <print1_image_path> <print2_image_path>

Runs both images through preprocessing (Stage 1), then compares them
quadrant-by-quadrant using groove feature matching (Stage 2 + Stage 4),
and prints an overall similarity score. Match visualizations (matched
points drawn between quadrant pairs) are saved to the output folder.
"""

import sys
import os
import cv2

from Preprocessing import preprocess_pipeline
from Feature_matching import compare_quadrant

QUADRANT_ORDER = ["UR", "UM", "UL", "LR", "LM", "LL"]


def compare_prints(path1, path2, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    result1 = preprocess_pipeline(path1)
    result2 = preprocess_pipeline(path2)

    quadrant_results = {}
    total_score = 0.0

    for q in QUADRANT_ORDER:
        img1 = result1["quadrants"][q]
        img2 = result2["quadrants"][q]

        if img1.size == 0 or img2.size == 0:
            quadrant_results[q] = {"score": 0.0}
            continue

        res = compare_quadrant(img1, img2)
        quadrant_results[q] = res
        total_score += res["score"]

        vis = cv2.drawMatches(
            res["enhanced_1"], res["kp1"],
            res["enhanced_2"], res["kp2"],
            res["matches"], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(os.path.join(output_dir, f"match_{q}.png"), vis)

    overall_score = round(total_score / len(QUADRANT_ORDER), 2)

    print("=" * 55)
    print(" LIP PRINT COMPARISON RESULT (PROTOTYPE)")
    print("=" * 55)
    for q in QUADRANT_ORDER:
        r = quadrant_results[q]
        print(
            f" Quadrant {q}: Similarity = {r.get('score', 0)}%  "
            f"(matches: {r.get('num_good_matches', 0)}, "
            f"pts: {r.get('num_keypoints_1', 0)}/{r.get('num_keypoints_2', 0)})"
        )
    print("-" * 55)
    print(f" OVERALL SIMILARITY SCORE: {overall_score}%")
    print("=" * 55)
    print(f"\nPer-quadrant match visualizations saved to: {output_dir}/")

    return overall_score, quadrant_results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <print1_image> <print2_image>")
        sys.exit(1)
    compare_prints(sys.argv[1], sys.argv[2])