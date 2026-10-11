"""
generate_samples.py
--------------------
Creates synthetic 'lip print' test images so you can verify the
comparator pipeline runs correctly on your machine BEFORE using
real lip print photos. These are NOT real lip prints — just
randomly generated groove-like patterns for a quick sanity check.

Usage:
    python generate_samples.py

Creates three files in the current folder:
    sampleA1.png  - synthetic identity "A", capture 1
    sampleA2.png  - synthetic identity "A", capture 2 (same pattern + noise)
    sampleB1.png  - synthetic identity "B" (different pattern)

Then test with:
    python main.py sampleA1.png sampleA2.png   -> should score HIGH
    python main.py sampleA1.png sampleB1.png   -> should score LOW
"""

import numpy as np
import cv2
import random


def make_fake_lipprint(seed, w=600, h=300):
    random.seed(seed)
    np.random.seed(seed)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.ellipse(img, (w // 2, h // 2), (w // 2 - 20, h // 2 - 20), 0, 0, 360, (200, 200, 200), -1)
    for _ in range(40):
        x0 = random.randint(20, w - 20)
        y0 = random.randint(20, h - 20)
        length = random.randint(20, 80)
        angle = random.uniform(0, np.pi)
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        thickness = random.choice([1, 1, 2])
        cv2.line(img, (x0, y0), (x1, y1), (60, 40, 40), thickness, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    imgA1 = make_fake_lipprint(seed=1)
    imgA2 = make_fake_lipprint(seed=1)
    noise = np.random.randint(-10, 10, imgA2.shape, dtype=np.int16)
    imgA2 = np.clip(imgA2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    imgB1 = make_fake_lipprint(seed=99)

    cv2.imwrite("sampleA1.png", imgA1)
    cv2.imwrite("sampleA2.png", imgA2)
    cv2.imwrite("sampleB1.png", imgB1)
    print("Created: sampleA1.png, sampleA2.png, sampleB1.png")
    print("\nNow run:")
    print("  python main.py sampleA1.png sampleA2.png   (expect HIGH score)")
    print("  python main.py sampleA1.png sampleB1.png   (expect LOW score)")