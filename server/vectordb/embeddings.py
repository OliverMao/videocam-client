import cv2
import numpy as np

HSV_BINS = 64
GRAY_BINS = 64
EDGE_DIRS = 8
EDGE_GRID = (2, 2)
RESIZE_WIDTH = 320


def compute_embedding(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    small = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [HSV_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [HSV_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [HSV_BINS], [0, 256]).flatten()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [GRAY_BINS], [0, 256]).flatten()

    edges = cv2.Canny(gray, 50, 150)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx)
    magnitudes = np.sqrt(gx ** 2 + gy ** 2)
    mask = edges > 0
    angles_masked = angles[mask]
    mags_masked = magnitudes[mask]

    gh, gw = EDGE_GRID
    cell_h, cell_w = small.shape[0] // gh, small.shape[1] // gw
    edge_features: list[np.ndarray] = []
    for gi in range(gh):
        for gj in range(gw):
            y0, y1 = gi * cell_h, (gi + 1) * cell_h
            x0, x1 = gj * cell_w, (gj + 1) * cell_w
            cell_mask = mask[y0:y1, x0:x1]
            if cell_mask.sum() > 0:
                cell_angles = angles[y0:y1, x0:x1][cell_mask]
                cell_mags = magnitudes[y0:y1, x0:x1][cell_mask]
                hist, _ = np.histogram(cell_angles, bins=EDGE_DIRS, range=(-np.pi, np.pi), weights=cell_mags)
            else:
                hist = np.zeros(EDGE_DIRS)
            edge_features.append(hist)

    vec = np.concatenate([h_hist, s_hist, v_hist, gray_hist, *edge_features])
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)
