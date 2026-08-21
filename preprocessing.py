import cv2
import numpy as np


def letterbox_resize(image, target_size=224, pad_color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def crop_to_roi(image, thresh_method="otsu", margin=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if thresh_method == "otsu":
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, image.shape[1])
    y1 = min(y + h + margin, image.shape[0])

    return image[y0:y1, x0:x1]


def gray_world_white_balance(image):
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    b = b * (mean_gray / (mean_b + 1e-6))
    g = g * (mean_gray / (mean_g + 1e-6))
    r = r * (mean_gray / (mean_r + 1e-6))

    balanced = cv2.merge([b, g, r])
    return np.clip(balanced, 0, 255).astype(np.uint8)


def match_histogram(image, reference):
    matched = np.zeros_like(image)
    for ch in range(3):
        src_hist, _ = np.histogram(image[:, :, ch].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference[:, :, ch].flatten(), 256, [0, 256])

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        lookup = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_idx in range(256):
            while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
                ref_idx += 1
            lookup[src_idx] = ref_idx

        matched[:, :, ch] = cv2.LUT(image[:, :, ch], lookup)
    return matched


def denoise(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def preprocess_image_array(image_bgr, reference_img=None, use_clahe=False, target_size=224):
    """Same as training's preprocess_image(), but takes an already-decoded
    BGR array instead of a file path (so it works with Streamlit uploads)."""
    image = crop_to_roi(image_bgr)
    image = gray_world_white_balance(image)
    if reference_img is not None:
        image = match_histogram(image, reference_img)
    image = denoise(image)
    if use_clahe:
        image = apply_clahe(image)
    image = letterbox_resize(image, target_size)
    return image