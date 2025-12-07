import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from scipy.signal import convolve2d

import os
import cv2


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy = np.copy(image)
    total_pixels = image.size

    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255

    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0

    return noisy


def iterative_median_filter(image, iterations=5, ksize=3):
    filtered = image.copy()
    for i in range(iterations):
        filtered = cv2.medianBlur(filtered, ksize)
    return filtered


def keep_center_object_only(mask):
    h, w = mask.shape
    center_x, center_y = w // 2, h // 2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_center = np.zeros_like(mask)
    valid_contours = []

    for cnt in contours:
        dist = cv2.pointPolygonTest(cnt, (center_x, center_y), False)
        if dist >= 0:
            valid_contours.append(cnt)

    if not valid_contours:
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if x <= center_x <= x + cw and y <= center_y <= y + ch:
                valid_contours.append(cnt)

    if valid_contours:
        c_max = max(valid_contours, key=cv2.contourArea)
        cv2.drawContours(mask_center, [c_max], -1, 255, thickness=cv2.FILLED)
        return mask_center, cv2.contourArea(c_max)
    else:
        return np.zeros_like(mask), 0


def remove_hair_noise(image_gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(image_gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(image_gray, hair_mask, 1, cv2.INPAINT_TELEA)
    return result


# ================================
# MAIN SECTION
# ================================

lokasi_path = os.getcwd()
parent_path = os.path.dirname(lokasi_path)
path = (
    r"d:\2 Kuliah\S2 - Universitas Gadjah Mada\Semester 1\RPP\Tugas\tugas_proyek_akhir"
)

os.chdir(path)

folder_img = Path("./Dataset/ISBI2016_ISIC_Part3_Training_Data")
path_csv = Path("./Dataset/ISBI2016_ISIC_Part3_Training_GroundTruth.csv")
output_folder = Path("./Dataset/Hasil_Segmentasi")
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path_csv, header=None, names=["image_id", "label"])
print(f"[INFO] Memulai proses pada {len(df)} gambar...\n")


for index, row in df.iterrows():
    img_id = row["image_id"]
    label_val = row["label"]
    img_filename = f"{img_id}.jpg"
    img_path = folder_img / img_filename

    if not img_path.exists():
        print(f"[WARNING] Gambar tidak ditemukan: {img_path}")
        continue

    img_rgb = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Step pipeline
    img_clean = remove_hair_noise(img_gray)
    img_noisy = add_salt_and_pepper_noise(img_clean, salt_prob=0.02, pepper_prob=0.02)
    img_filtered = iterative_median_filter(img_noisy, iterations=5, ksize=3)

    _, mask_raw = cv2.threshold(
        img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = mask_raw.shape
    center_region = mask_raw[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10]
    mean_center = np.mean(center_region)

    inverted = False
    if mean_center < 127:
        mask_inverted = cv2.bitwise_not(mask_raw)
        inverted = True
    else:
        mask_inverted = mask_raw

    final_mask, contour_area = keep_center_object_only(mask_inverted)

    save_name = f"{img_id}_mask.png"
    save_path = output_folder / save_name

    cv2.imwrite(str(save_path), final_mask)

    print(f"[{index+1}/{len(df)}] DONE {img_filename}")
    print(f"    | Class                  : {label_val}")
    print(f"    | Otsu Threshold Applied : Yes")
    print(f"    | Mask Inverted          : {'Yes' if inverted else 'No'}")
    print(f"    | Largest Area Selected  : {contour_area:.2f} px")
    print(f"    | Output Saved To        : {save_path}\n")


print("\n[PROCESS COMPLETE] Semua gambar telah selesai diproses.")
