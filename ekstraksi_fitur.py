import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Library Feature Extraction
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import entropy
from sklearn.cluster import KMeans

# --- 1. FUNGSI BANTUAN (LBP & HISTOGRAM) ---


def extract_lbp_features(image_gray, mask_bin):
    """
    Mengekstraksi fitur tekstur Local Binary Pattern (LBP).
    Menggunakan varian 'uniform' agar invarian terhadap rotasi dan grayscale.
    """
    # Parameter standar LBP
    P = 8  # Jumlah titik tetangga
    R = 1  # Radius lingkaran

    # Hitung LBP
    lbp_image = local_binary_pattern(image_gray, P, R, method="uniform")

    # Masking: Hanya ambil nilai LBP di dalam area lesi
    lbp_valid = lbp_image[mask_bin > 0]

    if len(lbp_valid) == 0:
        return {f"lbp_bin_{i}": 0 for i in range(10)}

    # Hitung Histogram LBP (Distribusi pola tekstur)
    # Untuk method='uniform' dengan P=8, akan ada P+2 = 10 kemungkinan nilai (bins)
    n_bins = int(lbp_valid.max() + 1)
    hist, _ = np.histogram(lbp_valid, density=True, bins=n_bins, range=(0, n_bins))

    # Normalize histogram
    hist = hist / (hist.sum() + 1e-7)

    features = {}
    for i, val in enumerate(hist):
        # Ambil 10 bin pertama (pola paling umum)
        if i < 10:
            features[f"lbp_bin_{i}"] = val

    # Tambahkan statistik LBP (Energy & Entropy dari tekstur LBP)
    features["lbp_energy"] = np.sum(hist**2)
    features["lbp_entropy"] = entropy(hist + 1e-7)

    return features


def extract_color_histogram(image_hsv, mask_bin, bins=8):
    """
    Mengekstraksi Histogram Warna (HSV) untuk menangkap distribusi warna.
    Melanoma cenderung memiliki histogram yang lebih lebar/acak dibanding benign.
    """
    features = {}

    # Hitung histogram untuk setiap channel (Hue, Saturation, Value)
    # Kita gunakan bins=8 agar jumlah fitur tidak meledak
    for i, channel_name in enumerate(["H", "S", "V"]):
        # Hitung hist hanya di area mask (mask_bin)
        hist = cv2.calcHist([image_hsv], [i], mask_bin, [bins], [0, 256])

        # Normalize agar tidak terpengaruh ukuran gambar
        hist = cv2.normalize(hist, None).flatten()

        # Simpan nilai bin sebagai fitur
        for b in range(bins):
            features[f"hist_{channel_name}_bin_{b}"] = hist[b]

    return features


# --- FUNGSI FRACTAL & GLCM (DARI KODE SEBELUMNYA) ---
# (Saya ringkas agar tidak kepanjangan, isinya SAMA PERSIS dengan sebelumnya)


def fractal_dimension(image_binary):
    Z = image_binary > 0
    p = min(Z.shape)
    if p == 0:
        return 0
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        s2 = size
        h_lim = Z.shape[0] // s2 * s2
        w_lim = Z.shape[1] // s2 * s2
        Z_trim = Z[:h_lim, :w_lim]
        blocks = (
            Z_trim.reshape(Z_trim.shape[0] // s2, s2, Z_trim.shape[1] // s2, s2).sum(
                axis=(1, 3)
            )
            > 0
        )
        counts.append(np.sum(blocks))
    if len(counts) < 2:
        return 0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def extract_glcm_features(image_gray, mask_bin):
    x, y, w, h = cv2.boundingRect(mask_bin)
    roi_gray = image_gray[y : y + h, x : x + w]
    roi_mask = mask_bin[y : y + h, x : x + w]
    roi_gray_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)
    glcm = graycomatrix(
        roi_gray_masked,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = {}
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    for prop in props:
        features[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()
    return features


# --- 2. MASTER FUNCTION EKSTRAKSI ---


def extract_all_features(img_id, img_rgb, img_mask, label):
    """
    Master function: ABCD + GLCM + LBP + Color Hist
    """
    features = {"image_id": img_id, "label": label}

    # Preprocessing Mask
    _, mask_bin = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return features

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # --- A. ASYMMETRY ---
    roi = mask_bin[y : y + h, x : x + w]
    roi_resized = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)

    flip_h = cv2.flip(roi_resized, 1)
    flip_v = cv2.flip(roi_resized, 0)
    features["asym_total"] = (
        np.sum(cv2.absdiff(roi_resized, flip_h))
        + np.sum(cv2.absdiff(roi_resized, flip_v))
    ) / (2 * cv2.countNonZero(roi_resized) + 1e-5)

    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        features[f"hu_{i}"] = (
            -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i])) if hu[i] != 0 else 0
        )

    hog_feats, _ = hog(
        roi_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None,
    )
    features["hog_mean"] = np.mean(hog_feats)
    features["hog_var"] = np.var(hog_feats)

    # --- B. BORDER ---
    hull = cv2.convexHull(cnt)
    area, perimeter = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
    area_hull, perimeter_hull = cv2.contourArea(hull), cv2.arcLength(hull, True)

    features["solidity"] = area / area_hull if area_hull > 0 else 0
    features["convexity"] = perimeter_hull / perimeter if perimeter > 0 else 0

    edge_canvas = np.zeros_like(mask_bin)
    cv2.drawContours(edge_canvas, [cnt], -1, 255, 1)
    features["fractal_dim"] = fractal_dimension(edge_canvas)

    # --- C. COLOR (Basic + Advanced) ---
    pixels_rgb = img_rgb[mask_bin > 0]
    if len(pixels_rgb) > 0:
        pixels_input = pixels_rgb.reshape(-1, 1, 3)
        pixels_hsv = cv2.cvtColor(pixels_input, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        pixels_lab = cv2.cvtColor(pixels_input, cv2.COLOR_RGB2LAB).reshape(-1, 3)

        # Std Devs
        features["std_R"], features["std_G"], features["std_B"] = np.std(
            pixels_rgb, axis=0
        )
        features["std_H"], features["std_S"], features["std_V"] = np.std(
            pixels_hsv, axis=0
        )

        # KMeans Spread
        sample_pixels = pixels_rgb[::5] if len(pixels_rgb) > 2000 else pixels_rgb
        try:
            kmeans = KMeans(n_clusters=3, n_init=3, random_state=42).fit(sample_pixels)
            c = kmeans.cluster_centers_
            features["color_spread"] = (
                np.linalg.norm(c[0] - c[1])
                + np.linalg.norm(c[0] - c[2])
                + np.linalg.norm(c[1] - c[2])
            ) / 3
        except:
            features["color_spread"] = 0

    # --- D. DIAMETER ---
    features["area"] = area
    features["perimeter"] = perimeter
    features["max_diameter"] = cv2.minEnclosingCircle(cnt)[1] * 2

    # --- E. TEXTURE (GLCM + LBP) - POWERFUL FOR RECALL ---
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. GLCM
    features.update(extract_glcm_features(img_gray, mask_bin))

    # 2. LBP
    features.update(extract_lbp_features(img_gray, mask_bin))

    # --- F. COLOR HISTOGRAM --
    img_hsv_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    features.update(extract_color_histogram(img_hsv_full, mask_bin, bins=8))

    return features


# --- 3. SETUP & MAIN LOOP ---

lokasi_path = os.getcwd()
target_path = (
    r"d:\2 Kuliah\S2 - Universitas Gadjah Mada\Semester 1\RPP\Tugas\tugas_proyek_akhir"
)

try:
    if lokasi_path != target_path:
        os.chdir(target_path)
        print(f"Direktori kerja: {os.getcwd()}")
except:
    pass

folder_img = Path("./Dataset/ISBI2016_ISIC_Part3_Training_Data")
folder_mask = Path("./Dataset/Hasil_Segmentasi")
path_csv = Path("./Dataset/ISBI2016_ISIC_Part3_Training_GroundTruth.csv")

try:
    df = pd.read_csv(path_csv, header=None, names=["image_id", "label"])
    print(f"[INFO] Total data: {len(df)}")
except Exception as e:
    print(f"[ERROR] CSV Error: {e}")
    exit()

all_data = []

print("Mulai ekstraksi fitur ABCD...")

for index, row in df.iterrows():
    img_id = row["image_id"]
    label_val = row["label"]

    path_orig = folder_img / f"{img_id}.jpg"
    path_mask = folder_mask / f"{img_id}_mask.png"

    if path_orig.exists() and path_mask.exists():
        # Load Images
        # PENTING: OpenCV baca BGR, kita konversi ke RGB agar sesuai logika Color
        img_bgr = cv2.imread(str(path_orig))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_mask = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)

        try:
            feats = extract_all_features(img_id, img_rgb, img_mask, label_val)
            all_data.append(feats)

            if (index + 1) % 50 == 0:
                print(f"[{index+1}/{len(df)}] {img_id} -> Selesai")

        except Exception as e:
            print(f"[ERROR] {img_id}: {e}")
    else:
        # print(f"[SKIP] File tidak lengkap: {img_id}")
        pass

# --- 4. SIMPAN HASIL ---
if len(all_data) > 0:
    df_features = pd.DataFrame(all_data)
    # Isi NaN dengan 0 jika ada perhitungan error
    df_features = df_features.fillna(0)

    output_file = "features_ABCD_final.csv"
    df_features.to_csv(output_file, index=False)

    print("\n" + "=" * 50)
    print(f"[SELESAI] Ekstraksi Sukses!")
    print(f"Jumlah Fitur per Gambar: {df_features.shape[1]}")
    print(f"File disimpan di: {output_file}")
    print("=" * 50)
    print(df_features.head())
else:
    print("[WARNING] Tidak ada data yang tersimpan.")
