import numpy as np
import matplotlib.pyplot as plt
import os


base = "data/0342-0349/"

variants = [
    ("ref", "python_ref_m3c2_distances.txt"),
    ("ref_ai", "python_ref_ai_m3c2_distances.txt"),
]
inlier_suffixes = [
    ("std", "Inlier _STD"),
    ("rmse", "Inlier _RMSE"),
    ("nmad", "Inlier _NMAD"),
    ("iqr", "Inlier _IQR"),
]

for variant, dist_file in variants:
    data_list = []
    labels = []
    # All Distances
    file_path = base + dist_file
    try:
        data = np.loadtxt(file_path)
        if data.ndim == 0 or data.size == 0:
            data = np.array([])
        else:
            data = data[~np.isnan(data)]
        if data.size > 0:
            data_list.append(data)
            labels.append("All Distances")
    except Exception:
        pass

    # Inlier Varianten
    for suffix, label in inlier_suffixes:
        file_path = f"{base}python_{variant}_m3c2_distances_coordinates_inlier_{suffix}.txt"
        try:
            arr = np.loadtxt(file_path, skiprows=1)
            data = arr[:, -1]
            data = data[~np.isnan(data)]
            if data.size > 0:
                data_list.append(data)
                labels.append(label)
        except Exception:
            pass

    plt.figure(figsize=(8, 6))
    plt.boxplot(data_list, labels=labels)
    plt.ylabel("Distanz")
    plt.title(f"Vergleich der Distanzverteilungen (0342-0349) {variant}")
    plt.grid(True)
    plt.tight_layout()

    outdir = os.path.join("outputs", "MARS_output", "Plots_MARS_Outlier")
    os.makedirs(outdir, exist_ok=True)
    basename = base.strip("/").split("/")[-1]  # ergibt "0342-0349"
    plt.savefig(os.path.join(outdir, f"{basename}_OutlierComparison_{variant}.png"))
    plt.close()