import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def exclude_outliers(data_folder, ref_variant, method, outlier_multiplicator=3):
    arr = np.loadtxt(f"{data_folder}/python_{ref_variant}_m3c2_distances_coordinates.txt", skiprows=1)
    mask_valid = ~np.isnan(arr[:, 3])
    arr_valid = arr[mask_valid]
    distances_valid = arr_valid[:, 3]

    if method == "rmse":
        rmse = np.sqrt(np.mean(distances_valid ** 2))
        outlier_mask = np.abs(distances_valid) > (outlier_multiplicator * rmse)
        logger.info(f"[Exclude Outliers] RMS: {rmse:.6f}")
        logger.info(f"[Exclude Outliers] Outlier-Schwelle: {outlier_multiplicator} * RMSE = {outlier_multiplicator * rmse:.6f}")
    elif method == "iqr":
        q1 = np.percentile(distances_valid, 25)
        q3 = np.percentile(distances_valid, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (distances_valid < lower_bound) | (distances_valid > upper_bound)
        logger.info(f"[Exclude Outliers] IQR: {iqr:.6f}")
        logger.info(f"[Exclude Outliers] Outlier-Schwellen: {lower_bound:.6f} bis {upper_bound:.6f}")
    elif method == "std":
        mu = np.mean(distances_valid)
        std = np.std(distances_valid)
        outlier_mask = np.abs(distances_valid - mu) > (outlier_multiplicator * std)
        logger.info(f"[Exclude Outliers] STD: {std:.6f}")
        logger.info(f"[Exclude Outliers] Outlier-Schwelle: {outlier_multiplicator} * STD = {outlier_multiplicator * std:.6f}")
    elif method == "nmad":
        med  = np.median(distances_valid)
        nmad = 1.4826 * np.median(np.abs(distances_valid - med))
        outlier_mask = np.abs(distances_valid - med) > (outlier_multiplicator * nmad)
    else:
        raise ValueError("Unbekannte Methode für Ausreißer-Erkennung: 'rms' oder 'iqr'")

    arr_excl_outlier = arr_valid[~outlier_mask]
    out_path_inlier = os.path.join(data_folder, f"python_{ref_variant}_m3c2_distances_coordinates_inlier_{method}.txt")
    header = "x y z distance"
    np.savetxt(out_path_inlier, arr_excl_outlier, fmt="%.6f", header=header)

    out_path_outlier = os.path.join(data_folder, f"python_{ref_variant}_m3c2_distances_coordinates_outlier_{method}.txt")
    np.savetxt(out_path_outlier, arr_valid[outlier_mask], fmt="%.6f", header=header)

    logger.info(f"[Exclude Outliers] Gesamt: {arr.shape[0]}")
    logger.info(f"[Exclude Outliers] NaN: {(np.isnan(arr[:, 3])).sum()}")
    logger.info(f"[Exclude Outliers] Valid (ohne NaN): {arr_valid.shape[0]}")
    logger.info(f"[Exclude Outliers] Methode: {method}")
    logger.info(f"[Exclude Outliers] Outlier: {arr_valid[outlier_mask].shape[0]}")
    logger.info(f"[Exclude Outliers] Inlier: {arr_excl_outlier.shape[0]}")
    logger.info(f"[Exclude Outliers] Inlier (ohne Outlier) gespeichert: {out_path_inlier}")
    logger.info(f"[Exclude Outliers] Outlier gespeichert: {out_path_outlier}")
