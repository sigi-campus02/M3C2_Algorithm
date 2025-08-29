from __future__ import annotations
import os
import re
import sys
import glob
import logging
import numpy as np
from typing import Iterable, Optional

logger = logging.getLogger("make_plys")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------
# Optional: deine vorhandene VisualizationService verwenden
# ------------------------------------------------------------
_USE_VIS_SERVICE = False
try:
    from services.visualization_service import VisualizationService  # type: ignore
    if hasattr(VisualizationService, "txt_to_ply_with_distance_color"):
        _USE_VIS_SERVICE = True
        logger.info("VisualizationService gefunden – verwende txt_to_ply_with_distance_color().")
except Exception as e:
    logger.info("VisualizationService nicht verfügbar (%s) – verwende Stand-alone Writer.", e)

# ------------------------------------------------------------
# Fallback: Stand-alone PLY Writer mit Distanz-Färbung
# - Colormap: einfache symmetrische Linearkarte um 0 (blau->weiß->rot)
# - Clipping: optional via Prozentilen, damit Ausreißer nicht alles dominieren
# ------------------------------------------------------------
def _colormap_blue_white_red(values: np.ndarray) -> np.ndarray:
    """Gibt uint8 RGB zurück. Erwartet Input in [-1, 1] (schon normalisiert & geclippt)."""
    v = values.clip(-1, 1)
    # Map: -1 -> blau(0,0,255), 0 -> weiß(255,255,255), +1 -> rot(255,0,0)
    r = np.where(v >= 0, 255, (1 + v) * 255)         # v in [-1,0] -> r: 0..255, v in [0,1] -> 255
    g = np.where(v >= 0, (1 - v) * 255, (1 + v) * 255)
    b = np.where(v >= 0, (1 - v) * 255, 255)         # v in [-1,0] -> 255..255? nein, (1+v)*? besser: konstant 255 nach blau
    # kleine Korrektur: weiß in der Mitte
    r = r.clip(0, 255).astype(np.uint8)
    g = g.clip(0, 255).astype(np.uint8)
    b = b.clip(0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1)

def _write_ply_ascii_xyzrgb(xyz: np.ndarray, rgb: np.ndarray, outpath: str) -> None:
    assert xyz.shape[0] == rgb.shape[0]
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def _txt_to_ply_standalone(txt_path: str, outply: str, clip_percentile: float = 98.0) -> None:
    """
    Liest TXT (x y z distance), färbt nach Distanz und schreibt ASCII .ply.
    'clip_percentile' steuert symmetrisches Clipping (z.B. 98 => +/- P98 vom Absolutwert).
    """
    data = np.loadtxt(txt_path, comments="#")
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Unerwartetes Format in {txt_path} – erwarte Spalten: x y z distance")

    xyz = data[:, :3]
    d = data[:, 3]

    # Symmetrisches Clipping um 0 (robust gg. Ausreißer)
    lim = np.percentile(np.abs(d[~np.isnan(d)]), clip_percentile) if np.isfinite(d).any() else 1.0
    lim = float(lim) if lim > 0 else 1.0
    d_norm = (d / lim).astype(np.float64)
    # NaNs -> 0 (weiß)
    d_norm[~np.isfinite(d_norm)] = 0.0

    rgb = _colormap_blue_white_red(d_norm)
    os.makedirs(os.path.dirname(outply), exist_ok=True)
    _write_ply_ascii_xyzrgb(xyz, rgb, outply)
    logger.info("PLY geschrieben: %s (n=%d, clip=±P%.0f=%.6g)", outply, xyz.shape[0], clip_percentile, lim)

# ------------------------------------------------------------
# Konvertierung-Logik
# ------------------------------------------------------------
PATTERNS = [
    "*_m3c2_distances_coordinates.txt",
    "*_m3c2_distances_coordinates_inlier_*.txt",
    "*_m3c2_distances_coordinates_outlier_*.txt",
]

def find_distance_txts(root: str) -> Iterable[str]:
    for pat in PATTERNS:
        yield from glob.iglob(os.path.join(root, "**", pat), recursive=True)

def outply_name(txt_path: str) -> str:
    stem, _ = os.path.splitext(txt_path)
    return stem + ".ply"

def convert_one(txt_path: str, overwrite: bool = False) -> Optional[str]:
    outply = outply_name(txt_path)
    if not overwrite and os.path.exists(outply):
        logger.info("Überspringe (bereits vorhanden): %s", outply)
        return outply

    if _USE_VIS_SERVICE:
        VisualizationService.txt_to_ply_with_distance_color(txt_path=txt_path, outply=outply)
        logger.info("PLY geschrieben (VisualizationService): %s", outply)
    else:
        _txt_to_ply_standalone(txt_path, outply)
    return outply

def convert_all(roots: list[str], overwrite: bool = False) -> None:
    total = 0
    for root in roots:
        for txt in find_distance_txts(root):
            try:
                convert_one(txt, overwrite=overwrite)
                total += 1
            except Exception as e:
                logger.warning("Fehler bei %s: %s", txt, e)
    logger.info("Fertig. %d PLY-Datei(en) erstellt/aktualisiert.", total)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    # Beispiele:
    #   python make_plys_from_distances.py data/Multi-Illumination
    #   python make_plys_from_distances.py data --overwrite
    import argparse
    ap = argparse.ArgumentParser(description="Erzeuge .ply aus *_m3c2_distances_coordinates*.txt")
    ap.add_argument("paths", nargs="+", help="Ordner oder Dateien (TXT). Bei Ordnern wird rekursiv gesucht.")
    ap.add_argument("--overwrite", action="store_true", help="Vorhandene .ply überschreiben")
    args = ap.parse_args()

    txts: list[str] = []
    dirs: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            dirs.append(os.path.abspath(p))
        elif os.path.isfile(p):
            if p.endswith(".txt"):
                txts.append(os.path.abspath(p))
            else:
                logger.warning("Ignoriere Datei (keine .txt): %s", p)
        else:
            logger.warning("Pfad nicht gefunden: %s", p)

    # Erst konkrete TXT-Dateien
    for t in txts:
        try:
            convert_one(t, overwrite=args.overwrite)
        except Exception as e:
            logger.warning("Fehler bei %s: %s", t, e)

    # Dann rekursiv in Ordnern nach Mustern suchen
    if dirs:
        convert_all(dirs, overwrite=args.overwrite)
