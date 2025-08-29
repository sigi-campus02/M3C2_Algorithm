# visualization_service.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# seaborn optional
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# plyfile optional
try:
    from plyfile import PlyData, PlyElement  # type: ignore
except Exception:
    PlyData = None
    PlyElement = None


class VisualizationService:

    # --- ÄNDERN: txt_to_ply_with_distance_color(...) so anpassen, dass 'distance' als Scalar mitgeschrieben wird ---
    @staticmethod
    def txt_to_ply_with_distance_color(
        txt_path: str,
        outply: str,
        nan_color: Tuple[int, int, int] = (255, 255, 255),
        percentile_range: Tuple[float, float] = (0.0, 100.0),
        scalar_name: str = "distance",              # <— NEU: frei benennbar
        write_binary: bool = True,                  # <— optional
    ) -> None:
        """
        Lädt eine TXT-Datei mit x, y, z, distance und exportiert als farbige PLY.
        - NaN-Distanzen: nan_color
        - percentile_range: z.B. (1, 99) für robustes Clipping
        - Schreibt 'distance' zusätzlich als Scalar Field in die PLY.
        """
        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

        arr = np.loadtxt(txt_path, skiprows=1)
        if arr.size == 0:
            raise ValueError(f"TXT-Datei enthält keine Werte: {txt_path}")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 4:
            raise ValueError(f"TXT-Datei muss 4 Spalten haben: {txt_path}")

        points = arr[:, :3]
        distances = arr[:, 3]
        n = len(distances)

        # Farben berechnen (wie bisher)
        colors = np.zeros((n, 3), dtype=np.uint8)
        valid_mask = ~np.isnan(distances)
        if valid_mask.any():
            v = distances[valid_mask]
            p_lo, p_hi = percentile_range
            vmin = float(np.percentile(v, p_lo))
            vmax = float(np.percentile(v, p_hi))
            if vmax <= vmin:
                vmax = vmin + 1e-12
            normed = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)

            cc_colors = [(0.0, "blue"), (0.33, "green"), (0.66, "yellow"), (1.0, "red")]
            cc_cmap = LinearSegmentedColormap.from_list("CC_Colormap", cc_colors)
            colored_valid = (cc_cmap(normed)[:, :3] * 255).astype(np.uint8)
            colors[valid_mask] = colored_valid

        colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)

        # --- HIER: PLY mit zusätzlichem Float-Property 'distance' schreiben ---
        _write_ply_xyzrgb(
            points=points,
            colors=colors,
            outply=outply,
            scalar=distances.astype(np.float32),
            scalar_name=scalar_name,
            binary=write_binary,
        )

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[TXT->PLY] {txt_path} -> {outply} ({n} Punkte, SF='{scalar_name}')")



    # ---------- Diagramme ----------

    @staticmethod
    def histogram(
        distances: np.ndarray,
        path: Optional[str] = None,
        bins: int = 256,
        title: str = "Verteilung der M3C2-Distanzen",
    ) -> None:
        """Speichert (oder zeigt) ein Histogramm der gültigen Distanzen."""
        vals = distances[~np.isnan(distances)]
        plt.figure(figsize=(10, 6))
        if _HAS_SNS:
            sns.histplot(vals, bins=bins, kde=False)
        else:
            plt.hist(vals, bins=bins)
        plt.title(title)
        plt.xlabel("M3C2-Distanz")
        plt.ylabel("Anzahl Punkte")
        plt.grid(True)
        plt.tight_layout()
        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            plt.savefig(path)
        plt.close()

    # ---------- PLY-Exports ----------

    @staticmethod
    def colorize(
        points: np.ndarray,
        distances: np.ndarray,
        outply: str,
        nan_color: Tuple[int, int, int] = (255, 255, 255),
        percentile_range: Tuple[float, float] = (0.0, 100.0),
    ) -> np.ndarray:
        """
        Punktwolke anhand Distanz einfärben und als PLY speichern.
        - NaN-Distanzen: nan_color
        - percentile_range: z.B. (1, 99) für robustes Clipping
        Rückgabe: colors (uint8, Nx3)
        """
        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

        n = len(distances)
        colors = np.zeros((n, 3), dtype=np.uint8)

        valid_mask = ~np.isnan(distances)
        if valid_mask.any():
            v = distances[valid_mask]
            p_lo, p_hi = percentile_range
            vmin = float(np.percentile(v, p_lo))
            vmax = float(np.percentile(v, p_hi))
            if vmax <= vmin:
                vmax = vmin + 1e-12
            normed = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)

            # CC-ähnliche Farbskala: blau → grün → gelb → rot
            cc_colors = [(0.0, "blue"), (0.33, "green"), (0.66, "yellow"), (1.0, "red")]
            cc_cmap = LinearSegmentedColormap.from_list("CC_Colormap", cc_colors)

            colored_valid = (cc_cmap(normed)[:, :3] * 255).astype(np.uint8)
            colors[valid_mask] = colored_valid

        # NaNs als weiß (oder nan_color)
        colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)

        # -> PLY schreiben
        vertex = np.array(
            [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        el = PlyElement.describe(vertex, "vertex")

        d = os.path.dirname(outply)
        if d:
            os.makedirs(d, exist_ok=True)   
        PlyData([el], text=False).write(outply)
        return colors

    # --- OPTIONAL: export_valid(...) gleich mit Scalar schreiben (nur wenn du willst) ---
    @staticmethod
    def export_valid(
        points: np.ndarray,
        colors: np.ndarray,
        distances: np.ndarray,
        outply: str,
        scalar_name: str = "distance",     # <— NEU
        write_binary: bool = True,         # <— NEU
    ) -> None:
        """Nur gültige Punkte (non-NaN) mit Farben als PLY exportieren – inkl. 'distance' als Scalar Field."""
        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

        mask = ~np.isnan(distances)
        pts = points[mask]
        cols = colors[mask]
        dists = distances[mask].astype(np.float32)

        _write_ply_xyzrgb(
            points=pts,
            colors=cols,
            outply=outply,
            scalar=dists,
            scalar_name=scalar_name,
            binary=write_binary,
        )



def _write_ply_xyzrgb(
    points: np.ndarray,
    colors: np.ndarray,
    outply: str,
    scalar: Optional[np.ndarray] = None,
    scalar_name: str = "distance",
    binary: bool = True,
) -> None:
    """
    Schreibt eine PLY mit x,y,z + r,g,b und optional einem zusätzlichen
    Float-Property als Scalar Field (z.B. 'distance').

    CloudCompare interpretiert zusätzliche Float-Vertex-Properties als Scalar Fields.
    """
    if PlyData is None or PlyElement is None:
        raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

    n = points.shape[0]
    if colors.shape[0] != n:
        raise ValueError("Anzahl colors != Anzahl Punkte")

    base_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    if scalar is not None:
        if scalar.shape[0] != n:
            raise ValueError("Anzahl scalar != Anzahl Punkte")
        base_dtype.append((scalar_name, "f4"))

    if scalar is None:
        vertex = np.array(
            [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors)],
            dtype=base_dtype,
        )
    else:
        vertex = np.array(
            [(x, y, z, r, g, b, s) for (x, y, z), (r, g, b), s in zip(points, colors, scalar)],
            dtype=base_dtype,
        )

    el = PlyElement.describe(vertex, "vertex")
    d = os.path.dirname(outply)
    if d:
        os.makedirs(d, exist_ok=True)
    # binary=True => kleinere Dateien; CloudCompare kann beides
    PlyData([el], text=not binary).write(outply)