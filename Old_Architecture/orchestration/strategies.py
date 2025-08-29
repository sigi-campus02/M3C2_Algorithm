from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class ScaleScan:
    # 'scale' ist D (Normalskala, nicht der Nachbarschaftsradius)
    scale: float
    valid_normals: int
    mean_population: float
    roughness: float              # mean σ(D): StdAbw orthogonaler Residuen
    coverage: float
    mean_lambda3: float           # mittleres λ_min (Planaritätsmaß)
    # optionale Zusatzmetriken
    total_points: Optional[int] = None
    std_population: Optional[float] = None
    perc97_population: Optional[int] = None
    relative_roughness: Optional[float] = None
    total_voxels: Optional[int] = None

# ============================================================
# Radius-basierte Strategie
# ============================================================

def _fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Bestimmt die Best-Fit-Ebene einer 3D-Punktnachbarschaft per PCA und gibt
    Schwerpunkt, (einheitsnormierte) Ebenennormale, Eigenwerte der Kovarianz
    sowie die orthogonale Rauigkeit σ(D) zurück.

    Parameters
    ----------
    points : (N, 3) ndarray of float
        3D-Punktmenge der Nachbarschaft (z. B. Kugel mit Radius D/2 um den Core-Point). N >= 3.

    Returns
    -------
    centroid : (3,) ndarray
        Schwerpunkt der Punktnachbarschaft.
    normal : (3,) ndarray
        Einheitsnormale der Best-Fit-Ebene (Eigenvektor zur kleinsten Eigenvalue).
    eigenvalues : (3,) ndarray
        Aufsteigend sortierte Eigenwerte der Kovarianzmatrix (λ_min, λ_mid, λ_max).
    sigma : float
        Orthogonale Rauigkeit σ(D): StdAbw der senkrechten Abstände aller Punkte zur Ebene.
    """
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    covariance = (centered_points.T @ centered_points) / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    normal_vector = eigenvectors[:, 0]
    norm = np.linalg.norm(normal_vector)
    if norm > 0:
        normal_vector = normal_vector / norm

    ortho = centered_points @ normal_vector
    sigma = float(np.std(ortho, ddof=1))
    return centroid, normal_vector, eigenvalues, sigma


class RadiusScanStrategy:
    """
    Radius-basierter Scan (paper-/CC-nah):
    - Nachbarschaften per KD-Tree im Kugelradius D/2
    - PCA-Ebene pro Nachbarschaft; Normale = Eigenvektor zur kleinsten Eigenvalue
    - Roughness σ(D) = StdAbw der orthogonalen Abstände ALLER Nachbarn zur PCA-Ebene
    - Aggregation (λ_min, σ, Populationsgrößen, Coverage) über alle Nachbarschaften
    - Skalen: D_i = min_spacing * 2^i  für i in [i_min, i_max]
    """

    def __init__(
        self,
        i_min: int = -3,           # z.B. 2^-3 = 1/8
        i_max: int = 8,            # z.B. 2^8  = 256
        sample_size: Optional[int] = None,
        min_points: int = 10,
        log_each_scale: bool = True,
        signed: bool = False,
        up_dir: np.ndarray | None = None,
    ) -> None:
        self.i_min = i_min
        self.i_max = i_max
        self.sample_size = sample_size
        self.min_points = min_points
        self.log_each_scale = log_each_scale
        self.signed = signed
        self.up_dir = up_dir

    # 1) Einzel-Skala evaluieren (D via neighborhood_radius = D/2)
    def evaluate_radius_scale(self, point_cloud: np.ndarray, neighborhood_radius: float) -> Dict:
        """
        Bewertet eine einzelne Normalskala D anhand eines Nachbarschaftsradius D/2 (M3C2, Schritt 1).
        """
        if point_cloud.dtype != np.float64:
            point_cloud = point_cloud.astype(np.float64, copy=False)

        neighbor_search = NearestNeighbors(radius=neighborhood_radius, algorithm="kd_tree")
        neighbor_search.fit(point_cloud)
        neighbor_indices_list = neighbor_search.radius_neighbors(point_cloud, return_distance=False)

        valid_neighbors_count = 0
        sigma_values: list[float] = []
        lambda_min_values: list[float] = []
        population_sizes: list[int] = []

        for neighbor_indices in neighbor_indices_list:
            if neighbor_indices.size < self.min_points:
                continue

            neighbor_points = point_cloud[neighbor_indices]
            _, normal_vec, eigenvalues, sigma = _fit_plane_pca(neighbor_points)

            lambda_min = float(eigenvalues[0])  # kleinster Eigenwert (Planarität)

            sigma_values.append(float(sigma))
            lambda_min_values.append(lambda_min)
            population_sizes.append(int(neighbor_indices.size))
            valid_neighbors_count += 1

        total_points = int(len(point_cloud))
        mean_population = float(np.mean(population_sizes)) if population_sizes else 0.0
        std_population = float(np.std(population_sizes)) if population_sizes else 0.0
        perc97_population = int(np.percentile(population_sizes, 97)) if population_sizes else 0

        mean_sigma = float(np.mean(sigma_values)) if sigma_values else np.nan
        mean_lambda3 = float(np.mean(lambda_min_values)) if lambda_min_values else np.nan

        scale_D = 2.0 * neighborhood_radius  # D = 2 * (D/2)
        relative_roughness = (mean_sigma / scale_D) if (scale_D > 0 and not np.isnan(mean_sigma)) else np.nan
        coverage = (valid_neighbors_count / total_points) if total_points > 0 else 0.0

        return {
            "scale":               float(scale_D),
            "valid_normals":       int(valid_neighbors_count),
            "total_points":        total_points,
            "mean_population":     mean_population,
            "std_population":      std_population,
            "perc97_population":   perc97_population,
            "roughness":           mean_sigma,          # mean σ(D)
            "mean_lambda3":        mean_lambda3,        # Planaritätsmaß
            "relative_roughness":  relative_roughness,  # σ(D)/D
            "coverage":            coverage,
        }

    # 2) Mehrere Skalen scannen (D_i = min_spacing * 2^i)
    def scan(self, points: np.ndarray, min_spacing: float) -> List[ScaleScan]:
        pts = points
        if self.sample_size and len(pts) > self.sample_size:
            idx = np.random.choice(len(pts), size=self.sample_size, replace=False)
            pts = pts[idx]
            logging.info(f"[RadiusScan] Subsample: {self.sample_size}/{len(points)}")

        scans: List[ScaleScan] = []
        for level in range(self.i_min, self.i_max + 1):
            D = float(min_spacing) * (2.0 ** float(level))
            radius = D / 2.0
            res = self.evaluate_radius_scale(pts, radius)

            scans.append(
                ScaleScan(
                    scale=res["scale"],                   # == D
                    valid_normals=res["valid_normals"],
                    mean_population=res["mean_population"],
                    roughness=res["roughness"],           # mean σ(D)
                    coverage=res["coverage"],
                    mean_lambda3=res["mean_lambda3"],     # Planaritätsmaß
                    total_points=res["total_points"],
                    std_population=res["std_population"],
                    perc97_population=res["perc97_population"],
                    relative_roughness=res["relative_roughness"],  # σ(D)/D
                )
            )

            if self.log_each_scale:
                logging.info(
                    "[RadiusScan] D=%g | pop=%4.1f±%3.1f | 97%%>%d | valid=%d/%d (%s) | sigma=%g | lambda_min=%g | Sigma/D=%s",
                    D,
                    res["mean_population"],
                    res["std_population"],
                    res["perc97_population"],
                    res["valid_normals"],
                    res["total_points"],
                    f"{res['coverage']:.0%}",
                    res["roughness"],
                    res["mean_lambda3"],
                    "nan" if np.isnan(res["relative_roughness"]) else f"{res['relative_roughness']:.4f}",
                )
        return scans
