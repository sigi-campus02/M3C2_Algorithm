from __future__ import annotations
import numpy as np
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from orchestration.strategies import ScaleScan


class ParamEstimator:
    """
    Kapselt Spacing-Schätzung + Scale-Scan + finale Auswahl.
    """
    @staticmethod
    def estimate_min_spacing(points: np.ndarray, k: int = 6) -> float:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        min_spacing = float(np.mean(distances[:, 1:]))
        return min_spacing

    @staticmethod
    def scan_scales(points: np.ndarray, strategy, avg_spacing: float) -> List[ScaleScan]:
        scans = strategy.scan(points, avg_spacing)
        return scans

    @staticmethod
    def select_scales(scans: List[ScaleScan]) -> Tuple[float, float]:
        """
        Paper-nahe Auswahl:
        - Primär: minimale mittlere λ_min (Planarität), dann hohe Abdeckung, dann geringe σ(D)
        - D/σ-Regel: bevorzuge ersten Kandidaten mit D/sigma >= 25
        - Normal D = gewählte Stufe
        - Projektion d = exakt die nächsthöhere getestete Stufe (> D); Fallback: d = D
        """
        if not scans:
            raise ValueError("Keine Scales gefunden.")

        # Nur valide Scans verwenden
        valid = [
            s for s in scans
            if (
                s.roughness is not None and not np.isnan(s.roughness)
                and s.mean_lambda3 is not None and not np.isnan(s.mean_lambda3)
                and s.valid_normals is not None and s.valid_normals > 0
            )
        ]

        # Fallback (keine validen Scans): nimm Median als D, nächsthöhere Stufe als d
        if not valid:
            ladder = sorted({float(s.scale) for s in scans})
            mid_idx = len(ladder) // 2
            normal = ladder[mid_idx]
            projection = ladder[mid_idx + 1] if mid_idx + 1 < len(ladder) else ladder[mid_idx]
            return float(normal), float(projection)

        # Primär Planarität (λ_min), dann Abdeckung, dann geringe σ(D)
        valid.sort(key=lambda s: (float(s.mean_lambda3), -int(s.valid_normals), float(s.roughness)))

        # D/σ-Regel: ersten Kandidaten nehmen, der sie erfüllt; sonst bestes λ_min
        chosen = None
        for s in valid:
            if s.roughness > 0 and (float(s.scale) / float(s.roughness)) >= 25.0:
                chosen = s
                break
        if chosen is None:
            chosen = valid[0]

        # Leiter (einmalig aus den getesteten Skalen)
        ladder = sorted({float(s.scale) for s in scans})

        # Index der gewählten Stufe (tolerant „snappen“)
        EPS = 1e-12
        try:
            idx = next(i for i, v in enumerate(ladder) if abs(v - float(chosen.scale)) <= EPS)
        except StopIteration:
            idx = min(range(len(ladder)), key=lambda i: abs(ladder[i] - float(chosen.scale)))

        # Normal = gewählte Stufe, Projektion = exakt die nächsthöhere Stufe (eins drüber)
        normal = ladder[idx]
        projection = ladder[idx + 1] if (idx + 1) < len(ladder) else ladder[idx]  # am oberen Rand: d = D

        return float(normal), float(projection)
