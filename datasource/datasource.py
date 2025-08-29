
from __future__ import annotations
import os
import logging
from typing import Tuple
import numpy as np
import py4dgeo
from plyfile import PlyData
import laspy

class DataSource:
    """
    Erwartetes Layout: <folder>/{mov,ref}.{xyz|las|laz|ply|obj|gpc}
    Priorität:
      1) Wenn BEIDE .xyz  -> py4dgeo.read_from_xyz
      2) Wenn BEIDE .las/.laz (gemischt ok) -> py4dgeo.read_from_las
      3) Wenn BEIDE .ply und read_from_ply existiert -> py4dgeo.read_from_ply
      4) Sonst: pro Seite in .xyz konvertieren und read_from_xyz
    Liefert: (mov_epoch, ref_epoch, corepoints_np)
    """

    def __init__(
            self, folder: str,
            mov_basename: str = "mov",
            ref_basename: str = "ref",
            mov_as_corepoints: bool = True,
            use_subsampled_corepoints: int = 1) -> None:
        self.folder = folder
        self.mov_base = os.path.join(folder, mov_basename)
        self.ref_base = os.path.join(folder, ref_basename)
        self.mov_as_corepoints = mov_as_corepoints
        self.use_subsampled_corepoints = use_subsampled_corepoints
        os.makedirs(folder, exist_ok=True)

    @staticmethod
    def _exists(p: str) -> bool:
        return os.path.exists(p)

    def _detect(self, base: str) -> tuple[str | None, str | None]:
        """Gibt (kind, path) zurück, wobei kind in {'xyz','laslike','ply','obj','gpc'} oder None ist."""
        xyz, las, laz, ply, obj, gpc = (
            base + ".xyz",
            base + ".las",
            base + ".laz",
            base + ".ply",
            base + ".obj",
            base + ".gpc",
        )
        if self._exists(xyz):
            return "xyz", xyz
        if self._exists(las) or self._exists(laz):
            # egal ob .las oder .laz: beides ist 'laslike'
            return "laslike", las if self._exists(las) else laz
        if self._exists(ply):
            return "ply", ply
        if self._exists(obj):
            return "obj", obj
        if self._exists(gpc):
            return "gpc", gpc
        return None, None

    def _read_las_or_laz_to_xyz_array(self, path: str) -> np.ndarray:
        if laspy is None:
            raise RuntimeError("LAS/LAZ gefunden, aber 'laspy' ist nicht installiert.")
        # Für .laz ggf. Backend nötig: pip install 'laspy[lazrs]'
        try:
            las = laspy.read(path)
        except ModuleNotFoundError as e:
            raise RuntimeError("LAZ erkannt, bitte 'pip install \"laspy[lazrs]\"' installieren.") from e
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

    def _read_obj_to_xyz_array(self, path: str) -> np.ndarray:
        """Parst ein einfaches Wavefront-OBJ mit 'v x y z'-Zeilen."""
        vertices: list[list[float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.asarray(vertices, dtype=np.float64)

    def _read_gpc_to_xyz_array(self, path: str) -> np.ndarray:
        """Lädt .gpc als einfache whitespace-separierte XYZ-Tabelle."""
        return np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2))

    def _ensure_xyz(self, base: str, detected: tuple[str | None, str | None]) -> str:
        """Sorgt dafür, dass base.xyz existiert (konvertiert bei Bedarf)."""
        kind, path = detected
        xyz = base + ".xyz"
        if kind == "xyz" and path:
            return path
        if kind == "laslike" and path:
            logging.info(f"[{base}] Konvertiere LAS/LAZ → XYZ …")
            arr = self._read_las_or_laz_to_xyz_array(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz
        if kind == "ply" and path:
            if PlyData is None:
                raise RuntimeError("PLY gefunden, aber 'plyfile' ist nicht installiert.")
            logging.info(f"[{base}] Konvertiere PLY → XYZ …")
            ply = PlyData.read(path)
            v = ply["vertex"]
            arr = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz
        if kind == "obj" and path:
            logging.info(f"[{base}] Konvertiere OBJ → XYZ …")
            arr = self._read_obj_to_xyz_array(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz
        if kind == "gpc" and path:
            logging.info(f"[{base}] Konvertiere GPC → XYZ …")
            arr = self._read_gpc_to_xyz_array(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz
        raise FileNotFoundError(f"Fehlt: {base}.xyz/.las/.laz/.ply/.obj/.gpc")

    def load_points(self) -> Tuple[py4dgeo.Epoch, py4dgeo.Epoch, np.ndarray]:
        m_kind, m_path = self._detect(self.mov_base)
        r_kind, r_path = self._detect(self.ref_base)

        # 1) beide xyz
        if m_kind == r_kind == "xyz":
            logging.info("Nutze py4dgeo.read_from_xyz")
            mov, ref = py4dgeo.read_from_xyz(m_path, r_path)
            logging.info(f"Mov Points: {mov.cloud.shape}")
            logging.info(f"Ref Points: {ref.cloud.shape}")

        # 2) beide laslike (las/laz gemischt erlaubt)
        elif (m_kind == "laslike") and (r_kind == "laslike"):
            logging.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            mov, ref = py4dgeo.read_from_las(m_path, r_path)
            logging.info(f"Mov Points: {mov.cloud.shape}")
            logging.info(f"Ref Points: {ref.cloud.shape}")

        # 3) beide ply (nur wenn API vorhanden)
        elif (m_kind == r_kind == "ply") and hasattr(py4dgeo, "read_from_ply"):
            logging.info("Nutze py4dgeo.read_from_ply")
            mov, ref = py4dgeo.read_from_ply(m_path, r_path)
            logging.info(f"Mov Points: {mov.cloud.shape}")
            logging.info(f"Ref Points: {ref.cloud.shape}")

        # 4) Mischtypen → zu XYZ
        else:
            m_xyz = self._ensure_xyz(self.mov_base, (m_kind, m_path))
            r_xyz = self._ensure_xyz(self.ref_base, (r_kind, r_path))
            logging.info("Mischtypen → konvertiert zu XYZ → py4dgeo.read_from_xyz")
            mov, ref = py4dgeo.read_from_xyz(m_xyz, r_xyz)
            logging.info(f"Mov Points: {mov.cloud.shape}")
            logging.info(f"Ref Points: {ref.cloud.shape}")

        # Corepoints: Nx3
        if self.mov_as_corepoints:
            logging.info(f"Nutze mov als Corepoints und nutze Subsamling: {self.use_subsampled_corepoints}")
            corepoints = mov.cloud[::self.use_subsampled_corepoints] if hasattr(mov, "cloud") else mov
            logging.info(f"Corepoints: {corepoints.shape}")
        else:
            logging.info(f"Nutze ref als Corepoints und nutze Subsamling: {self.use_subsampled_corepoints}")
            corepoints = ref.cloud[::self.use_subsampled_corepoints] if hasattr(ref, "cloud") else ref
            logging.info(f"Corepoints: {corepoints.shape}")

        if not isinstance(corepoints, np.ndarray):
            raise TypeError("Unerwarteter Typ für corepoints; erwarte np.ndarray (Nx3).")
        return mov, ref, corepoints