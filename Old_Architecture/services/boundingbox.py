# clip_obb_overlap_multi.py
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Tuple

def read_ply(path: str) -> o3d.geometry.PointCloud:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden:\n  {p}")
    pc = o3d.io.read_point_cloud(str(p))
    if pc.is_empty():
        raise RuntimeError(f"Leere/ungültige PLY gelesen (evtl. falsches Format?):\n  {p}")
    return pc

def to_local_frame(xyz: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    # lokales OBB-Frame: erst zum Zentrum, dann Rotation transponiert (Inverse)
    return (xyz - C) @ R

def to_world_frame(xyz_local: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    # zurück: erst drehen, dann verschieben
    return xyz_local @ R.T + C

def clip_obbf_aligned_many(in_paths: List[str], out_paths: List[str], pad: float = 0.0):
    if len(in_paths) != len(out_paths):
        raise ValueError("in_paths und out_paths müssen gleich lang sein.")

    # 1) Einlesen
    pcs = [read_ply(p) for p in in_paths]

    # 2) OBB der ersten Cloud als Referenz
    ref_obb = pcs[0].get_oriented_bounding_box()
    C = ref_obb.center                  # (3,)
    R = ref_obb.R                       # (3,3) Rotation (Spalten = Achsen des OBBs)
    # Hinweis: Open3D nutzt R, so dass p_local = (p - C) @ R

    # 3) Alle Clouds ins lokale OBB-Frame der ersten Cloud transformieren
    xyz_locals = []
    color_locals = []
    normal_locals = []
    for pc in pcs:
        xyz = np.asarray(pc.points)
        xyz_l = to_local_frame(xyz, R, C)
        xyz_locals.append(xyz_l)

        if pc.has_colors():
            color_locals.append(np.asarray(pc.colors))
        else:
            color_locals.append(None)

        if pc.has_normals():
            # Normale nur rotieren (nicht verschieben), im lokalen Frame: n_local = n @ R
            nrm = np.asarray(pc.normals)
            normal_locals.append(nrm @ R)
        else:
            normal_locals.append(None)

    # 4) Gemeinsame AABB-Schnittmenge im lokalen Frame berechnen
    mins = np.vstack([xyz_l.min(axis=0) for xyz_l in xyz_locals])
    maxs = np.vstack([xyz_l.max(axis=0) for xyz_l in xyz_locals])
    inter_min = mins.max(axis=0)
    inter_max = maxs.min(axis=0)

    # Optional: leichtes Padding
    if pad != 0.0:
        inter_min -= pad
        inter_max += pad

    if np.any(inter_min >= inter_max):
        raise RuntimeError("Keine gemeinsame OBB-Überlappung gefunden (in ref-OBB-Frame).")

    # 5) Clipping & Rücktransformation
    for xyz_l, cols, nrms, outp in zip(xyz_locals, color_locals, normal_locals, out_paths):
        mask = np.all((xyz_l >= inter_min) & (xyz_l <= inter_max), axis=1)
        clipped_local = xyz_l[mask]

        # zurück in Weltkoordinaten
        clipped_world = to_world_frame(clipped_local, R, C)

        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(clipped_world)

        if cols is not None:
            pc_out.colors = o3d.utility.Vector3dVector(cols[mask])

        if nrms is not None:
            # Normale zurück ins Weltkoordinatensystem: n_world = n_local @ R.T
            nrms_world = (nrms[mask]) @ R.T
            pc_out.normals = o3d.utility.Vector3dVector(nrms_world)

        outp = str(Path(outp).expanduser().resolve())
        o3d.io.write_point_cloud(outp, pc_out, write_ascii=False, compressed=False)
        print(f"{outp}  |  {xyz_l.shape[0]} -> {clipped_local.shape[0]} Punkte")

    print("Gemeinsame OBB-Grenzen (im lokalen Frame der ersten Cloud):")
    print("  min:", inter_min, "\n  max:", inter_max)

if __name__ == "__main__":
    # === HIER deine drei (oder mehr) Dateien eintragen ===
    in_paths  = [
        r"data\Multi-illumination\Job_0378_8400-110\1-1\Job_0378_8400-110-rad-1-1_cloud.ply",
        r"data\Multi-illumination\Job_0378_8400-110\1-1\Job_0378_8400-110-rad-1-1-AI_cloud.ply",
    ]
    out_paths = [
        r"data\Multi-illumination\Job_0378_8400-110\1-1\Job_0378_8400-110-rad-1-1_cloud_overlap.ply",
        r"data\Multi-illumination\Job_0378_8400-110\1-1\Job_0378_8400-110-rad-1-1-AI_cloud_overlap.ply",
    ]

    # Kleines numerisches Polster (optional), z.B. 1e-6 oder 1e-4 je nach Maßstab
    clip_obbf_aligned_many(in_paths, out_paths, pad=0.0)
