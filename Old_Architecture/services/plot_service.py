from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, weibull_min, probplot
from config.plot_config import PlotConfig, PlotOptions
import re
from collections import defaultdict
from collections import OrderedDict

logger = logging.getLogger(__name__)

class PlotService:
    CASE_ORDER = ("CASE1", "CASE2", "CASE3", "CASE4")

    @staticmethod
    def _labels_by_case_map(case_map: Dict[str, str], case_order: Tuple[str, ...] | None = None) -> List[str]:
        """Stabile Labels: erst CASE1, dann CASE2, ...; innerhalb eines CASE in eingelesener Reihenfolge."""
        order = case_order or PlotService.CASE_ORDER
        labels: List[str] = []
        for c in order:
            labels.extend([lbl for lbl, cas in case_map.items() if cas == c])
        return labels
    

    @staticmethod
    def _reorder_data(data: Dict[str, np.ndarray], labels_order: List[str]) -> "OrderedDict[str, np.ndarray]":
        """Daten in gewünschter Label-Reihenfolge anordnen."""
        return OrderedDict((lbl, data[lbl]) for lbl in labels_order if lbl in data)

    @staticmethod
    def _colors_by_case(labels_order: List[str], label_to_case: Dict[str, str], case_colors: Dict[str, str]) -> Dict[str, str]:
        """Case → Farbe, stabil pro Label in labels_order."""
        return {lbl: case_colors.get(label_to_case.get(lbl, "CASE1"), "#777777") for lbl in labels_order}
    
    @staticmethod
    def _scan_distance_files_by_index(data_dir: str, versions=("python", "CC")):
        """
        Findet Dateien wie:
        {ver}_a-<i>[-AI]-b-<i>[-AI]_m3c2_distances.txt
        {ver}_a-<i>[-AI]-b-<i>[-AI]_m3c2_distances_coordinates_inlier_*.txt
        Gruppiert pro Index i.
        Labels: nur 'a-<i>[-AI] vs b-<i>[-AI]'
        Cases: CASE1=a-i vs b-i, CASE2=a-i vs b-i-AI, CASE3=a-i-AI vs b-i, CASE4=beide AI (falls vorhanden)
        """
        import re, os
        from collections import defaultdict

        logger.info(f"[Scan] Scanne Distanzdateien in {data_dir} für Versionen: {versions}")

        pat_with = re.compile(
            r'^(?P<ver>(?:' + "|".join(versions) + r'))_'
            r'(?P<mov>[ab]-\d+(?:-AI)?)'
            r'-'
            r'(?P<ref>[ab]-\d+(?:-AI)?)'
            r'_m3c2_distances\.txt$', re.IGNORECASE
        )
        pat_inl = re.compile(
            r'^(?P<ver>(?:' + "|".join(versions) + r'))_'
            r'(?P<mov>[ab]-\d+(?:-AI)?)'
            r'-'
            r'(?P<ref>[ab]-\d+(?:-AI)?)'
            r'_m3c2_distances_coordinates_inlier_(?P<meth>[a-zA-Z0-9_]+)\.txt$', re.IGNORECASE
        )

        def idx_of(tag: str) -> int:
            m = re.match(r'^[ab]-(\d+)(?:-AI)?$', tag, re.IGNORECASE)
            return int(m.group(1)) if m else -1

        def to_case_and_label(mov: str, ref: str, i: int) -> tuple[str, str]:
            mov_ai = "-AI" in mov
            ref_ai = "-AI" in ref
            if not mov_ai and not ref_ai:
                return "CASE1", f"a-{i} vs b-{i}"
            if not mov_ai and ref_ai:
                return "CASE2", f"a-{i} vs b-{i}-AI"
            if mov_ai and not ref_ai:
                return "CASE3", f"a-{i}-AI vs b-{i}"
            if mov_ai and ref_ai:
                return "CASE4", f"a-{i}-AI vs b-{i}-AI"

        # Struktur: pro i -> WITH/INLIER (label -> array) + CASE maps (label -> CASEX)
        per_index = defaultdict(lambda: {"WITH": {}, "INLIER": {}, "CASE_WITH": {}, "CASE_INLIER": {}})

        for name in os.listdir(data_dir):
            p = os.path.join(data_dir, name)
            if not os.path.isfile(p):
                continue

            mW = pat_with.match(name)
            if mW:
                mov, ref = mW.group("mov"), mW.group("ref")
                i_mov, i_ref = idx_of(mov), idx_of(ref)
                if i_mov == i_ref and i_mov != -1:
                    i = i_mov
                    cas, label = to_case_and_label(mov, ref, i)
                    try:
                        arr = PlotService._load_1col_distances(p)
                        per_index[i]["WITH"][label] = arr
                        per_index[i]["CASE_WITH"][label] = cas
                    except Exception as e:
                        logger.warning(f"[Scan] Laden fehlgeschlagen (WITH: {name}): {e}")
                continue

            mI = pat_inl.match(name)
            if mI:
                mov, ref = mI.group("mov"), mI.group("ref")
                i_mov, i_ref = idx_of(mov), idx_of(ref)
                if i_mov == i_ref and i_mov != -1:
                    i = i_mov
                    cas, label = to_case_and_label(mov, ref, i)
                    try:
                        arr = PlotService._load_coordinates_inlier_distances(p)
                        per_index[i]["INLIER"][label] = arr
                        per_index[i]["CASE_INLIER"][label] = cas
                    except Exception as e:
                        logger.warning(f"[Scan] Laden fehlgeschlagen (INLIER: {name}): {e}")
                continue

        # Farben pro Case (stabil über alle Parts)
        case_colors = {
            "CASE1": "#1f77b4",  # a-i vs b-i
            "CASE2": "#ff7f0e",  # a-i vs b-i-AI
            "CASE3": "#2ca02c",  # a-i-AI vs b-i
            "CASE4": "#9467bd",  # beide AI (falls vorhanden)
        }
        return per_index, case_colors


    @classmethod
    def overlay_by_index(
        cls,
        data_dir: str,
        outdir: str,
        versions=("python",),
        bins: int = 256,
        options: PlotOptions | None = None,
        skip_existing: bool = True,
    ):
        options = options or PlotOptions()  # alle True als Default (wie früher)
        os.makedirs(outdir, exist_ok=True)
        per_index, case_colors = cls._scan_distance_files_by_index(data_dir, versions=versions)

        def _png(fid: str, mode: str, suffix: str) -> str:
            return os.path.join(outdir, f"{fid}_{mode}_{suffix}.png")

        if not per_index:
            logger.warning("[Report] Keine Distanzdateien gefunden in %s.", data_dir)
            return

        for i in sorted(per_index.keys()):
            fid = f"Part_{i}"

            # ----- WITH -----
            data_with = per_index[i]["WITH"]
            if data_with:
                case_map_w = per_index[i]["CASE_WITH"]
                labels_w   = cls._labels_by_case_map(case_map_w)                     # feste Reihenfolge
                data_with  = cls._reorder_data(data_with, labels_w)                  # in Order bringen
                colors_w   = cls._colors_by_case(labels_w, case_map_w, case_colors)  # Case-Farben

                need_range = options.plot_hist or options.plot_gauss or options.plot_weibull
                if need_range:
                    data_min, data_max, x = cls._get_common_range(data_with)
                gauss_with = {k: norm.fit(v) for k, v in data_with.items()} if options.plot_gauss else {}

                if options.plot_hist:
                    cls._plot_overlay_histogram(fid, "WITH", data_with, bins, data_min, data_max, colors_w, outdir,
                        title_text=f"Histogram – Part {i} / incl. Outliers", labels_order=labels_w)
                if options.plot_gauss:
                    cls._plot_overlay_gauss(fid, "WITH", data_with, gauss_with, x, colors_w, outdir,
                        title_text=f"Gaussian fit – Part {i} / incl. Outliers", labels_order=labels_w)
                if options.plot_weibull:
                    cls._plot_overlay_weibull(fid, "WITH", data_with, x, colors_w, outdir,
                        title_text=f"Weibull fit – Part {i} / incl. Outliers", labels_order=labels_w)
                if options.plot_box:
                    cls._plot_overlay_boxplot(fid, "WITH", data_with, colors_w, outdir,
                        title_text=f"Box plot – Part {i} / incl. Outliers", labels_order=labels_w)
                if options.plot_qq:
                    cls._plot_overlay_qq(fid, "WITH", data_with, colors_w, outdir,
                        title_text=f"Q–Q plot – Part {i} / incl. Outliers", labels_order=labels_w)

            # ----- INLIER -----
            data_inl = per_index[i]["INLIER"]
            if data_inl:
                case_map_i = per_index[i]["CASE_INLIER"]
                labels_i   = cls._labels_by_case_map(case_map_i)
                data_inl   = cls._reorder_data(data_inl, labels_i)
                colors_i   = cls._colors_by_case(labels_i, case_map_i, case_colors)

                need_range = options.plot_hist or options.plot_gauss or options.plot_weibull
                if need_range:
                    data_min, data_max, x = cls._get_common_range(data_inl)
                gauss_inl = {k: norm.fit(v) for k, v in data_inl.items()} if options.plot_gauss else {}

                if options.plot_hist:
                    cls._plot_overlay_histogram(fid, "INLIER", data_inl, bins, data_min, data_max, colors_i, outdir,
                        title_text=f"Histogram – Part {i} / excl. Outliers", labels_order=labels_i)
                if options.plot_gauss:
                    cls._plot_overlay_gauss(fid, "INLIER", data_inl, gauss_inl, x, colors_i, outdir,
                        title_text=f"Gaussian fit – Part {i} / excl. Outliers", labels_order=labels_i)
                if options.plot_weibull:
                    cls._plot_overlay_weibull(fid, "INLIER", data_inl, x, colors_i, outdir,
                        title_text=f"Weibull fit – Part {i} / excl. Outliers", labels_order=labels_i)
                if options.plot_box:
                    cls._plot_overlay_boxplot(fid, "INLIER", data_inl, colors_i, outdir,
                        title_text=f"Box plot – Part {i} / excl. Outliers", labels_order=labels_i)
                if options.plot_qq:
                    cls._plot_overlay_qq(fid, "INLIER", data_inl, colors_i, outdir,
                        title_text=f"Q–Q plot – Part {i} / excl. Outliers", labels_order=labels_i)

            # ----- DUAL Grouped Bars (WITH vs INLIER) -----
            if options.plot_grouped_bar and per_index[i]["WITH"] and per_index[i]["INLIER"]:
                combined_case_map = {**per_index[i]["CASE_WITH"], **per_index[i]["CASE_INLIER"]}
                labels_dual = cls._labels_by_case_map(combined_case_map)
                colors_dual = cls._colors_by_case(labels_dual, combined_case_map, case_colors)
                cls._plot_grouped_bar_means_stds_dual_by_case(
                    fid=f"Part_{i}",
                    data_with=per_index[i]["WITH"],
                    data_inlier=per_index[i]["INLIER"],
                    colors=colors_dual,
                    outdir=outdir,
                    title_text="Means & Std – WITH vs INLIER",
                    labels_order=labels_dual,
                )


    @classmethod
    def overlay_plots(cls, config: PlotConfig, options: PlotOptions) -> None:
        colors = config.ensure_colors()
        os.makedirs(config.path, exist_ok=True)

        # ---- WITH (inkl. Outlier) sammeln ----
        data_with_all: Dict[str, np.ndarray] = {}
        for fid in config.folder_ids:
            data_with, _ = cls._load_data(fid, config.filenames, config.versions)
            if not data_with:
                logger.warning(f"[Report] Keine WITH-Daten für {fid} gefunden.")
                continue
            data_with_all.update(data_with)

        if not data_with_all:
            logger.warning("[Report] Keine Daten gefunden – keine Plots erzeugt.")
            return

        # ---- INLIER (aus *_coordinates_inlier_std.txt) sammeln ----
        data_inlier_all: Dict[str, np.ndarray] = {}
        for fid in config.folder_ids:
            for v in config.versions:
                label = f"{v}_{fid}"
                base_inl = f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances_coordinates_inlier_std.txt"
                path_inl = cls._resolve(fid, base_inl)
                logger.info(f"[Report] Lade INLIER: {path_inl}")
                if not os.path.exists(path_inl):
                    logger.warning(f"[Report] Datei fehlt (INLIER): {path_inl}")
                    continue
                try:
                    arr = cls._load_coordinates_inlier_distances(path_inl)
                except Exception as e:
                    logger.error(f"[Report] Laden fehlgeschlagen (INLIER: {path_inl}): {e}")
                    continue
                if arr.size:
                    data_inlier_all[label] = arr

        # Gemeinsamer Range (über WITH, damit Seiten vergleichbar sind)
        data_min, data_max, x = cls._get_common_range(data_with_all)

        # EIN Satz Overlays für ALLE Folder gemeinsam
        fid = "ALLFOLDERS"

        # -------- Seite 1: WITH --------
        fname = "ALL_WITH"
        gauss_with = {k: norm.fit(v) for k, v in data_with_all.items() if v.size}
        if options.plot_hist:
            cls._plot_overlay_histogram(fid, fname, data_with_all, config.bins, data_min, data_max, colors, config.path)
        if options.plot_gauss:
            cls._plot_overlay_gauss(fid, fname, data_with_all, gauss_with, x, colors, config.path)
        if options.plot_weibull:
            cls._plot_overlay_weibull(fid, fname, data_with_all, x, colors, config.path)
        if options.plot_box:
            cls._plot_overlay_boxplot(fid, fname, data_with_all, colors, config.path)
        if options.plot_qq:
            cls._plot_overlay_qq(fid, fname, data_with_all, colors, config.path)
        if options.plot_grouped_bar:
            cls._plot_grouped_bar_means_stds_dual(fid, fname, data_with_all, data_inlier_all, colors, config.path)
        if options.plot_violin:
            cls._plot_overlay_violin(fid, fname, data_with_all, colors, config.path)
        logger.info(f"[Report] PNGs für {fid} (WITH) erzeugt.")

        # -------- Seite 2: INLIER --------
        fname = "ALL_INLIER"
        if data_inlier_all:
            gauss_inl = {k: norm.fit(v) for k, v in data_inlier_all.items() if v.size}
            if options.plot_hist:
                cls._plot_overlay_histogram(fid, fname, data_inlier_all, config.bins, data_min, data_max, colors, config.path)
            if options.plot_gauss:
                cls._plot_overlay_gauss(fid, fname, data_inlier_all, gauss_inl, x, colors, config.path)
            if options.plot_weibull:
                cls._plot_overlay_weibull(fid, fname, data_inlier_all, x, colors, config.path)
            if options.plot_box:
                cls._plot_overlay_boxplot(fid, fname, data_inlier_all, colors, config.path)
            if options.plot_qq:
                cls._plot_overlay_qq(fid, fname, data_inlier_all, colors, config.path)
            if options.plot_grouped_bar:
                cls._plot_grouped_bar_means_stds_dual(fid, fname, data_with_all, data_inlier_all, colors, config.path)
            if options.plot_violin:
                cls._plot_overlay_violin(fid, fname, data_inlier_all, colors, config.path)
            logger.info(f"[Report] PNGs für {fid} (INLIER) erzeugt.")
        else:
            logger.warning("[Report] Keine INLIER-Daten gefunden – zweite Seite bleibt leer.")


    @classmethod
    def summary_pdf(cls, config: PlotConfig) -> None:
        plot_types = [
            ("OverlayHistogramm", "Histogramm", (0, 0)),
            ("Boxplot", "Boxplot", (0, 1)),
            ("OverlayGaussFits", "Gauss-Fit", (0, 2)),
            ("OverlayWeibullFits", "Weibull-Fit", (1, 0)),
            ("QQPlot", "Q-Q-Plot", (1, 1)),
            ("GroupedBar_Mean_Std", "Mittelwert & Std Dev", (1, 2)),
        ]

        fid = "ALLFOLDERS"
        outfile = os.path.join(config.path, f"{fid}_comparison_report.pdf")
        pdf = PdfPages(outfile)

        def _add_page(suffix_label: str, title_suffix: str):
            fig, axs = plt.subplots(2, 3, figsize=(24, 16))
            for suffix, title, (row, col) in plot_types:
                ax = axs[row, col]
                png = os.path.join(config.path, f"{fid}_{suffix_label}_{suffix}.png")
                if os.path.exists(png):
                    img = mpimg.imread(png)
                    ax.imshow(img)
                    ax.axis("off")
                    ax.set_title(title, fontsize=22)
                else:
                    ax.axis("off")
                    ax.set_title(f"{title}\n(nicht gefunden)", fontsize=18)
            plt.suptitle(f"{fid} – Vergleichsplots ({title_suffix})", fontsize=28)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08, wspace=0.08, hspace=0.15)
            pdf.savefig(fig)
            plt.close(fig)

        # Seite 1: WITH
        _add_page("ALL_WITH", "inkl. Outlier")

        # Seite 2: INLIER
        _add_page("ALL_INLIER", "ohne Outlier (Inlier)")

        pdf.close()
        logger.info(f"[Report] Zusammenfassung gespeichert: {outfile}")


    @classmethod
    def build_parts_pdf(
        cls,
        outdir: str,
        pdf_path: str | None = None,
        include_with: bool = True,
        include_inlier: bool = True,
    ) -> str:
        """
        Erzeugt eine PDF, pro Part genau EINE Seite.
        Layout pro Seite: 2 Zeilen × 3 Spalten = 6 Plots:
        Zeile 1: Histogramm | Gaussian fit | Weibull fit
        Zeile 2: Box plot   | Q–Q plot     | Means & Std (DUAL)
        WICHTIG: Diese PDF enthält ENTWEDER WITH ODER INLIER Plots (nicht beide).
        """
        # --- genau einen Modus zulassen ---
        if include_with == include_inlier:
            raise ValueError("Bitte genau einen Modus wählen: include_with XOR include_inlier.")
        mode = "WITH" if include_with else "INLIER"
        subtitle = "incl. outliers" if include_with else "excl. outliers"

        # --- Part-IDs sammeln (auch wenn nur DUAL existiert) ---
        part_ids: list[int] = []
        pat5   = re.compile(r"^Part_(\d+)_(WITH|INLIER)_(OverlayHistogramm|OverlayGaussFits|OverlayWeibullFits|Boxplot|QQPlot)\.png$")
        patDual= re.compile(r"^Part_(\d+)_DUAL_GroupedBar_Mean_Std\.png$")
        for fn in os.listdir(outdir):
            m = pat5.match(fn)
            if m:
                part_ids.append(int(m.group(1)))
                continue
            m = patDual.match(fn)
            if m:
                part_ids.append(int(m.group(1)))
        part_ids = sorted(set(part_ids))
        if not part_ids:
            logger.warning("[Report] No part PNGs found in %s – nothing to summarize.", outdir)
            return ""

        pdf_path = pdf_path or os.path.join(outdir, "parts_summary.pdf")

        # 6 Plot-Typen in gewünschter Reihenfolge (die letzten sind DUAL und haben keinen Mode-Präfix)
        plot_defs = [
            ("OverlayHistogramm",          "Histogram"),
            ("OverlayGaussFits",           "Gaussian fit"),
            ("OverlayWeibullFits",         "Weibull fit"),
            ("Boxplot",                    "Box plot"),
            ("QQPlot",                     "Q–Q plot"),
            ("DUAL_GroupedBar_Mean_Std",   "Means & Std (WITH vs INLIER)"),
        ]

        with PdfPages(pdf_path) as pdf:
            for i in part_ids:
                fid = f"Part_{i}"
                fig, axs = plt.subplots(2, 3, figsize=(24, 12))  # 2×3 Raster

                for idx, (suffix, title) in enumerate(plot_defs):
                    r, c = divmod(idx, 3)
                    ax = axs[r, c]

                    # Dateiname: DUAL ohne Modus, sonst mit Modus
                    if suffix == "DUAL_GroupedBar_Mean_Std":
                        png = os.path.join(outdir, f"{fid}_DUAL_GroupedBar_Mean_Std.png")
                    else:
                        png = os.path.join(outdir, f"{fid}_{mode}_{suffix}.png")

                    if os.path.exists(png):
                        img = mpimg.imread(png)
                        ax.imshow(img)
                        ax.axis("off")
                        ax.set_title(f"{title} – {subtitle}", fontsize=12)
                    else:
                        ax.axis("off")
                        ax.set_title(f"{title} – {subtitle}\n(missing)", fontsize=12)

                plt.suptitle(f"{fid}", fontsize=20)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

        logger.info("[Report] PDF created: %s", pdf_path)
        return pdf_path

    

    # ------- Loader & Helpers --------------------------------

    @staticmethod
    def _resolve(fid: str, filename: str) -> str:
        """
        Unterstützt sowohl '<fid>/<file>' als auch 'data/<fid>/<file>'.
        """
        p1 = os.path.join(fid, filename)
        if os.path.exists(p1):
            return p1
        return os.path.join("data","Multi-illumination", "Job_0378_8400-110", "1-3_2-3", fid, filename)

    @staticmethod
    def _load_1col_distances(path: str) -> np.ndarray:
        """Lädt 1-Spalten Distanzdatei ohne Header."""
        arr = np.loadtxt(path, ndmin=2)        # shape (N,1)
        vals = arr[:, 0].astype(float)
        return vals[np.isfinite(vals)]

    @staticmethod
    def _load_coordinates_inlier_distances(path: str) -> np.ndarray:
        """Lädt 4-Spalten coordinates_inlier_* mit Header; nimmt letzte Spalte als Distanz."""
        # Header vorhanden -> skiprows=1
        arr = np.loadtxt(path, ndmin=2, skiprows=1)  # shape (N,4) erwartet
        if arr.shape[1] < 4:
            raise ValueError(f"Erwarte 4 Spalten (x y z distance) in: {path}")
        vals = arr[:, -1].astype(float)
        return vals[np.isfinite(vals)]


    @classmethod
    def _load_data(cls, fid: str, filenames: List[str], versions: List[str]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Tuple[float, float]]
    ]:
        """
        Rückwärts-kompatibel: liefert weiterhin 'WITH' (inkl. Outlier).
        Für INLIER wird separat in overlay_plots geladen (siehe unten).
        """
        data_with: Dict[str, np.ndarray] = {}
        gauss_with: Dict[str, Tuple[float, float]] = {}

        for v in versions:
            # Deine Dateinamen-Patterns:
            base_with = f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances.txt"
            path_with = cls._resolve(fid, base_with)
            logger.info(f"[Report] Lade WITH: {path_with}")

            if not os.path.exists(path_with):
                logger.warning(f"[Report] Datei fehlt (WITH): {path_with}")
                continue

            try:
                # CC könnte auch Semikolon-CSV sein; deine Angabe oben sagt 1 Spalte ohne Header.
                if v.lower() == "cc":
                    # Fallback: zuerst versuchen wir einfach als 1-Spalten-Text:
                    try:
                        arr = cls._load_1col_distances(path_with)
                    except Exception:
                        # Falls es doch CSV ist:
                        df = pd.read_csv(path_with, sep=";")
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        if len(num_cols) == 0:
                            raise ValueError("Keine numerische Spalte gefunden (CC).")
                        arr = df[num_cols[0]].astype(float).to_numpy()
                        arr = arr[np.isfinite(arr)]
                else:
                    arr = cls._load_1col_distances(path_with)
            except Exception as e:
                logger.error(f"[Report] Laden fehlgeschlagen (WITH: {path_with}): {e}")
                continue

            if arr.size:
                label = f"{v}_{fid}"
                data_with[label] = arr
                mu, std = norm.fit(arr)
                gauss_with[label] = (float(mu), float(std))

        return data_with, gauss_with


    @staticmethod
    def _get_common_range(data: Dict[str, np.ndarray]) -> Tuple[float, float, np.ndarray]:
        all_vals = np.concatenate(list(data.values())) if data else np.array([])
        data_min, data_max = (float(np.min(all_vals)), float(np.max(all_vals))) if all_vals.size else (0.0, 1.0)
        x = np.linspace(data_min, data_max, 500)
        return data_min, data_max, x

    # ------- Einzelplots -------------------------------------

    @staticmethod
    def _plot_overlay_histogram(fid, fname, data, bins, data_min, data_max, colors, outdir, title_text=None, labels_order=None):
        plt.figure(figsize=(10, 6))
        labels = labels_order or list(data.keys())
        for v in labels:
            arr = data[v]
            plt.hist(arr, bins=bins, range=(data_min, data_max), density=True,
                    histtype="step", linewidth=2, label=v, color=colors.get(v))
        plt.title(title_text or f"Histogramm – {title_text}")
        plt.xlabel("M3C2 distance")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayHistogramm.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_gauss(fid, fname, data, gauss_params, x, colors, outdir, title_text=None, labels_order=None):
        plt.figure(figsize=(10, 6))
        labels = labels_order or list(data.keys())
        for v in labels:
            if v in gauss_params:
                mu, std = gauss_params[v]
                plt.plot(x, norm.pdf(x, mu, std), color=colors.get(v),
                        linestyle="--" if v.lower() != "cc" else "-", linewidth=2,
                        label=rf"{v} Gauss ($\mu$={mu:.4f}, $\sigma$={std:.4f})")
        plt.title(title_text or f"Overlay Gauss-Fits – {fid}/{fname}")
        plt.xlabel("M3C2 distance")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayGaussFits.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_weibull(fid, fname, data, x, colors, outdir, title_text=None, labels_order=None):
        weibull_params = {}
        for v, arr in data.items():
            try:
                a, loc, b = weibull_min.fit(arr)
                weibull_params[v] = (float(a), float(loc), float(b))
            except Exception as e:
                logger.warning(f"[Report] Weibull-Fit fehlgeschlagen ({fid}/{fname}, {v}): {e}")

        plt.figure(figsize=(10, 6))
        labels = labels_order or list(weibull_params.keys())
        for v in labels:
            if v in weibull_params:
                a, loc, b = weibull_params[v]
                plt.plot(x, weibull_min.pdf(x, a, loc=loc, scale=b),
                        color=colors.get(v), linestyle="--" if v.lower() != "cc" else "-",
                        linewidth=2, label=rf"{v} Weibull (a={a:.2f}, b={b:.4f})")
        plt.title(title_text or f"Overlay Weibull-Fits – {fid}/{fname}")
        plt.xlabel("M3C2 distance")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayWeibullFits.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_boxplot(fid, fname, data, colors, outdir, title_text=None, labels_order=None):
        try:
            import seaborn as sns
            records = [pd.DataFrame({"Version": v, "Distanz": arr}) for v, arr in data.items()]
            if not records:
                return
            df = pd.concat(records, ignore_index=True)
            order = labels_order or list(df["Version"].unique())
            palette = {v: colors.get(v) for v in order}
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="Version", y="Distanz", palette=palette, legend=False, order=order)
            plt.title(title_text or f"Boxplot – {fid}/{fname}")
            plt.xlabel("Version")
            plt.ylabel("M3C2 distance")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Boxplot.png"))
            plt.close()
        except Exception:
            labels = labels_order or list(data.keys())
            arrs = [data[v] for v in labels]
            plt.figure(figsize=(10, 6))
            b = plt.boxplot(arrs, labels=labels, patch_artist=True)
            for patch, v in zip(b["boxes"], labels):
                c = colors.get(v, "#aaaaaa")
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            plt.title(title_text or f"Boxplot – {fid}/{fname}")
            plt.xlabel("Version")
            plt.ylabel("M3C2 distance")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Boxplot.png"))
            plt.close()

    @staticmethod
    def _plot_overlay_qq(fid, fname, data, colors, outdir, title_text=None, labels_order=None):
        plt.figure(figsize=(10, 6))
        labels = labels_order or list(data.keys())
        for v in labels:
            arr = data[v]
            (osm, osr), (slope, intercept, r) = probplot(arr, dist="norm")
            plt.plot(osm, osr, marker="o", linestyle="", label=v, color=colors.get(v))
            plt.plot(osm, slope * osm + intercept, color=colors.get(v), linestyle="--", alpha=0.7)
        plt.title(title_text or f"Q-Q-Plot – {fid}/{fname}")
        plt.xlabel("Theoretical quantiles")
        plt.ylabel("Ordered values")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_QQPlot.png"))
        plt.close()

    @staticmethod
    def _plot_grouped_bar_means_stds_dual(
        fid: str,
        fname: str,
        data_with: Dict[str, np.ndarray],
        data_inlier: Dict[str, np.ndarray],
        colors: Dict[str, str],
        outdir: str,
    ) -> None:
        """
        Pro FOLDER (aus keys wie 'python_1-1') nebeneinander:
        - Balken 'WITH' (mit Outlier)
        - Balken 'INLIER' (ohne Outlier)
        Aggregation: concat über alle Versionen eines Folders.
        """
        # Hilfsfunktion: FOLDER-ID aus Label "version_folder"
        def _folder_of(label: str) -> str:
            # label ist "version_fid" -> wir wollen die komplette fid, auch wenn sie Unterstriche enthält
            return label.split("_", 1)[1] if "_" in label else label

        # Ordne Werte je Folder (concat über Versionen)
        folder_to_with: Dict[str, np.ndarray] = {}
        folder_to_inl: Dict[str, np.ndarray] = {}

        for k, arr in data_with.items():
            f = _folder_of(k)
            folder_to_with.setdefault(f, [])
            folder_to_with[f].append(arr)
        for k, arr in data_inlier.items():
            f = _folder_of(k)
            folder_to_inl.setdefault(f, [])
            folder_to_inl[f].append(arr)

        # Einheitliche Folder-Reihenfolge
        all_folders = sorted(set(folder_to_with.keys()) | set(folder_to_inl.keys()))

        # Kennzahlen je Folder
        means_with, means_inl, stds_with, stds_inl, xlabels, bar_colors = [], [], [], [], [], []
        for f in all_folders:
            arr_with = np.concatenate(folder_to_with.get(f, [])) if f in folder_to_with else np.array([])
            arr_inl  = np.concatenate(folder_to_inl.get(f, [])) if f in folder_to_inl  else np.array([])

            mean_w_signed = float(np.mean(arr_with)) if arr_with.size else np.nan
            std_w         = float(np.std(arr_with))  if arr_with.size else np.nan
            mean_i_signed = float(np.mean(arr_inl))  if arr_inl.size  else np.nan
            std_i         = float(np.std(arr_inl))   if arr_inl.size  else np.nan

            xlabels.append(f)
            mean_w = float(np.abs(mean_w_signed)) if np.isfinite(mean_w_signed) else np.nan
            mean_i = float(np.abs(mean_i_signed)) if np.isfinite(mean_i_signed) else np.nan
            
            means_with.append(mean_w); stds_with.append(std_w)
            means_inl.append(mean_i);  stds_inl.append(std_i)

            # Farbe pro Folder (nimm erste passende Serie, sonst Default)
            # Versuche label "{irgendeine_version}_{folder}" zu finden:
            candidate_label = next((k for k in data_with.keys() if k.endswith("_" + f)), None)
            c = colors.get(candidate_label, "#8aa2ff")
            bar_colors.append(c)

        x = np.arange(len(all_folders))
        width = 0.4

        fig, ax = plt.subplots(2, 1, figsize=(max(10, len(all_folders) * 1.8), 8), sharex=True)

        # Mittelwerte
        ax[0].bar(x - width/2, means_with, width, label="mit Outlier (WITH)", color=bar_colors)
        ax[0].bar(x + width/2, means_inl,  width, label="ohne Outlier (INLIER)", color=bar_colors, alpha=0.55)
        ax[0].set_ylabel("Mittelwert (|μ|)")      # optional klarstellen
        ax[0].set_title(f"Mittelwert je Folder – {fid}/{fname}")
        ax[0].set_ylim(bottom=0)                  # NEU: nie unter 0
        ax[0].legend()

        # Standardabweichungen
        ax[1].bar(x - width/2, stds_with, width, label="mit Outlier (WITH)", color=bar_colors)
        ax[1].bar(x + width/2, stds_inl,  width, label="ohne Outlier (INLIER)", color=bar_colors, alpha=0.55)
        ax[1].set_ylabel("Standardabweichung (σ)")
        ax[1].set_title(f"Standardabweichung je Folder – {fid}/{fname}")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(xlabels, rotation=30, ha="right")
        ax[1].set_ylim(bottom=0)                  # NEU: nie unter 0
        ax[1].legend()

        plt.tight_layout()
        out = os.path.join(outdir, f"{fid}_{fname}_GroupedBar_Mean_Std.png")
        plt.savefig(out)
        plt.close()
        logger.info(f"[Report] Plot gespeichert: {out}")


    @staticmethod
    def _plot_grouped_bar_means_stds_dual_by_case(
        fid, data_with, data_inlier, colors, outdir, title_text="Means & Std – Incl. vs. Excl. Outliers", labels_order=None
    ):
        labels = labels_order or list(dict.fromkeys(list(data_with.keys()) + list(data_inlier.keys())))

        means_with, stds_with, means_inl, stds_inl, bar_colors = [], [], [], [], []
        for lbl in labels:
            arr_w = data_with.get(lbl, np.array([]))
            arr_i = data_inlier.get(lbl, np.array([]))

            m_w = float(np.abs(np.mean(arr_w))) if arr_w.size else np.nan
            s_w = float(np.std(arr_w))          if arr_w.size else np.nan
            m_i = float(np.abs(np.mean(arr_i))) if arr_i.size else np.nan
            s_i = float(np.std(arr_i))          if arr_i.size else np.nan

            means_with.append(m_w); stds_with.append(s_w)
            means_inl.append(m_i);  stds_inl.append(s_i)
            bar_colors.append(colors.get(lbl, "#8aa2ff"))

        x = np.arange(len(labels))
        width = 0.4
        fig, ax = plt.subplots(2, 1, figsize=(max(10, len(labels)*1.8), 8), sharex=True)

        # Means
        ax[0].bar(x - width/2, means_with, width, label="incl. outliers",  color=bar_colors)
        ax[0].bar(x + width/2, means_inl,  width, label="excl. outliers", color=bar_colors, alpha=0.55)
        ax[0].set_ylabel("|μ|")
        ax[0].set_title(f"{title_text} – {fid}")
        ax[0].set_ylim(bottom=0)
        ax[0].legend()

        # Standard deviations
        ax[1].bar(x - width/2, stds_with, width, label="incl. outliers",  color=bar_colors)
        ax[1].bar(x + width/2, stds_inl,  width, label="excl. outliers", color=bar_colors, alpha=0.55)
        ax[1].set_ylabel("σ")
        ax[1].set_title(f"Std. deviation – {fid}")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels, rotation=30, ha="right")
        ax[1].set_ylim(bottom=0)
        ax[1].legend()

        plt.tight_layout()
        out = os.path.join(outdir, f"{fid}_DUAL_GroupedBar_Mean_Std.png")
        plt.savefig(out)
        plt.close()
        logger.info("[Report] Saved grouped bar: %s", out)

    @staticmethod
    def _plot_overlay_violin(fid: str, fname: str, data: Dict[str, np.ndarray],
                             colors: Dict[str, str], outdir: str) -> None:
        try:
            import seaborn as sns
            records = [pd.DataFrame({"Version": v, "Distanz": arr}) for v, arr in data.items()]
            if not records:
                return
            df = pd.concat(records, ignore_index=True)
            palette = {v: colors.get(v) for v in df["Version"].unique()}

            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df, x="Version", y="Distanz", palette=palette, cut=0, inner="quartile")
            plt.title(f"Violinplot – {fid}/{fname}")
            plt.xlabel("Version")
            plt.ylabel("M3C2 distance")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Violinplot.png"))
            plt.close()
        except Exception as e:
            logger.warning(f"[Report] Violinplot fehlgeschlagen ({fid}/{fname}): {e}")
