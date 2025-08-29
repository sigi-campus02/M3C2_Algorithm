from __future__ import annotations
import logging
import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from config.plot_config import PlotOptionsComparedistances, PlotConfig

logger = logging.getLogger(__name__)


class PlotServiceCompareDistances:
    @classmethod
    def overlay_plots(cls, config: PlotConfig, options: PlotOptionsComparedistances) -> None:
        os.makedirs(config.path, exist_ok=True)
        folder_ids = config.folder_ids
        ref_variants = config.filenames

        if options.plot_blandaltman:
            logging.info("Generating Bland-Altman plots...")
            cls._bland_altman_plot(folder_ids, ref_variants, outdir=config.path)
        if options.plot_passingbablok:
            logging.info("Generating Passing-Bablok plots...")
            cls._passing_bablok_plot(folder_ids, ref_variants, outdir=config.path)
        if options.plot_linearregression:
            logging.info("Generating Linear Regression plots...")
            cls._linear_regression_plot(folder_ids, ref_variants, outdir=config.path)

    @staticmethod
    def _resolve(fid: str, filename: str) -> str:
        """Return the path to *filename* for the given folder ID.

        The helper searches first in ``<fid>/`` and then in ``data/<fid>/``
        to mirror the behaviour of other services in this repository.
        """

        p1 = os.path.join(fid, filename)
        if os.path.exists(p1):
            return p1
        return os.path.join("data", fid, filename)

    @staticmethod
    def _load_ref_variant_data(fid: str, variant: str) -> np.ndarray | None:
        basename = f"python_{variant}_m3c2_distances.txt"
        path = PlotServiceCompareDistances._resolve(fid, basename)
        print("Current working directory:", os.getcwd())
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        try:
            return np.loadtxt(path)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    @staticmethod
    def _load_and_mask(fid: str, ref_variants: List[str]) -> tuple[np.ndarray, np.ndarray] | None:
        """Loads and masks the two reference variant arrays for a folder."""
        data = [PlotServiceCompareDistances._load_ref_variant_data(fid, v) for v in ref_variants]
        if any(d is None for d in data):
            return None
        a_raw, b_raw = data
        mask = ~np.isnan(a_raw) & ~np.isnan(b_raw)
        a = np.asarray(a_raw[mask], dtype=float)
        b = np.asarray(b_raw[mask], dtype=float)
        if a.size == 0 or b.size == 0:
            logger.warning(f"Empty values in {fid}, skipped")
            return None
        return a, b

    @classmethod
    def _bland_altman_plot(
        cls,
        folder_ids: List[str],
        ref_variants: List[str],
        outdir: str = "BlandAltman",
    ) -> None:

        if len(ref_variants) != 2:
            raise ValueError("ref_variants must contain exactly two entries")

        os.makedirs(outdir, exist_ok=True)

        for fid in folder_ids:
            result = cls._load_and_mask(fid, ref_variants)
            if result is None:
                continue
            a, b = result

            if a.size == 0 or b.size == 0:
                print(f"[BlandAltman] Leere Distanzwerte in {fid}, übersprungen")
                continue

            # Bland–Altman calculations
            mean_vals = (a + b) / 2.0
            diff_vals = a - b
            mean_diff = float(np.mean(diff_vals))
            std_diff = float(np.std(diff_vals, ddof=1))
            upper = mean_diff + 1.96 * std_diff
            lower = mean_diff - 1.96 * std_diff

            logger.info(
                f"[BlandAltman] {fid}: mean_diff={mean_diff:.6f}, std_diff={std_diff:.6f}, "
                f"upper={upper:.6f}, lower={lower:.6f}, n={a.size} -> {outdir}"
            )

            # Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(mean_vals, diff_vals, alpha=0.3)
            plt.axhline(mean_diff, color="red", linestyle="--",
                        label=f"Mean diff {mean_diff:.4f}")
            plt.axhline(upper, color="green", linestyle="--",
                        label=f"+1.96 SD {upper:.4f}")
            plt.axhline(lower, color="green", linestyle="--",
                        label=f"-1.96 SD {lower:.4f}")
            plt.xlabel("Mean of measurements")
            plt.ylabel("Difference")
            plt.title(
                f"Bland-Altman {fid}: {ref_variants[0]} vs {ref_variants[1]}"
            )
            plt.legend()
            outpath = os.path.join(
                outdir,
                f"bland_altman_{fid}_{ref_variants[0]}_vs_{ref_variants[1]}.png",
            )
            plt.tight_layout()
            plt.savefig(outpath, dpi=300)
            plt.close()

    @classmethod
    def _passing_bablok_plot(
        cls,
        folder_ids: List[str],
        ref_variants: List[str],
        outdir: str = "PassingBablok",
    ) -> None:
        """Create Passing–Bablok regression plots (nach Rowannicholls-Tutorial)."""

        if len(ref_variants) != 2:
            raise ValueError("ref_variants must contain exactly two entries")

        os.makedirs(outdir, exist_ok=True)

        for fid in folder_ids:
            logger.info(f"[PassingBablok] Processing folder: {fid}")
            result = cls._load_and_mask(fid, ref_variants)
            if result is None:
                continue
            x, y = result

            # Optionales Downsampling (nur zur Plot-Ästhetik; Regression bleibt robust)
            max_n = 1000
            if x.size > max_n:
                idx = np.random.choice(x.size, size=max_n, replace=False)
                x, y = x[idx], y[idx]

            if x.size < 2:
                logger.warning(f"[PassingBablok] Zu wenige Punkte in {fid} – übersprungen")
                continue

            # ------- Passing–Bablok nach Tutorial -------
            n = int(len(x))  # Anzahl Rohpunkte
            S = []           # Initialise a list of the gradients between each combination of two points

            # Iterate over all combinations of two points
            for i in range(n - 1):
                x_i, y_i = x[i], y[i]
                for j in range(i + 1, n):
                    x_j, y_j = x[j], y[j]

                    # Ignore identical points
                    if (x_i == x_j) and (y_i == y_j):
                        continue

                    # Vertikale Verbindungen -> +/- inf je nach Vorzeichen von Δy
                    if x_i == x_j:
                        S.append(np.inf if (y_i > y_j) else -np.inf)
                        continue

                    # Calculate the gradient between this pair of points
                    g = (y_i - y_j) / (x_i - x_j)

                    # Ignore any gradient equal to -1
                    if g == -1:
                        continue
                    
                    # Add the gradient to the list of gradients
                    S.append(g)

            if not S:
                logger.warning(f"[PassingBablok] Keine gültigen Paare in {fid}")
                continue

            S = np.array(S, dtype=float)

            # Sort the list of gradients in preparation for taking the median
            S.sort()

            # However, as Passing & Bablok point out, the values of these gradients are not independent and so their median 
            # would be a biased estimator of the gradient of the overall line-of-best-fit. As such, 
            # we need to use an offset, K, to calculate a shifted median, b, which can be used as an estimate for the overall gradient. 
            # This offset is defined as the number of gradients that have a value of less than -1:

            N = int(len(S))                # Anzahl Steigungen (gradients)
            K = int((S < -1).sum())        # K is the number of gradients less than -1

            # Calculate the shifted median

            # If N is odd
            if N % 2 != 0:  
                # Convert to an integer and adjust for the fact that Python is 0-indexed
                idx = int((N + 1) / 2 + K) - 1  
                b = float(S[idx])
            
            # If N is even
            else:     
                # Convert to an integer and adjust for the fact that Python is 0-indexed
                idx = int(N / 2 + K) - 1 
                b = float(0.5 * (S[idx] + S[idx + 1]))

            #Using this estimated gradient of the line-of-best-fit, we can plug in the raw data to get the estimated y-intercept of the line-of-best-fit, 
            # a, as follows:

            # y-Achsenabschnitt a (Median der Residuen)
            a = float(np.median(y - b * x))


            # Calculate the Confidence Intervals
            # Usually we are interested in the 95% confidence interval and so could simply use the 
            # well-known fact that this interval width corresponds to about 1.96 (or about 2) standard deviations either side of the mean. 
            # However, let’s calculate this explicitly to make sure we are being accurate:


            # 95%-Konfidenzintervalle für b und a
            from scipy import stats as st
            C = 0.95
            gamma = 1 - C # 0.05
            # Quantile (the cumulative probability; two-tailed)
            q = 1 - (gamma / 2.0) # 0.975
            # Critical z-score, calculated using the percent-point function (aka the
            # quantile function) of the normal distribution
            w = float(st.norm.ppf(q))  # ~1.96


            # Passing & Bablok provide formulas for getting the indexes of the gradients 
            # that correspond to the bounds of the confidence intervals:
                
            # Intermediate values
            C_gamma = w * np.sqrt((n * (n - 1) * (2 * n + 5)) / 18.0)
            M1 = int(np.round((N - C_gamma) / 2.0))
            M2 = int(N - M1 + 1)

            # Get the lower and upper bounds of the confidence interval for the gradient
            # (convert floats to ints and subtract 1 to adjust for Python being 0-indexed)
            b_L = float(S[M1 + K - 1])
            b_U = float(S[M2 + K - 1])

            # Get the lower and upper bounds of the confidence interval for the y-intercept
            # CI für a via b_U / b_L
            a_L = float(np.median(y - b_U * x))
            a_U = float(np.median(y - b_L * x))

            # ------- Plot -------
            # 1) Immer die gleiche Figurgröße
            fig = plt.figure(figsize=(8, 6), constrained_layout=True)
            ax = fig.add_subplot(111)

            ax.scatter(x, y, alpha=0.35, label="Daten", s=12)

            # Identitäts- und Regressionslinien
            (xl, xu), (yl, yu) = _square_limits(x, y, pad=0.05)
            xx = np.array([xl, xu], dtype=float)

            ax.plot(xx, xx, linestyle="--", color="grey", label="y = x")
            ax.plot(xx, a + b*xx, color="red", label=f"PB: y = {a:.4f} + {b:.4f} x")
            ax.plot(xx, (a_U + b_U*xx), linestyle="--", alpha=0.7, label=f"CI oben: y = {a_U:.4f} + {b_U:.4f} x")
            ax.plot(xx, (a_L + b_L*xx), linestyle="--", alpha=0.7, label=f"CI unten: y = {a_L:.4f} + {b_L:.4f} x")
            ax.fill_between(xx, a_L + b_L*xx, a_U + b_U*xx, alpha=0.12)

            # 2) Quadratische Limits anwenden
            ax.set_xlim(xl, xu)
            ax.set_ylim(yl, yu)
            ax.set_aspect("equal", adjustable="box")  # gleiches Seitenverhältnis, volle Fläche

            ax.set_xlabel(ref_variants[0])
            ax.set_ylabel(ref_variants[1])
            ax.set_title(f"Passing–Bablok {fid}: {ref_variants[0]} vs {ref_variants[1]}")
            ax.legend(frameon=False)


            outpath = os.path.join(
                outdir,
                f"passing_bablok_{fid}_{ref_variants[0]}_vs_{ref_variants[1]}.png",
            )
            plt.savefig(outpath, dpi=300)
            plt.close()

            logger.info(
                f"[PassingBablok] {fid}: b={b:.6f} "
                f"[{b_L:.6f},{b_U:.6f}], a={a:.6f} [{a_L:.6f},{a_U:.6f}] -> {outpath}"
            )

    @classmethod
    def _linear_regression_plot(
        cls,
        folder_ids: List[str],
        ref_variants: List[str],
        outdir: str = "LinearRegression",
    ) -> None:
        """
        Erzeugt OLS-Linearregressionsplots (y = a + b x) für alle folder_ids.
        - gleiche Figurgröße wie beim PB-Plot
        - quadratische Achsenlimits mit _square_limits
        - 95%-CIs für a und b (t-Verteilung)
        """

        if len(ref_variants) != 2:
            raise ValueError("ref_variants must contain exactly two entries")

        os.makedirs(outdir, exist_ok=True)

        for fid in folder_ids:
            logger.info(f"[OLS] Processing folder: {fid}")
            result = cls._load_and_mask(fid, ref_variants)
            if result is None:
                continue
            x, y = result

            # Optionales Downsampling (nur fürs Plotten)
            max_n = 1000
            if x.size > max_n:
                idx = np.random.choice(x.size, size=max_n, replace=False)
                x, y = x[idx], y[idx]

            if x.size < 3:  # für OLS-CIs brauchen wir mind. n>=3 (df = n-2)
                logger.warning(f"[OLS] Zu wenige Punkte in {fid} – übersprungen")
                continue

            # -------- OLS-Schätzer + Standardfehler (ohne weitere Abhängigkeiten) --------
            n = x.size
            xbar = float(np.mean(x))
            ybar = float(np.mean(y))
            Sxx = float(np.sum((x - xbar) ** 2))
            if Sxx == 0.0:
                logger.warning(f"[OLS] Sxx=0 (keine Varianz in x) – übersprungen: {fid}")
                continue

            Sxy = float(np.sum((x - xbar) * (y - ybar)))
            b = Sxy / Sxx
            a = ybar - b * xbar

            resid = y - (a + b * x)
            SSE = float(np.sum(resid ** 2))
            s2 = SSE / (n - 2)  # Residualvarianz
            se_b = float(np.sqrt(s2 / Sxx))
            se_a = float(np.sqrt(s2 * (1.0 / n + (xbar ** 2) / Sxx)))

            # 95%-Konfidenzintervalle mit t-Verteilung
            from scipy.stats import t
            tcrit = float(t.ppf(0.975, df=n - 2))
            b_L, b_U = b - tcrit * se_b, b + tcrit * se_b
            a_L, a_U = a - tcrit * se_a, a + tcrit * se_a

            # -------- Plot --------
            fig = plt.figure(figsize=(8, 6), constrained_layout=True)
            ax = fig.add_subplot(111)

            ax.scatter(x, y, alpha=0.35, label="Daten", s=12)

            # Identitäts- und Regressionslinien
            (xl, xu), (yl, yu) = _square_limits(x, y, pad=0.05)
            xx = np.array([xl, xu], dtype=float)

            ax.plot(xx, xx, linestyle="--", color="grey", label="y = x")
            ax.plot(xx, a + b * xx, color="red", label=f"OLS: y = {a:.4f} + {b:.4f} x")

            # Konservative CI-Hüllkurve über getrennte CIs von a & b
            ax.plot(xx, a_U + b_U * xx, linestyle="--", alpha=0.7,
                    label=f"CI oben: y = {a_U:.4f} + {b_U:.4f} x")
            ax.plot(xx, a_L + b_L * xx, linestyle="--", alpha=0.7,
                    label=f"CI unten: y = {a_L:.4f} + {b_L:.4f} x")
            ax.fill_between(xx, a_L + b_L * xx, a_U + b_U * xx, alpha=0.12)

            # Quadratische Limits & Achsen
            ax.set_xlim(xl, xu)
            ax.set_ylim(yl, yu)
            ax.set_aspect("equal", adjustable="box")

            ax.set_xlabel(ref_variants[0])
            ax.set_ylabel(ref_variants[1])
            ax.set_title(f"Linear Regression {fid}: {ref_variants[0]} vs {ref_variants[1]}")
            ax.legend(frameon=False)

            outpath = os.path.join(
                outdir,
                f"linear_regression_{fid}_{ref_variants[0]}_vs_{ref_variants[1]}.png",
            )
            plt.savefig(outpath, dpi=300)
            plt.close()

            logger.info(
                f"[OLS] {fid}: b={b:.6f} [{b_L:.6f},{b_U:.6f}], "
                f"a={a:.6f} [{a_L:.6f},{a_U:.6f}] -> {outpath}"
            )


def _square_limits(x: np.ndarray, y: np.ndarray, pad: float = 0.05):
    """
    Liefert quadratische Achsenlimits, die alle Punkte abdecken
    und den Plotbereich maximal ausnutzen.
    pad = 5% Rand.
    """
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # Gemeinsamer Min/Max-Bereich über beide Achsen
    v_min = min(x_min, y_min)
    v_max = max(x_max, y_max)

    # Quadratischer Bereich um das Zentrum
    cx = cy = (v_min + v_max) / 2.0
    half = max((x_max - x_min), (y_max - y_min)) / 2.0
    half = half * (1.0 + pad) if half > 0 else 1.0  # fallback falls alle Punkte identisch

    return (cx - half, cx + half), (cy - half, cy + half)