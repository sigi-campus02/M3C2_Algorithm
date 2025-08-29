# generation/generate_plots.py
import os, sys, logging
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.plot_service import PlotService
from log_utils.logging_utils import setup_logging
from config.plot_config import PlotOptions

DATA_DIR = os.path.join(ROOT, "data", "Multi-Illumination")
OUT_DIR  = os.path.join(ROOT, "outputs", "MARS_Multi_Illumination", "plots")

if __name__ == "__main__":
    setup_logging()

    # only_grouped = PlotOptions(
    #     plot_hist=True, plot_gauss=True, plot_weibull=True,
    #     plot_box=True, plot_qq=True, plot_grouped_bar=True, plot_violin=False,
    # )

    # PlotService.overlay_by_index(
    #     DATA_DIR, OUT_DIR,
    #     versions=("python",),
    #     bins=256,
    #     options=only_grouped,
    #     skip_existing=True,   # vorhandene PNGs nicht überschreiben
    # )

    pdf_incl = PlotService.build_parts_pdf(
        OUT_DIR,
        pdf_path=os.path.join(OUT_DIR, "parts_summary_incl_outliers.pdf"),
        include_with=True,
        include_inlier=False,
    )
    pdf_excl = PlotService.build_parts_pdf(
        OUT_DIR,
        pdf_path=os.path.join(OUT_DIR, "parts_summary_excl_outliers.pdf"),
        include_with=False,
        include_inlier=True,
    )


    print(f"PDF (incl. outliers): {pdf_incl}")
    print(f"PDF (excl. outliers): {pdf_excl}")
    
