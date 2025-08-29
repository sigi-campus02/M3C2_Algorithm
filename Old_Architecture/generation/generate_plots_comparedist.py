import logging
logger = logging.getLogger(__name__)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.plot_service_comparedistances import PlotServiceCompareDistances, PlotConfig, PlotOptionsComparedistances

# folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]
# ref_variants = ["ref", "ref_ai"]

folder_ids = ["0342-0349"]
ref_variants = ["ref", "ref_ai"]

cfg = PlotConfig(
    folder_ids=folder_ids,
    filenames=ref_variants,
    bins=256,
    outdir="outputs",
    project="MARS"
)

opts = PlotOptionsComparedistances(
    plot_blandaltman=True,
    plot_passingbablok=True,
    plot_linearregression=True
)

logging.info(f"Starting plot generation {cfg}, {opts}")
PlotServiceCompareDistances.overlay_plots(cfg, opts)




