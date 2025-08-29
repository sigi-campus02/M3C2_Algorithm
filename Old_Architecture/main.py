import os, re, logging
from orchestration.batch_orchestrator import BatchOrchestrator
from config.pipeline_config import PipelineConfig
from log_utils.logging_utils import setup_logging

# --- Settings ---
base_data_dir = "data"
folder_ids = ["Multi-Illumination"]
mov_as_corepoints = True
use_subsampled_corepoints = 1
sample_size = 10000
only_stats = False
stats_singleordistance = "distance"
output_format = "excel"
project = "MARS_Multi_Illumination"
normal_override = 0.002
proj_override = 0.004
use_existing_params = False
outlier_detection_method = "rmse"
outlier_multiplicator = 3

# <<< NEU: nur bestimmte Indizes zulassen (hier: nur Part 1) >>>
allowed_indices = {1}
# Tipp: für alle laufen lassen -> allowed_indices = None oder set()
# ----------------

log = logging.getLogger(__name__)

def parse_files(dirpath: str):
    import os, re
    re_pat = re.compile(r'^(?P<grp>[ab])-(?P<idx>\d+)(?P<ai>-AI)?$')

    a_plain, a_ai, b_plain, b_ai = {}, {}, {}, {}

    with os.scandir(dirpath) as it:
        for entry in it:
            if not entry.is_file():
                continue
            stem, ext = os.path.splitext(entry.name)  # z.B. ("a-1", ".ply")
            m = re_pat.match(stem)
            if not m:
                continue
            grp = m.group('grp')
            idx = int(m.group('idx'))
            ai  = bool(m.group('ai'))
            if grp == 'a':
                (a_ai if ai else a_plain)[idx] = stem
            else:
                (b_ai if ai else b_plain)[idx] = stem

    return a_plain, a_ai, b_plain, b_ai


def _apply_index_filter(idxs: set[int]) -> list[int]:
    """Hilfsfunktion: optional nach allowed_indices filtern und sortieren."""
    if isinstance(allowed_indices, set) and len(allowed_indices) > 0:
        idxs = idxs & allowed_indices
    return sorted(idxs)

def main() -> None:
    cfgs = []

    for fid in folder_ids:
        folder = os.path.join(base_data_dir, fid)
        a_plain, a_ai, b_plain, b_ai = parse_files(folder)

        # --- Übersicht / Logging ---
        log.info("[Scan] %s", folder)
        log.info("  a-*  plain: %d | AI: %d | indices=%s / %s",
                 len(a_plain), len(a_ai),
                 sorted(a_plain.keys()), sorted(a_ai.keys()))
        log.info("  b-*  plain: %d | AI: %d | indices=%s / %s",
                 len(b_plain), len(b_ai),
                 sorted(b_plain.keys()), sorted(b_ai.keys()))

        # Case 1: a-i vs b-i
        case1_idx = _apply_index_filter(set(a_plain) & set(b_plain))
        log.info("  Case1 (a-i vs b-i): %d Runs, indices=%s", len(case1_idx), case1_idx)
        for i in case1_idx:
            cfgs.append(PipelineConfig(
                folder, a_plain[i], b_plain[i],
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        # Case 2: a-i vs b-i-AI
        case2_idx = _apply_index_filter(set(a_plain) & set(b_ai))
        log.info("  Case2 (a-i vs b-i-AI): %d Runs, indices=%s", len(case2_idx), case2_idx)
        for i in case2_idx:
            cfgs.append(PipelineConfig(
                folder, a_plain[i], b_ai[i],
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        # Case 3: a-i-AI vs b-i
        case3_idx = _apply_index_filter(set(a_ai) & set(b_plain))
        log.info("  Case3 (a-i-AI vs b-i): %d Runs, indices=%s", len(case3_idx), case3_idx)
        for i in case3_idx:
            cfgs.append(PipelineConfig(
                folder, a_ai[i], b_plain[i],
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        # Case 4: a-i-AI vs b-i-AI
        case4_idx = _apply_index_filter(set(a_ai) & set(b_ai))
        log.info("  Case4 (a-i-AI vs b-i-AI): %d Runs, indices=%s", len(case4_idx), case4_idx)
        for i in case4_idx:
            cfgs.append(PipelineConfig(
                folder, a_ai[i], b_ai[i],
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        log.info("  => Total Runs so far (inkl. %s): %d", fid, len(cfgs))

    orchestrator = BatchOrchestrator(cfgs, sample_size, output_format)
    orchestrator.run_all()

if __name__ == "__main__":
    setup_logging()
    main()
