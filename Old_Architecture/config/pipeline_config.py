# pipeline_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    folder_id: str
    filename_mov: str
    filename_ref: str
    mov_as_corepoints: bool
    use_subsampled_corepoints: int
    only_stats: bool
    stats_singleordistance: str
    project: str
    normal_override: Optional[float] = None
    proj_override: Optional[float] = None
    use_existing_params: bool = False
    outlier_multiplicator: float = 3.0 # usual range 3.0 - 5.0 für RMSE Multiplication for Outlier detection and removal
    outlier_detection_method: str = "rmse"
    process_python_CC: str = "python"

