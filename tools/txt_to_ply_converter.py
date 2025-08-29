#!/usr/bin/env python3
# New_Architecture/tools/txt_to_ply_converter.py
"""
Standalone-Tool zur Konvertierung von TXT-Dateien mit Distanzen zu PLY-Dateien.
Kompatibel mit CloudCompare durch Scalar Fields.
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Iterable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("txt_to_ply")

# Try to use project's visualization service if available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from application.services.enhanced_visualization_service import EnhancedVisualizationService
    USE_SERVICE = True
    logger.info("Using EnhancedVisualizationService for conversion")
except ImportError:
    USE_SERVICE = False
    logger.info("Using standalone conversion (plyfile required)")

# Fallback implementation if service not available
if not USE_SERVICE:
    try:
        from plyfile import PlyData, PlyElement
        PLYFILE_AVAILABLE = True
    except ImportError:
        PLYFILE_AVAILABLE = False
        logger.error("plyfile package required. Install with: pip install plyfile")


class StandaloneTXTtoPLYConverter:
    """Standalone converter for TXT to PLY with distance coloring"""
    
    PATTERNS = [
        "*_m3c2_distances_coordinates.txt",
        "*_m3c2_distances_coordinates_inlier_*.txt",
        "*_m3c2_distances_coordinates_outlier_*.txt",
        "*_distances_with_coordinates.txt"
    ]
    
    def __init__(self, clip_percentile: float = 98.0):
        self.clip_percentile = clip_percentile
        self.viz_service = EnhancedVisualizationService() if USE_SERVICE else None
    
    def find_txt_files(self, roots: List[str], recursive: bool = True) -> Iterable[Path]:
        """Find all matching TXT files"""
        for root in roots:
            root_path = Path(root)
            if not root_path.exists():
                logger.warning(f"Path not found: {root}")
                continue
            
            for pattern in self.PATTERNS:
                if recursive:
                    yield from root_path.rglob(pattern)
                else:
                    yield from root_path.glob(pattern)
    
    def convert_with_service(
        self,
        txt_path: Path,
        ply_path: Path,
        scalar_name: str = "distance"
    ) -> bool:
        """Convert using the visualization service"""
        try:
            self.viz_service.txt_to_ply_with_distance_color(
                txt_path,
                ply_path,
                scalar_name=scalar_name,
                percentile_range=(100 - self.clip_percentile, self.clip_percentile)
            )
            return True
        except Exception as e:
            logger.error(f"Service conversion failed: {e}")
            return False
    
    def convert_standalone(
        self,
        txt_path: Path,
        ply_path: Path,
        scalar_name: str = "distance"
    ) -> bool:
        """Standalone conversion without service"""
        if not PLYFILE_AVAILABLE:
            logger.error("plyfile not available for standalone conversion")
            return False
        
        try:
            # Load data
            data = np.loadtxt(str(txt_path), comments="#")
            if data.ndim != 2 or data.shape[1] < 4:
                raise ValueError(f"Expected 4 columns (x,y,z,distance), got {data.shape[1]}")
            
            points = data[:, :3]
            distances = data[:, 3]
            
            # Compute colors
            colors = self._compute_colors(distances)
            
            # Create PLY
            dtype = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                (scalar_name, 'f4')
            ]
            
            vertex = np.zeros(len(points), dtype=dtype)
            vertex['x'] = points[:, 0]
            vertex['y'] = points[:, 1]
            vertex['z'] = points[:, 2]
            vertex['red'] = colors[:, 0]
            vertex['green'] = colors[:, 1]
            vertex['blue'] = colors[:, 2]
            vertex[scalar_name] = distances
            
            # Write PLY
            el = PlyElement.describe(vertex, 'vertex')
            ply_path.parent.mkdir(parents=True, exist_ok=True)
            PlyData([el], text=False).write(str(ply_path))
            
            return True
            
        except Exception as e:
            logger.error(f"Standalone conversion failed: {e}")
            return False
    
    def _compute_colors(self, distances: np.ndarray) -> np.ndarray:
        """Compute colors for distances (blue-white-red colormap)"""
        n = len(distances)
        colors = np.zeros((n, 3), dtype=np.uint8)
        
        valid_mask = ~np.isnan(distances)
        if not valid_mask.any():
            colors[:] = [128, 128, 128]  # Gray for all NaN
            return colors
        
        valid_dists = distances[valid_mask]
        
        # Symmetric clipping around 0
        lim = np.percentile(np.abs(valid_dists), self.clip_percentile)
        if lim <= 0:
            lim = 1.0
        
        # Normalize to [-1, 1]
        normed = np.clip(valid_dists / lim, -1, 1)
        
        # Blue-white-red colormap
        # -1 -> blue (0,0,255)
        #  0 -> white (255,255,255)
        # +1 -> red (255,0,0)
        r = np.where(normed >= 0, 255, (1 + normed) * 255)
        g = (1 - np.abs(normed)) * 255
        b = np.where(normed <= 0, 255, (1 - normed) * 255)
        
        colors[valid_mask, 0] = r.clip(0, 255).astype(np.uint8)
        colors[valid_mask, 1] = g.clip(0, 255).astype(np.uint8)
        colors[valid_mask, 2] = b.clip(0, 255).astype(np.uint8)
        
        # NaN -> white
        colors[~valid_mask] = [255, 255, 255]
        
        return colors
    
    def convert_file(
        self,
        txt_path: Path,
        output_dir: Optional[Path] = None,
        overwrite: bool = False
    ) -> Optional[Path]:
        """Convert a single TXT file to PLY"""
        # Determine output path
        if output_dir:
            ply_path = output_dir / txt_path.with_suffix('.ply').name
        else:
            ply_path = txt_path.with_suffix('.ply')
        
        # Check if already exists
        if ply_path.exists() and not overwrite:
            logger.info(f"Skipping (exists): {ply_path}")
            return ply_path
        
        logger.info(f"Converting: {txt_path} -> {ply_path}")
        
        # Try conversion
        if USE_SERVICE:
            success = self.convert_with_service(txt_path, ply_path)
        else:
            success = self.convert_standalone(txt_path, ply_path)
        
        if success:
            logger.info(f"  Success: {ply_path}")
            return ply_path
        else:
            logger.error(f"  Failed: {txt_path}")
            return None
    
    def convert_batch(
        self,
        input_paths: List[str],
        output_dir: Optional[str] = None,
        recursive: bool = True,
        overwrite: bool = False
    ) -> int:
        """Convert multiple files/directories"""
        # Separate files and directories
        txt_files = []
        directories = []
        
        for path in input_paths:
            p = Path(path)
            if p.is_file() and p.suffix == '.txt':
                txt_files.append(p)
            elif p.is_dir():
                directories.append(str(p))
            else:
                logger.warning(f"Ignoring: {path}")
        
        # Find all TXT files
        all_files = list(txt_files)
        if directories:
            all_files.extend(self.find_txt_files(directories, recursive))
        
        # Remove duplicates
        all_files = list(set(all_files))
        
        logger.info(f"Found {len(all_files)} TXT files to process")
        
        # Convert all
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        converted = 0
        for txt_file in all_files:
            result = self.convert_file(txt_file, output_path, overwrite)
            if result:
                converted += 1
        
        return converted


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Convert TXT files with coordinates and distances to colored PLY files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/Multi-Illumination
  %(prog)s file1.txt file2.txt --output converted/
  %(prog)s data/ --recursive --overwrite
  %(prog)s . --clip-percentile 95
        """
    )
    
    parser.add_argument(
        'paths',
        nargs='+',
        help='Input TXT files or directories to search'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for PLY files (default: same as input)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        default=True,
        help='Search directories recursively (default: True)'
    )
    
    parser.add_argument(
        '--no-recursive',
        dest='recursive',
        action='store_false',
        help='Disable recursive search'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing PLY files'
    )
    
    parser.add_argument(
        '--clip-percentile',
        type=float,
        default=98.0,
        help='Percentile for color clipping (default: 98.0)'
    )
    
    parser.add_argument(
        '--scalar-name',
        default='distance',
        help='Name for the scalar field in PLY (default: distance)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create converter
    converter = StandaloneTXTtoPLYConverter(clip_percentile=args.clip_percentile)
    
    # Run conversion
    start_time = Path.cwd()
    converted = converter.convert_batch(
        args.paths,
        args.output,
        args.recursive,
        args.overwrite
    )
    
    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"Conversion complete: {converted} files converted")
    logger.info(f"{'='*60}")
    
    return 0 if converted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())