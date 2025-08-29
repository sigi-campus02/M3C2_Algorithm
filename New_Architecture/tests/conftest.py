# New_Architecture/tests/conftest.py
"""Pytest configuration file - fixes import paths"""

import sys
import os
from pathlib import Path

# Add the New_Architecture directory to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Also add parent directory if needed for Old_Architecture imports
PARENT_DIR = ROOT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

print(f"Python path configured for tests:")
print(f"  - Root: {ROOT_DIR}")
print(f"  - Parent: {PARENT_DIR}")


# New_Architecture/tests/__init__.py
"""Test package initialization"""


# New_Architecture/tests/test_repositories.py
"""Tests für Repository Implementierungen - FIXED IMPORTS"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile
import os

# Correct imports - no need for 'New_Architecture' prefix when running from within New_Architecture
from infrastructure.repositories.enhanced_point_cloud_repository import EnhancedPointCloudRepository
from infrastructure.repositories.distance_repository import DistanceRepository
from infrastructure.repositories.file_point_cloud_repository import (
    FilePointCloudRepository,
    FileParameterRepository,
    FileStatisticsRepository
)
from domain.entities import (
    PointCloud,
    CloudPair,
    M3C2Parameters,
    Statistics
)


class TestEnhancedPointCloudRepository(unittest.TestCase):
    """Tests für Enhanced Point Cloud Repository"""
    
    def setUp(self):
        """Setup für jeden Test"""
        self.temp_dir = tempfile.mkdtemp()
        self.repo = EnhancedPointCloudRepository(self.temp_dir)
    
    def tearDown(self):
        """Cleanup nach jedem Test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_xyz_format(self):
        """Test XYZ Format Loading"""
        # Create test XYZ file
        xyz_path = Path(self.temp_dir) / "test.xyz"
        xyz_content = """1.0 2.0 3.0
        4.0 5.0 6.0
        7.0 8.0 9.0"""
        xyz_path.write_text(xyz_content)
        
        # Load cloud
        cloud = self.repo.load_point_cloud("test.xyz")
        
        # Assertions
        self.assertIsInstance(cloud, PointCloud)
        self.assertEqual(cloud.cloud.shape, (3, 3))
        np.testing.assert_array_equal(cloud.cloud[0], [1.0, 2.0, 3.0])
    
    def test_load_ply_format(self):
        """Test PLY Format Loading"""
        # Create minimal PLY file
        ply_path = Path(self.temp_dir) / "test.ply"
        ply_content = """ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
end_header
1.0 2.0 3.0
4.0 5.0 6.0
"""
        ply_path.write_text(ply_content)
        
        # Mock plyfile since it might not be installed
        with patch('infrastructure.repositories.enhanced_point_cloud_repository.plyfile') as mock_ply:
            mock_plydata = MagicMock()
            mock_vertex = MagicMock()
            mock_vertex.data = np.array(
                [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            )
            mock_plydata.elements = [mock_vertex]
            mock_ply.PlyData.read.return_value = mock_plydata
            
            cloud = self.repo.load_point_cloud("test.ply")
            
            self.assertIsInstance(cloud, PointCloud)
    
    def test_detect_format(self):
        """Test Format Detection"""
        # Test various extensions
        test_cases = [
            ("file.xyz", "xyz"),
            ("file.ply", "ply"),
            ("file.las", "las"),
            ("file.laz", "las"),
            ("file.obj", "obj"),
            ("file.txt", "xyz"),  # Default
        ]
        
        for filename, expected_format in test_cases:
            detected = self.repo._detect_format(filename)
            self.assertEqual(detected, expected_format, f"Failed for {filename}")
    
    def test_batch_load(self):
        """Test Batch Loading"""
        # Create multiple test files
        for i in range(3):
            xyz_path = Path(self.temp_dir) / f"test_{i}.xyz"
            xyz_path.write_text(f"{i} {i} {i}")
        
        # Batch load
        files = ["test_0.xyz", "test_1.xyz", "test_2.xyz"]
        clouds = self.repo.batch_load(files)
        
        # Assertions
        self.assertEqual(len(clouds), 3)
        for cloud in clouds:
            self.assertIsInstance(cloud, PointCloud)


class TestDistanceRepository(unittest.TestCase):
    """Tests für Distance Repository"""
    
    def setUp(self):
        """Setup für jeden Test"""
        self.temp_dir = tempfile.mkdtemp()
        self.repo = DistanceRepository(self.temp_dir)
    
    def tearDown(self):
        """Cleanup nach jedem Test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_distances(self):
        """Test Save and Load Distances"""
        # Create test data
        distances = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        filepath = "test_distances.txt"
        
        # Save
        self.repo.save_distances(distances, filepath)
        
        # Load
        loaded = self.repo.load_distances(filepath)
        
        # Assertions
        np.testing.assert_array_equal(distances, loaded)
    
    def test_save_with_coordinates(self):
        """Test Save with Coordinates"""
        # Create test data
        distances = np.array([1.0, 2.0, 3.0])
        coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        filepath = "test_with_coords.txt"
        
        # Save
        self.repo.save_with_coordinates(distances, coordinates, filepath)
        
        # Check file exists
        full_path = Path(self.temp_dir) / filepath
        self.assertTrue(full_path.exists())
        
        # Load and verify format
        content = full_path.read_text()
        lines = content.strip().split('\n')
        self.assertEqual(len(lines), 3)
        
        # Check first line format
        values = lines[0].split()
        self.assertEqual(len(values), 4)  # x, y, z, distance
    
    def test_split_outliers_inliers(self):
        """Test Split Outliers and Inliers"""
        # Create test data
        distances = np.array([1.0, 10.0, 2.0, 20.0, 3.0])
        outlier_mask = np.array([False, True, False, True, False])
        tag = "test"
        
        # Split
        outlier_file, inlier_file = self.repo.split_outliers_inliers(
            distances, outlier_mask, tag
        )
        
        # Load and verify
        outliers = self.repo.load_distances(outlier_file)
        inliers = self.repo.load_distances(inlier_file)
        
        # Check lengths
        self.assertEqual(len(outliers), 2)  # 2 outliers
        self.assertEqual(len(inliers), 3)   # 3 inliers
        
        # Check values
        np.testing.assert_array_equal(outliers, [10.0, 20.0])
        np.testing.assert_array_equal(inliers, [1.0, 2.0, 3.0])


class TestFileParameterRepository(unittest.TestCase):
    """Tests für Parameter Repository"""
    
    def setUp(self):
        """Setup für jeden Test"""
        self.temp_dir = tempfile.mkdtemp()
        self.repo = FileParameterRepository(self.temp_dir)
    
    def tearDown(self):
        """Cleanup nach jedem Test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_parameters(self):
        """Test Save and Load Parameters"""
        # Create test parameters
        params = M3C2Parameters(
            normal_scale=0.5,
            search_scale=1.0,
            max_depth=2.0
        )
        
        cloud_pair = CloudPair(
            moving_cloud="mov.ply",
            reference_cloud="ref.ply",
            folder_id="test_folder",
            index=1
        )
        
        # Save
        self.repo.save(params, cloud_pair)
        
        # Load
        loaded = self.repo.load(cloud_pair)
        
        # Assertions
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.normal_scale, 0.5)
        self.assertEqual(loaded.search_scale, 1.0)
        self.assertEqual(loaded.max_depth, 2.0)
    
    def test_exists(self):
        """Test Parameter Exists Check"""
        cloud_pair = CloudPair(
            moving_cloud="mov.ply",
            reference_cloud="ref.ply",
            folder_id="test_folder",
            index=1
        )
        
        # Should not exist initially
        self.assertFalse(self.repo.exists(cloud_pair))
        
        # Save parameters
        params = M3C2Parameters(0.5, 1.0)
        self.repo.save(params, cloud_pair)
        
        # Should exist now
        self.assertTrue(self.repo.exists(cloud_pair))


class TestFileStatisticsRepository(unittest.TestCase):
    """Tests für Statistics Repository"""
    
    def setUp(self):
        """Setup für jeden Test"""
        self.temp_dir = tempfile.mkdtemp()
        self.repo = FileStatisticsRepository(self.temp_dir)
    
    def tearDown(self):
        """Cleanup nach jedem Test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_statistics(self):
        """Test Save and Load Statistics"""
        # Create test statistics
        cloud_pair = CloudPair(
            moving_cloud="mov.ply",
            reference_cloud="ref.ply",
            folder_id="test_folder",
            index=1
        )
        
        stats = Statistics(
            cloud_pair=cloud_pair,
            with_outliers={'mean': 1.0, 'std': 0.5},
            inliers_only={'mean': 0.8, 'std': 0.3},
            outlier_count=10,
            total_count=100
        )
        
        # Save
        self.repo.save(stats, cloud_pair)
        
        # Load
        loaded = self.repo.load(cloud_pair)
        
        # Assertions
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.outlier_count, 10)
        self.assertEqual(loaded.total_count, 100)
        self.assertEqual(loaded.with_outliers['mean'], 1.0)


if __name__ == '__main__':
    unittest.main()

