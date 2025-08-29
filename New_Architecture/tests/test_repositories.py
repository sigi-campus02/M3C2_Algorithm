# New_Architecture/tests/test_repositories_clean.py
"""Bereinigte Tests für Repository-Implementierungen"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import py4dgeo

from infrastructure.repositories.enhanced_point_cloud_repository import EnhancedPointCloudRepository
from infrastructure.repositories.distance_repository import DistanceRepository
from infrastructure.repositories.file_point_cloud_repository import (
    FilePointCloudRepository,
    FileParameterRepository,
    FileStatisticsRepository
)


class TestEnhancedPointCloudRepository:
    """Tests für EnhancedPointCloudRepository"""
    
    @pytest.fixture
    def temp_dir(self):
        """Erstellt temporäres Verzeichnis für Tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def repository(self, temp_dir):
        """Erstellt Repository-Instanz"""
        return EnhancedPointCloudRepository(temp_dir)
    
    @pytest.fixture
    def sample_points(self):
        """Erstellt Sample-Punktwolke"""
        np.random.seed(42)
        return np.random.randn(1000, 3) * 10
    
    def test_save_load_xyz(self, repository, sample_points):
        """Test XYZ Format Save/Load"""
        file_path = "test_cloud.xyz"
        repository.save_point_cloud(sample_points, file_path)
        
        loaded = repository.load_point_cloud(file_path)
        loaded_points = np.asarray(loaded.cloud)
        
        np.testing.assert_allclose(loaded_points, sample_points, rtol=1e-5)
    
    def test_save_load_ply(self, repository, sample_points):
        """Test PLY Format Save/Load"""
        pytest.importorskip("plyfile")
        
        file_path = "test_cloud.ply"
        repository.save_point_cloud(sample_points, file_path)
        
        loaded = repository.load_point_cloud(file_path)
        loaded_points = np.asarray(loaded.cloud)
        
        np.testing.assert_allclose(loaded_points, sample_points, rtol=1e-4)
    
    def test_format_detection(self, repository, sample_points, temp_dir):
        """Test automatische Format-Erkennung"""
        xyz_path = Path(temp_dir) / "test.xyz"
        np.savetxt(xyz_path, sample_points)
        
        detected = repository.detect_format(xyz_path)
        assert detected == '.xyz'
    
    def test_format_conversion(self, repository, sample_points):
        """Test Format-Konvertierung"""
        xyz_path = "source.xyz"
        repository.save_point_cloud(sample_points, xyz_path)
        
        ply_path = "target.ply"
        repository.convert_format(xyz_path, ply_path)
        
        assert repository.exists(ply_path)
        
        loaded = repository.load_point_cloud(ply_path)
        loaded_points = np.asarray(loaded.cloud)
        np.testing.assert_allclose(loaded_points, sample_points, rtol=1e-4)


class TestDistanceRepository:
    """Tests für DistanceRepository"""
    
    @pytest.fixture
    def temp_dir(self):
        """Erstellt temporäres Verzeichnis"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def repository(self, temp_dir):
        """Erstellt Repository-Instanz"""
        return DistanceRepository(temp_dir)
    
    @pytest.fixture
    def sample_distances(self):
        """Erstellt Sample-Distanzen"""
        np.random.seed(42)
        distances = np.random.randn(500) * 0.01
        distances[::10] = np.nan  # Füge NaN hinzu
        return distances
    
    def test_save_load_distances(self, repository, sample_distances):
        """Test einfache Distanzen Save/Load"""
        file_path = "distances.txt"
        repository.save_distances(sample_distances, file_path)
        
        loaded = repository.load_distances(file_path)
        
        np.testing.assert_array_equal(loaded, sample_distances)
    
    def test_save_load_with_coordinates(self, repository, sample_distances):
        """Test Distanzen mit Koordinaten"""
        coords = np.random.randn(len(sample_distances), 3)
        
        file_path = "distances_coords.txt"
        repository.save_distances_with_coordinates(
            coords, sample_distances, file_path
        )
        
        loaded_coords, loaded_dists = repository.load_distances_with_coordinates(file_path)
        
        np.testing.assert_allclose(loaded_coords, coords, rtol=1e-5)
        np.testing.assert_array_equal(loaded_dists, sample_distances)
    
    def test_split_by_outliers(self, repository, sample_distances):
        """Test Aufteilen nach Outliers"""
        coords = np.random.randn(len(sample_distances), 3)
        outlier_mask = np.abs(sample_distances) > 0.03
        
        outlier_path, inlier_path = repository.split_by_outliers(
            coords, sample_distances, outlier_mask,
            "base_distances.txt", method="test"
        )
        
        assert Path(outlier_path).exists() or np.sum(outlier_mask) == 0
        assert Path(inlier_path).exists() or np.sum(~outlier_mask) == 0


class TestParameterRepository:
    """Tests für Parameter Repository"""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def repository(self, temp_dir):
        return FileParameterRepository(temp_dir)
    
    def test_save_load_params(self, repository):
        """Test Parameter Save/Load"""
        params = {
            'NormalScale': 0.002,
            'SearchScale': 0.004
        }
        
        repository.save_params(params, "params.txt")
        loaded = repository.load_params("params.txt")
        
        assert loaded['NormalScale'] == 0.002
        assert loaded['SearchScale'] == 0.004


class TestIntegration:
    """Integration Tests für das Zusammenspiel der Repositories"""
    
    @pytest.fixture
    def setup(self):
        """Setup für Integration Tests"""
        temp_dir = tempfile.mkdtemp()
        
        pc_repo = EnhancedPointCloudRepository(temp_dir)
        dist_repo = DistanceRepository(temp_dir)
        param_repo = FileParameterRepository(temp_dir)
        
        yield {
            'temp_dir': temp_dir,
            'pc_repo': pc_repo,
            'dist_repo': dist_repo,
            'param_repo': param_repo
        }
        
        shutil.rmtree(temp_dir)
    
    def test_full_workflow(self, setup):
        """Test kompletter Workflow"""
        pc_repo = setup['pc_repo']
        dist_repo = setup['dist_repo']
        param_repo = setup['param_repo']
        
        # 1. Erstelle und speichere Punktwolken
        np.random.seed(42)
        mov_points = np.random.randn(100, 3)
        ref_points = np.random.randn(100, 3) + 0.1
        
        pc_repo.save_point_cloud(mov_points, "mov.xyz")
        pc_repo.save_point_cloud(ref_points, "ref.xyz")
        
        # 2. Simuliere M3C2 Distanzen
        distances = np.linalg.norm(mov_points - ref_points, axis=1)
        distances[::10] = np.nan
        
        dist_repo.save_distances(distances, "distances.txt")
        
        # 3. Speichere Parameter
        params = {'NormalScale': 0.002, 'SearchScale': 0.004}
        param_repo.save_params(params, "params.txt")
        
        # 4. Speichere mit Koordinaten
        dist_repo.save_distances_with_coordinates(
            mov_points, distances, "distances_coords.txt"
        )
        
        # 5. Lade alles wieder
        loaded_mov = pc_repo.load_point_cloud("mov.xyz")
        loaded_distances = dist_repo.load_distances("distances.txt")
        loaded_params = param_repo.load_params("params.txt")
        
        # Verifikation
        assert loaded_mov is not None
        assert len(loaded_distances) == 100
        assert loaded_params['NormalScale'] == 0.002


if __name__ == "__main__":
    pytest.main([__file__, "-v"])