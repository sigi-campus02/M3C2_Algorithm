# infrastructure/repositories/file_point_cloud_repository.py
class FilePointCloudRepository:
    """Konkrete Implementierung für Dateisystem"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def load_point_cloud(self, path: str) -> np.ndarray:
        # Implementierung
        pass