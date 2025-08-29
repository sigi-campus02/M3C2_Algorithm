# domain/validators/point_cloud_validators.py
class PointCountValidator(Validator):
    def __init__(self, min_points: int = 1000):
        super().__init__()
        self.min_points = min_points
    
    def _validate_impl(self, cloud: np.ndarray) -> bool:
        return len(cloud) >= self.min_points

class DimensionValidator(Validator):
    def _validate_impl(self, cloud: np.ndarray) -> bool:
        return cloud.shape[1] >= 3