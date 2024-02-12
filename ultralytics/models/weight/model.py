from ultralytics.engine.model import Model
from ultralytics.models import weight
from ultralytics.nn.tasks import WeightDetectionModel, WeightSegmentationModel


class WeightModel(Model):
    """Object detection and segmentation model with weight prediction."""

    def __init__(self, model, task, verbose=False):
        super().__init__(model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                'model': WeightDetectionModel,
                'trainer': weight.WeightDetectionTrainer,
                'validator': weight.WeightDetectionValidator,
                'predictor': weight.WeightDetectionPredictor,
            },
            "segment": {
                'model': WeightSegmentationModel,
                'trainer': weight.WeightSegmentationTrainer,
                'validator': weight.WeightSegmentationValidator,
                'predictor': weight.WeightSegmentationPredictor,
            },
        }
