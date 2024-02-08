from pathlib import Path
from typing import Union
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import WeightSegmentationModel


class WeightModel(Model):
    """Object detection and segmentation model with weight prediction."""

    def __init__(self, model, verbose=False):
        super().__init__(model, task="segment", verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "segment": {
                'model': WeightSegmentationModel,
                'trainer': yolo.segment.WeightSegmentationTrainer,
                'validator': yolo.segment.WeightSegmentationValidator,
                'predictor': yolo.segment.WeightSegmentationPredictor,
            },
        }
