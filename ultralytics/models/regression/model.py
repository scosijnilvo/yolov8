from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.models import regression
from ultralytics.nn.tasks import RegressionDetectionModel, RegressionSegmentationModel


class RegressionModel(Model):
    """
    Extends detection and segmentation models with a regression component for extra variables.
    
    Args:
        model (str, Path): Path to the model file to load or create.
        task (str, optional): Task type for the model. Supported values: 'detect' and 'segment'.
        verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
    """

    def __init__(self, model, task=None, verbose=False):
        """Try to guess task from config filename and initialize the model."""
        assert task == None or task == "detect" or task == "segment", f"Task '{task}' not supported by model."
        path = Path(model)
        if not task and path.suffix == ".yaml":
            if "-det" in path.stem:
                task = "detect"
            elif "-seg" in path.stem:
                task = "segment"
        super().__init__(model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                'model': RegressionDetectionModel,
                'trainer': regression.RegressionDetectionTrainer,
                'validator': regression.RegressionDetectionValidator,
                'predictor': regression.RegressionDetectionPredictor,
            },
            "segment": {
                'model': RegressionSegmentationModel,
                'trainer': regression.RegressionSegmentationTrainer,
                'validator': regression.RegressionSegmentationValidator,
                'predictor': regression.RegressionSegmentationPredictor,
            },
        }
