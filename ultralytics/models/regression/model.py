from ultralytics.engine.model import Model
from ultralytics.models import regression
from ultralytics.nn.tasks import RegressionDetectionModel, RegressionSegmentationModel


class RegressionModel(Model):
    """
    Extends detection and segmentation models with a regression component for extra variables.
    
    Args:
        model (str, Path): Path to the model file to load or create.
        task (str): Task type for the model. Supported values: 'detect' and 'segment'.
        verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
    """

    def __init__(self, model, task, verbose=False):
        """Initializes the model."""
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
