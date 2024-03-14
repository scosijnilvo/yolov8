from copy import copy
from ultralytics.utils import RANK
from ultralytics.models import yolo, regression
from ultralytics.nn.tasks import RegressionDetectionModel, RegressionSegmentationModel
from ultralytics.data import build_custom_dataset
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import de_parallel


class RegressionTrainer():
    """A mixin class with shared methods for `RegressionDetectionTrainer` and `RegressionSegmentationTrainer`."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a `CustomDataset` in train mode."""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_custom_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)


class RegressionDetectionTrainer(RegressionTrainer, yolo.detect.DetectionTrainer):
    """Extends `DetectionTrainer` with a custom model and validator."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return RegressionDetectionModel initialized with specified config and extra vars."""
        model = RegressionDetectionModel(cfg, ch=3, nc=self.data["nc"], nv=self.args["num_vars"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of RegressionDetectionValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "reg_loss"
        return regression.RegressionDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, regression=True, on_plot=self.on_plot)


class RegressionSegmentationTrainer(RegressionTrainer, yolo.segment.SegmentationTrainer):
    """Extends `SegmentationTrainer` with a custom model and validator."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return RegressionSegmentationModel initialized with specified config and extra vars."""
        model = RegressionSegmentationModel(cfg, ch=3, nc=self.data["nc"], nv=self.args["num_vars"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of RegressionSegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "reg_loss"
        return regression.RegressionSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, regression=True, segment=True, on_plot=self.on_plot)
