from copy import copy
from ultralytics.utils import RANK
from ultralytics.models import yolo, weight
from ultralytics.nn.tasks import WeightDetectionModel, WeightSegmentationModel
from ultralytics.data import build_weight_dataset
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import de_parallel


class WeightTrainer():
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_weight_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)


class WeightDetectionTrainer(WeightTrainer, yolo.detect.DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return WeightDetectionModel initialized with specified config and weights."""
        model = WeightDetectionModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of WeightDetectionValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "wgt_loss"
        return weight.WeightDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, weight=True, on_plot=self.on_plot)


class WeightSegmentationTrainer(WeightTrainer, yolo.segment.SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return WeightSegmentationModel initialized with specified config and weights."""
        model = WeightSegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of WeightSegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "wgt_loss"
        return weight.WeightSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, weight=True, segment=True, on_plot=self.on_plot)
