import torch
import numpy as np
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.metrics import RegressionSegmentMetrics, RegressionDetMetrics
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.data.build import build_custom_dataset


class RegressionValidator():
    """A mixin class with shared methods for `RegressionDetectionValidator` and `RegressionSegmentationValidator`."""

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        prepared_batch = super()._prepare_batch(si, batch)
        idx = batch["batch_idx"] == si
        prepared_batch["extra_vars"] = batch["extra_vars"][idx]
        return prepared_batch

    def _process_batch_vars(self, pred, gt_bboxes, gt_cls, gt_vars):
        """
        Process extra vars in a batch of predictions.
        Returns ground-truth vars, predicted vars, and predicted classes for true positive detections in the batch.
        """
        conf = 0.25 if self.args.conf in (None, 0.001) else self.args.conf
        pred = pred[pred[:, 4] > conf]
        pred_vars = pred[:, -self.num_vars:]
        pred_cls = pred[:, 5]
        correct_class = gt_cls[:, None] == pred_cls
        iou = box_iou(gt_bboxes, pred[:, :4])
        iou = iou * correct_class
        iou = iou.cpu().numpy()
        tp_idx = np.nonzero(iou > 0.45)
        tp_idx = np.array(tp_idx).T
        if tp_idx.shape[0]:
            if tp_idx.shape[0] > 1:
                tp_idx = tp_idx[iou[tp_idx[:, 0], tp_idx[:, 1]].argsort()[::-1]]
                tp_idx = tp_idx[np.unique(tp_idx[:, 1], return_index=True)[1]]
                tp_idx = tp_idx[np.unique(tp_idx[:, 0], return_index=True)[1]]
        tp_vars = []
        for tp in tp_idx:
            gt_idx, pred_idx = tp[0], tp[1]
            tp_vars.append(torch.stack([
                gt_vars[gt_idx],
                pred_vars[pred_idx],
                pred_cls[pred_idx].expand(self.num_vars)
            ]))
        return torch.stack(tp_vars) if tp_vars else torch.zeros(0, 3, self.num_vars, device=pred.device)

    def preprocess(self, batch):
        """Preprocesses batch by sending extra_vars to device."""
        batch = super().preprocess(batch)
        batch["extra_vars"] = batch["extra_vars"].to(self.device)
        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build a `CustomDataset` in val mode."""
        return build_custom_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels, masks (if segmentation), and variable values."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch.get("masks", np.zeros(0, dtype=np.uint8)),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
            extra_vars=batch["extra_vars"]
        )


class RegressionDetectionValidator(RegressionValidator, DetectionValidator):
    """Extends `DetectionValidator` with a custom metrics class to calculate regression metrics of predicted variables."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize the validator with `RegressionDetMetrics`."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        reg_fitness = args.get("reg_fitness", False)
        self.metrics = RegressionDetMetrics(save_dir=self.save_dir, on_plot=self.on_plot, reg_fitness=reg_fitness)
        self.num_vars = None # value set during postprocess

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        self.num_vars = preds[1][1].shape[1]
        return ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
    
    def init_metrics(self, model):
        """Initialize evaluation metrics for detections and extra variables."""
        super().init_metrics(model)
        self.stats = dict(
            tp=[],
            conf=[],
            pred_cls=[],
            target_cls=[],
            tp_vars=[]
        )

    def update_metrics(self, preds, batch):
        """Update metrics with values from predictions and batch."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_vars=torch.zeros(0, 3, self.num_vars, device=self.device)
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox, extra_vars = pbatch.pop("cls"), pbatch.pop("bbox"), pbatch.pop("extra_vars")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_vars"] = self._process_batch_vars(predn, bbox, cls, extra_vars)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])

    def pred_to_json(self, predn, filename):
        """Serialize predictions to json."""
        super().pred_to_json(predn, filename)
        for i, p in enumerate(predn):
            self.jdict[i]["extra_vars"] = p[:, -self.num_vars:]

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 7) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "MAPE",
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with bounding boxes and predicted values."""
        extra_vars = []
        for p in preds:
            extra_vars.append(p[:self.args.max_det, -self.num_vars:].cpu())
        extra_vars = torch.cat(extra_vars, 0).numpy()
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
            extra_vars=extra_vars
        )


class RegressionSegmentationValidator(RegressionValidator, SegmentationValidator):
    """Extends `SegmentationValidator` with a custom metrics class to calculate regression metrics of predicted variables."""
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize the validator with `RegressionSegmentMetrics`."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        reg_fitness = args.get("reg_fitness", False)
        self.metrics = RegressionSegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot, reg_fitness=reg_fitness)
        self.num_vars = None # value set during postprocess

    def _prepare_pred(self, pred, pbatch, proto):
        """Prepares a batch for training or inference by processing images and targets."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        pred_masks = self.process(proto, pred[:, 6:-self.num_vars], pred[:, :4], shape=pbatch["imgsz"])
        return predn, pred_masks

    def postprocess(self, preds):
        """Post-processes predictions and returns output detections with proto."""
        self.num_vars = preds[1][3].shape[1]
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        proto = preds[1][2] if len(preds[1]) == 4 else preds[1]  # second output is len 4 if pt, but only 1 if exported
        return p, proto

    def init_metrics(self, model):
        """Initialize evaluation metrics for masks, detections, and extra variables."""
        super().init_metrics(model)
        self.stats = dict(
            tp_m=[],
            tp=[],
            conf=[],
            pred_cls=[],
            target_cls=[],
            tp_vars=[]
        )

    def update_metrics(self, preds, batch):
        """Update metrics with values from predictions and batch."""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_vars=torch.zeros(0, 3, self.num_vars, device=self.device)
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox, extra_vars = pbatch.pop("cls"), pbatch.pop("bbox"), pbatch.pop("extra_vars")
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Masks
            gt_masks = pbatch.pop("masks")
            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_m"] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                )
                stat["tp_vars"] = self._process_batch_vars(predn, bbox, cls, extra_vars)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save
            if self.args.save_json:
                pred_masks = ops.scale_image(
                    pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                    pbatch["ori_shape"],
                    ratio_pad=batch["ratio_pad"][si],
                )
                self.pred_to_json(predn, batch["im_file"][si], pred_masks)

    def pred_to_json(self, predn, filename, pred_masks):
        """Serialize predictions to json."""
        super().pred_to_json(predn, filename, pred_masks)
        for i, p in enumerate(predn):
            self.jdict[i]["extra_vars"] = p[:, -self.num_vars:]

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 11) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "MAPE",
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks, bounding boxes, and predicted values."""
        max_det = 15 # not set to self.args.max_det due to slow plotting speed
        extra_vars = []
        for p in preds[0]:
            extra_vars.append(p[:max_det, -self.num_vars:].cpu())
        extra_vars = torch.cat(extra_vars, 0).numpy()
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=max_det),
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
            extra_vars=extra_vars
        )
        self.plot_masks.clear()
