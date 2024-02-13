from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.engine.results import WeightResults
from ultralytics.utils import ops


class WeightDetectionPredictor(DetectionPredictor):
    """
    Extends `DetectionPredictor` with weight of objects.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of WeightResults objects."""
        p = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            nc=len(self.model.names),
            classes=self.args.classes,
        )
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):
                weights = None
            else:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                weights = pred[:, -1]
            results.append(
                WeightResults(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    weights=weights
                )
            )
        return results


class WeightSegmentationPredictor(SegmentationPredictor):
    """
    Extends `SegmentationPredictor` with weight of objects.
    """
    
    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        proto = preds[1][2] if len(preds[1]) == 4 else preds[1]  # second output is len 4 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
                weights = None
            else:
                if self.args.retina_masks:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    masks = ops.process_mask_native(proto[i], pred[:, 6:-1], pred[:, :4], orig_img.shape[:2])  # HWC
                else:
                    masks = ops.process_mask(proto[i], pred[:, 6:-1], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                weights = pred[:, -1]
            results.append(
                WeightResults(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    masks=masks,
                    weights=weights
                )
            )
        return results
