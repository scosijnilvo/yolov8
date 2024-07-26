from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RegressionDetectionPredictor(DetectionPredictor):
    """
    Extends `DetectionPredictor` with regression for extra variables.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of RegressionResults objects."""
        num_vars = preds[1][1].shape[1]
        p = ops.non_max_suppression(
            preds[0],
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
                extra_vars = None
            else:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                extra_vars = pred[:, -num_vars:]
            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    extra_vars=extra_vars
                )
            )
        return results


class RegressionSegmentationPredictor(SegmentationPredictor):
    """
    Extends `SegmentationPredictor` with regression for extra variables.
    """
    
    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        num_vars = preds[1][3].shape[1]
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
                extra_vars = None
            else:
                if self.args.retina_masks:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    masks = ops.process_mask_native(proto[i], pred[:, 6:-num_vars], pred[:, :4], orig_img.shape[:2])  # HWC
                else:
                    masks = ops.process_mask(proto[i], pred[:, 6:-num_vars], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                extra_vars = pred[:, -num_vars:]
            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    masks=masks,
                    extra_vars=extra_vars
                )
            )
        return results
