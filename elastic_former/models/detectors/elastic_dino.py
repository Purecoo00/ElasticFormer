# Copyright (c) 2026 ElasticFormer authors. SPDX-License-Identifier: Apache-2.0
"""DINO with foreground-ratio supervision for ElasticSelector."""

import torch
from shapely.geometry import box
from shapely.ops import unary_union

from mmdet.models.detectors import DINO
from mmdet.registry import MODELS


def foreground_ratio(boxes, image_shape):
    """Measure clipped box coverage, counting overlapping areas only once."""
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    rectangles = []
    for x1, y1, x2, y2 in boxes.detach().cpu().tolist():
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 > x1 and y2 > y1:
            rectangles.append(box(x1, y1, x2, y2))
    return unary_union(rectangles).area / (height * width) if rectangles else 0.0


def foreground_targets(instances, image_shapes, alpha=0.8, score_threshold=None):
    targets = []
    for result, shape in zip(instances, image_shapes):
        boxes = result.bboxes
        if score_threshold is not None:
            boxes = boxes[result.scores > score_threshold]
        targets.append(alpha * foreground_ratio(boxes, shape))
    return targets


@MODELS.register_module()
class ElasticDINO(DINO):
    """Keep DINO detection losses and add the stage-wise selection objective.

    Pseudo boxes supervise the selector. The detector itself still uses the
    PANDA ground-truth boxes in this fully supervised configuration.
    """

    def __init__(
        self,
        *args,
        selector_supervision="pseudo",
        foreground_alpha=0.8,
        pseudo_score_threshold=0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if selector_supervision not in ("pseudo", "gt"):
            raise ValueError("Selector supervision must be pseudo or gt")
        self.selector_supervision = selector_supervision
        self.foreground_alpha = foreground_alpha
        self.pseudo_score_threshold = pseudo_score_threshold

    def loss(self, batch_inputs, batch_data_samples):
        features = self.extract_feat(batch_inputs)
        head_inputs = self.forward_transformer(features, batch_data_samples)
        losses = self.bbox_head.loss(**head_inputs, batch_data_samples=batch_data_samples)
        if not self.backbone.stages[0].train_selector:
            return losses
        if self.selector_supervision == "pseudo":
            # Reuse this forward pass; pseudo labels must not receive gradients.
            with torch.no_grad():
                instances = self.bbox_head.predict(
                    hidden_states=head_inputs["hidden_states"],
                    references=head_inputs["references"],
                    batch_data_samples=batch_data_samples,
                    rescale=False,
                )
            threshold = self.pseudo_score_threshold
        else:
            instances = [sample.gt_instances for sample in batch_data_samples]
            threshold = None
        # Predictions and image bounds use the same augmented coordinates.
        targets = foreground_targets(
            instances, [s.img_shape for s in batch_data_samples], self.foreground_alpha, threshold
        )
        losses["loss_select"] = self.backbone.selection_loss(batch_inputs.new_tensor(targets))
        return losses
