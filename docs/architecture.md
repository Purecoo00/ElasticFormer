# Architecture and training

The backbone has stage depths `[2, 2, 6, 2]`, channels `[96, 192, 384, 768]`, and attention heads `[3, 6, 12, 24]`. Local windows have size 7. Each stage pools one token per window for global attention, scores local residuals for window selection, and applies attention and an FFN to selected windows. Alternating local blocks use shifted windows with boundary and padding masks.

ElasticSelector applies adaptive pooling at sizes `[32, 16, 8, 4]` followed by a two-layer MLP with hidden dimensions `[256, 128, 64, 32]` and a sigmoid. The window count is `ceil(predicted_ratio * number_of_windows)`, bounded to at least one. Budgets are independent for each image in a batch.

Window scores come from the feature residual after average pooling and bilinear interpolation. A softmax normalizes the scores. An identity-forward straight-through gain gives the scorer a gradient through hard top-k selection. Global context is interpolated to the original feature grid and weighted by `1 - window_score` before the local blocks.

## Selector supervision

The target is `0.8 * union_area(boxes) / image_area`. Boxes are clipped to the image bounds, and overlapping regions count once. The selection loss is the mean of `(prediction - target)^2 + 0.1 * prediction` across images and stages.

The default DINO configuration uses predicted boxes with confidence above 0.3 to supervise the selector. Predictions are detached and use the same augmented coordinates as the current training images. The DINO detection losses still use ground-truth annotations. Set `model.selector_supervision=gt` for the ground-truth selector ablation.

## Three-phase schedule

For the 36-epoch configuration, Phase 1 lasts 24 epochs, Phase 2 lasts 3, and Phase 3 lasts 9. The first two phases process all windows. Only the selector is frozen in Phase 1. Phase 2 trains the selector before its predictions control computation; Phase 3 enables dynamic selection and joint training. Inference always uses dynamic selection.

`ElasticThreePhaseHook` runs before each training epoch and uses zero-based epoch indices. Thus indices 0–23 are Phase 1, 24–26 are Phase 2, and 27–35 are Phase 3. This also makes phase selection correct when resuming. Distributed training uses `find_unused_parameters=True` because selector parameters do not participate in Phase 1.

Activation checkpointing reduces memory use. This implementation uses PyTorch and MMDetection operators; no custom sparse CUDA kernel or benchmark weights are bundled.
