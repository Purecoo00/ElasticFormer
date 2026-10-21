# PANDA data

Obtain PANDA from the dataset maintainers and follow its usage terms. Images, annotations and weights are not distributed in this repository.

The training/test entrypoints accept COCO-format patch datasets:

```text
data/panda/
  annotations/train.json
  annotations/val.json
  annotations/test.json
  images/train/...
  images/val/...
  images/test/...
```

The default category name is `person`. If converted annotations use another name, update each dataset's `metainfo.classes` to match it. Split by original PANDA scene **before** extracting patches so related tiles cannot cross splits. Keep full-frame image IDs and tile offsets if full-frame evaluation is required.

The DINO training configuration uses multiscale resize/crop augmentation and one image per GPU. Prepare and validate your patch splits before training. For PANDA-specific preparation examples, consult [SparseFormer](https://github.com/liwenxi/SparseFormer/tree/main/tools).

Replace the default root in `configs/_base_/datasets/panda_detection.py`, or override roots and evaluator annotation paths with `--cfg-options` when running the entrypoints.

## Full-frame inference and evaluation

`tools/infer_tiles.py` consumes a COCO annotation JSON whose `images` entries refer to full-resolution images. It processes 1280x800 windows, maps predictions back to the full image, and merges overlapping detections using class-aware NMS. The tile stride/overlap, score threshold, NMS threshold and final detection limit are command-line options rather than hard-coded experiment claims.

```bash
python tools/infer_tiles.py \
  configs/dino/dino_elasticformer_t_4xb1_36e_panda.py path/to/checkpoint.pth \
  --ann-file data/panda/full_frame_test.json --image-root data/panda/full_frames \
  --out predictions.json --device cuda:0
python tools/evaluate_panda.py data/panda/full_frame_test.json predictions.json
```

The PANDA evaluator follows SparseFormer's AP at IoU=0.5, area ranges below 96x96, 96x96--288x288 and above 288x288, with a 500-detection evaluation limit. Ordinary `CocoMetric` patch evaluation uses different area ranges and measures a different task.
