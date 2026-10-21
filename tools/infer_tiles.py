"""Tiled high-resolution inference with full-image coordinate restoration."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from mmcv.ops import batched_nms
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector


def tile_starts(length, tile, stride):
    if min(length, tile, stride) <= 0:
        raise ValueError("Image, tile and stride dimensions must be positive")
    if length <= tile:
        return [0]
    # Add the final tile explicitly so the far edge is always covered.
    return sorted(set(range(0, length - tile + 1, stride)) | {length - tile})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--ann-file", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tile-width", type=int, default=1280)
    p.add_argument("--tile-height", type=int, default=800)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--score-thr", type=float, default=0.05)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--max-detections", type=int, default=500)
    args = p.parse_args()
    if not 0 <= args.overlap < 1:
        p.error("--overlap must be in [0, 1)")
    with open(args.ann_file) as f:
        annotations = json.load(f)
    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.dataset.pipeline = [
        t for t in cfg.test_dataloader.dataset.pipeline if t["type"] != "LoadAnnotations"
    ]
    model = init_detector(cfg, args.checkpoint, device=args.device)
    categories = annotations["categories"]
    classes = model.dataset_meta.get("classes")
    if classes is not None and all(any(c["name"] == n for c in categories) for n in classes):
        cat_ids = [next(c["id"] for c in categories if c["name"] == n) for n in classes]
    else:
        cat_ids = [c["id"] for c in categories]
    if len(cat_ids) != cfg.model.bbox_head.num_classes:
        raise ValueError("Annotation categories and detector class count differ")
    records = []
    for image in annotations["images"]:
        path = Path(args.image_root) / image["file_name"]
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        height, width = frame.shape[:2]
        boxes, scores, labels = [], [], []
        ys = tile_starts(
            height, args.tile_height, max(1, int(args.tile_height * (1 - args.overlap)))
        )
        xs = tile_starts(width, args.tile_width, max(1, int(args.tile_width * (1 - args.overlap))))
        for y in ys:
            for x in xs:
                tile = frame[y : y + args.tile_height, x : x + args.tile_width]
                pred = inference_detector(model, tile).pred_instances
                keep = pred.scores >= args.score_thr
                b = pred.bboxes[keep].detach().cpu().clone()
                # Map tile predictions to the original frame before merging.
                b[:, 0::2] += x
                b[:, 1::2] += y
                b[:, 0::2].clamp_(0, width)
                b[:, 1::2].clamp_(0, height)
                boxes.append(b)
                scores.append(pred.scores[keep].detach().cpu())
                labels.append(pred.labels[keep].detach().cpu())
        boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
        # Class-aware NMS removes duplicate detections from overlapping tiles.
        if len(boxes):
            _, keep = batched_nms(
                boxes, scores, labels, dict(type="nms", iou_threshold=args.nms_iou)
            )
            for i in keep[: args.max_detections].tolist():
                x1, y1, x2, y2 = boxes[i].tolist()
                records.append(
                    dict(
                        image_id=image["id"],
                        category_id=cat_ids[int(labels[i])],
                        bbox=[x1, y1, x2 - x1, y2 - y1],
                        score=float(scores[i]),
                    )
                )
        print("Processed image", image["id"], "tiles", len(xs) * len(ys))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records))
    print("Saved", len(records), "detections to", output)


if __name__ == "__main__":
    main()
