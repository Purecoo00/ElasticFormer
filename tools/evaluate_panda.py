"""Evaluate merged full-frame predictions using SparseFormer's PANDA ranges."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pycocotools.coco import COCO
from elastic_former.evaluation.panda import PANDAEval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("annotations")
    p.add_argument("predictions")
    args = p.parse_args()
    ground_truth = COCO(args.annotations)
    with open(args.predictions) as f:
        records = json.load(f)
    if records:
        detections = ground_truth.loadRes(records)
    else:
        detections = COCO()
        detections.dataset = dict(
            images=ground_truth.dataset["images"],
            categories=ground_truth.dataset["categories"],
            annotations=[],
        )
        detections.createIndex()
    evaluator = PANDAEval(ground_truth, detections, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


if __name__ == "__main__":
    main()
