import json
import pytest
from pycocotools.coco import COCO
from tools.infer_tiles import tile_starts
from elastic_former.evaluation.panda import PANDAEval


def test_tiles_cover_edges_without_duplicate_starts():
    assert tile_starts(500, 1280, 960) == [0]
    starts = tile_starts(2551, 1280, 960)
    assert starts == [0, 960, 1271]
    assert starts[-1] + 1280 == 2551
    assert all(b <= a + 1280 for a, b in zip(starts, starts[1:]))


def test_panda_evaluator_reports_perfect_detection(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            dict(
                info={},
                images=[dict(id=1, width=1000, height=1000, file_name="synthetic.png")],
                categories=[dict(id=1, name="person")],
                annotations=[
                    dict(
                        id=1, image_id=1, category_id=1, bbox=[10, 20, 20, 30], area=600, iscrowd=0
                    )
                ],
            )
        )
    )
    gt = COCO(str(path))
    dt = gt.loadRes([dict(image_id=1, category_id=1, bbox=[10, 20, 20, 30], score=0.99)])
    evaluator = PANDAEval(gt, dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    assert evaluator.params.areaRng[1][1] == 96**2
    assert evaluator.params.areaRng[2][1] == 288**2
    assert evaluator.params.maxDets == [500, 500, 500]
    assert evaluator.stats[1] == pytest.approx(1.0)
    assert evaluator.stats[3] == pytest.approx(1.0)
