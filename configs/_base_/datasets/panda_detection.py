# COCO-format PANDA patches; prepare splits independently at the scene level.
dataset_type = "CocoDataset"
data_root = "data/panda/"
backend_args = None
metainfo = dict(classes=("person",), palette=[(220, 20, 60)])
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomResize", scale=(1333, 800), ratio_range=(0.5, 1.5), keep_ratio=True),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1280, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/train.json",
        data_prefix=dict(img="images/train/"),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/val/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/test.json",
        data_prefix=dict(img="images/test/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
# This is patch-level COCO evaluation. Full-frame PANDA evaluation requires
# merging tile predictions; see tools/infer_tiles.py and docs/data.md.
val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/val.json", metric="bbox")
test_evaluator = dict(
    type="CocoMetric", ann_file=data_root + "annotations/test.json", metric="bbox"
)
