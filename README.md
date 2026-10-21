# ElasticFormer

**ElasticFormer: Detecting Objects in HRW Shots via Elastic Computing Vision Transformer**  
CVPR 2026  
Wenxi Li*, Jingchen Huang*, Chenyang Lyu, Moran Liu, Haozhe Lin, Guiguang Ding†, Yuchen Guo†  
*Equal contribution. †Corresponding authors.

[Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_ElasticFormer_Detecting_Objects_in_HRW_Shots_via_Elastic_Computing_Vision_CVPR_2026_paper.pdf) · [Architecture and training](docs/architecture.md) · [Data preparation](docs/data.md)

PyTorch implementation of **ElasticFormer** for high-resolution, wide-field object detection. ElasticSelector predicts the foreground proportion at each backbone stage and selects high-scoring windows for local attention. Global attention over pooled tokens supplies context at a lower cost. The number of selected windows adapts to each image.

This repository includes the ElasticFormer backbone, DINO integration, the three-phase PANDA training configuration, and tiled inference and evaluation tools. The detector uses bounding-box annotations; pseudo boxes supervise ElasticSelector. The separate ElasticNet and VOC weakly supervised experiments are outside this release. Trained weights are not included.

## Installation

```bash
conda create -n elasticformer python=3.10 -y
conda activate elasticformer
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -r requirements.txt
pip install -e . --no-deps
```

Use the MMCV wheel matching your PyTorch/CUDA build. DINO requires the compiled deformable-attention operator provided by `mmcv`.

## Training

Prepare COCO-format PANDA patches following [data.md](docs/data.md). The configuration trains for 36 epochs with AdamW, a learning rate of 0.0002, and one image per GPU. The launcher below uses four GPUs.

```bash
bash scripts/dist_train.sh configs/dino/dino_elasticformer_t_4xb1_36e_panda.py 4
```

The selector follows this schedule:

| Epochs | Selector | Window budget |
| --- | --- | --- |
| 1–24 | Frozen | All windows |
| 25–27 | Train with pseudo boxes | All windows |
| 28–36 | Jointly trained | Predicted per image |

The hook switches after 24 and 27 completed epochs. It derives the phase from the runner's epoch counter, including when resuming training. At inference, the backbone uses its predicted window budgets.

```bash
# Resume the latest checkpoint in the selected work directory.
python tools/train.py configs/dino/dino_elasticformer_t_4xb1_36e_panda.py \
  --work-dir work_dirs/elasticformer --resume

# Use ground-truth boxes to supervise the selector for the GT ablation.
python tools/train.py configs/dino/dino_elasticformer_t_4xb1_36e_panda.py \
  --cfg-options model.selector_supervision=gt
```

## Evaluation

```bash
bash scripts/dist_test.sh configs/dino/dino_elasticformer_t_4xb1_36e_panda.py path/to/checkpoint.pth 4
```

The standard configuration reports **patch-level COCO metrics**. For full-frame PANDA evaluation, use [tiled inference and PANDA evaluation](docs/data.md#full-frame-inference-and-evaluation). Tile predictions must be mapped back to full-image coordinates and merged before applying PANDA's area ranges and detection limits.

## Checks

```bash
python -m pytest tests -q
python tools/smoke_test.py --device cuda:0
```

The tests cover per-image window budgets, sparse updates, foreground coverage, phase boundaries and gradients. The synthetic smoke test exercises all three training phases, an optimizer step and DINO inference. See the [validation record](docs/validation.md) for the environment and checks. These checks validate code execution and training integration; they do not measure benchmark accuracy.

## Repository structure

```text
elastic_former/models/backbones/elasticformer.py  # global/local blocks and ElasticSelector
elastic_former/models/detectors/elastic_dino.py   # DINO and selector supervision
elastic_former/engine/three_phase_hook.py         # three-phase training schedule
configs/                                         # dataset, model and training configuration
tools/                                           # train, test, tiled inference and evaluation
tests/                                           # routing, schedule and numerical checks
```

## Citation

```bibtex
@inproceedings{li2026elasticformer,
  title={ElasticFormer: Detecting Objects in HRW Shots via Elastic Computing Vision Transformer},
  author={Li, Wenxi and Huang, Jingchen and Lyu, Chenyang and Liu, Moran and Lin, Haozhe and Ding, Guiguang and Guo, Yuchen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Acknowledgments and license

The sparse global/local backbone and PANDA evaluator build on [liwenxi/SparseFormer](https://github.com/liwenxi/SparseFormer). Patch embedding, patch merging, shifted-window attention, DINO and the training framework use [MMDetection](https://github.com/open-mmlab/mmdetection), MMCV and MMEngine. See [NOTICE](NOTICE) for attribution and [LICENSE](LICENSE) for the Apache License 2.0.
