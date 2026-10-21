import logging
from types import SimpleNamespace

import pytest
import torch
from mmengine.structures import InstanceData

from elastic_former.engine import ElasticThreePhaseHook
from elastic_former.models.backbones.elasticformer import ElasticFormer, LocalAttention
from elastic_former.models.detectors.elastic_dino import foreground_ratio, foreground_targets


def tiny_backbone(with_cp=False):
    return ElasticFormer(
        embed_dims=8,
        depths=(2, 1, 2, 1),
        num_heads=(1, 2, 4, 8),
        window_size=3,
        selector_pool_sizes=(4, 3, 2, 1),
        selector_hidden_dims=(8, 8, 8, 8),
        drop_path_rate=0,
        with_cp=with_cp,
    )


def test_union_area_clips_boxes_without_double_counting():
    boxes = torch.tensor([[-2.0, -2.0, 6.0, 6.0], [4.0, 4.0, 12.0, 12.0]])
    assert foreground_ratio(boxes, (10, 10)) == pytest.approx(0.68)
    assert foreground_ratio(torch.empty(0, 4), (10, 10)) == 0


def test_pseudo_targets_filter_scores_and_detach_boxes():
    instances = InstanceData(
        bboxes=torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 5.0, 5.0]], requires_grad=True),
        scores=torch.tensor([0.2, 0.9]),
    )
    assert foreground_targets([instances], [(10, 10)], 0.8, 0.3) == pytest.approx([0.2])
    assert foreground_targets([instances], [(10, 10)], 0.8) == pytest.approx([0.8])


def test_three_phase_boundaries_and_resume():
    hook = ElasticThreePhaseHook()
    assert [hook.phase_for_epoch(i) for i in (0, 23, 24, 26, 27, 35)] == [1, 1, 2, 2, 3, 3]
    backbone = tiny_backbone()
    runner = SimpleNamespace(
        model=SimpleNamespace(backbone=backbone), epoch=26, logger=logging.getLogger(__name__)
    )
    for epoch, phase in [(0, 1), (24, 2), (30, 3), (26, 2)]:
        runner.epoch = epoch
        hook.before_train_epoch(runner)
        for stage in backbone.stages:
            assert stage.use_dynamic_topk == (phase == 3)
            assert all(p.requires_grad == (phase != 1) for p in stage.top_k_generator.parameters())


def test_batch_window_budgets_are_independent_and_use_ceiling():
    model = tiny_backbone()
    model.train()
    stage = model.stages[0]
    scores = torch.arange(10).float().repeat(2, 1)
    ratios = torch.tensor([0.11, 0.61])
    indices = stage.select_windows(scores, ratios)
    assert [len(i) for i in indices] == [2, 7]
    assert set(indices[0].tolist()) == {8, 9}
    model.set_training_phase(1)
    assert [len(i) for i in stage.select_windows(scores, ratios)] == [10, 10]


def test_unselected_windows_receive_no_local_attention_update():
    attention = LocalAttention(8, 2, 3, False, True, None, 0, 0, 0).eval()
    x = torch.randn(1, 6 * 6, 8)
    update = attention(x, (6, 6), [torch.tensor([0])]).reshape(1, 6, 6, 8)
    assert update[:, :3, :3].abs().sum() > 0
    assert torch.count_nonzero(update[:, 3:]) == 0
    assert torch.count_nonzero(update[:, :, 3:]) == 0


@pytest.mark.parametrize("with_cp", [False, True])
def test_odd_size_backbone_trains_all_global_blocks_and_selectors(with_cp):
    torch.manual_seed(5)
    model = tiny_backbone(with_cp)
    model.train()
    outputs = model(torch.randn(2, 3, 125, 133))
    assert [o.shape[-2:] for o in outputs] == [(32, 34), (16, 17), (8, 9), (4, 5)]
    assert all(torch.isfinite(o).all() for o in outputs)
    loss = sum((o * torch.randn_like(o)).mean() for o in outputs) + model.selection_loss(
        torch.tensor([0.1, 0.8])
    )
    loss.backward()
    for stage in model.stages:
        assert stage.top_k_generator[0].weight.grad.abs().sum() > 0
        assert torch.isfinite(stage.score_net.weight.grad).all()
        assert stage.score_net.weight.grad.abs().sum() > 0
        for block in stage.global_blocks:
            assert block.attn.qkv.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        assert all(torch.isfinite(o).all() for o in model(torch.randn(1, 3, 125, 133)))
