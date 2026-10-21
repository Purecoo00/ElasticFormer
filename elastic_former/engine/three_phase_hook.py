# Copyright (c) 2026 ElasticFormer authors. SPDX-License-Identifier: Apache-2.0
"""The paper's 24/3/9-epoch training schedule."""

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ElasticThreePhaseHook(Hook):
    """Switch after 24 and 27 completed epochs in a 36-epoch run."""

    def __init__(self, phase1_epochs=24, phase2_epochs=3):
        if phase1_epochs < 0 or phase2_epochs < 0:
            raise ValueError("Phase lengths must be non-negative")
        self.phase1_epochs, self.phase2_epochs = phase1_epochs, phase2_epochs

    def phase_for_epoch(self, epoch):
        if epoch < self.phase1_epochs:
            return 1
        if epoch < self.phase1_epochs + self.phase2_epochs:
            return 2
        return 3

    def before_train_epoch(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        # Derive the phase from the runner epoch so resuming needs no extra state.
        phase = self.phase_for_epoch(runner.epoch)
        model.backbone.set_training_phase(phase)
        runner.logger.info("ElasticFormer training phase %d (epoch %d)", phase, runner.epoch + 1)
