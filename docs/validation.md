# Validation, 2026-10-21

Environment: Python 3.10.19; PyTorch 2.1.0+cu118; torchvision 0.16.0+cu118; MMCV 2.1.0; MMEngine 0.10.7; MMDetection 3.3.0; NumPy 1.26.4; Shapely 2.1.2.

- Nine tests passed, covering clipped foreground union area, pseudo-box filtering, per-image window budgets, sparse attention updates, three-phase boundaries and resume behavior, odd-size backbone gradients with and without checkpointing, edge-covering tiling, and PANDA evaluation.
- On an NVIDIA GeForce RTX 4090, the full DINO configuration completed synthetic training forward/backward passes in all three phases. Phase 1 left the selector frozen; Phases 2 and 3 trained it. Selector, window scorer, global attention and local attention parameters received finite nonzero gradients in Phase 3.
- The synthetic check completed an AdamW optimizer step and detector prediction. It used 224x224 inputs.
- The standard MMEngine Runner completed a training iteration with the released model, optimizer, registry scope and three-phase hook, using a synthetic dataset.

These checks validate code execution and training integration. They do not measure 36-epoch PANDA accuracy, paper FLOPs or end-to-end benchmark latency. Trained weights are not included.
