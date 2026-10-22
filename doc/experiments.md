# Architecture decision

On 22 October 2026 the user accepted the screened architecture: question/candidate self-attention, then cross-attention to full SPOTR signal tokens, followed by a shared MLP scorer. The screen confirmed the direction; its checkpoint is not the starting artifact for the next run. Fresh training is required.

The supported implementation is in `paya/`. There are no active alternative-head selectors, fallback MLPs, reversed attention orders, text-pair encoders, extra answer slots or metadata branches. The head configuration names only the supported computation and its dimensions/dropout. NormWear is outside the active work.

The previous global-vector MLP implementation, its exact checkpoint and matching reports are preserved in ignored `retired/mlp-fusion/`. The experiment framework and original guide are preserved in `retired/attention-experiments/README.md`. The local screen records remain in `artifacts/gpu-screen-20261022/`. These are historical evidence, not measurements of a newly trained attention model.

The promotion preserves the attention operations, parameter names/shapes, frozen encoder math and signal-token mask policy. Local verification checks numerical agreement against the archived head implementation, parameter count, attention order, masks, candidate permutation, cache integrity, head-only gradients, recovery, fine-tuning initialization, independent calibration and export/reload. Real pretrained assets are exercised with synthetic recordings; this establishes wiring and numerical behavior rather than physiological accuracy.

The next controlled run uses `config/paya.yaml`. Follow [training and fine-tuning](finetune.md) for data isolation, selection, calibration and artifact evaluation. No package version is bumped; source/config/data/asset identities distinguish runs. Additional architecture experiments, if later needed, belong in `experiments/` until justified for promotion.
