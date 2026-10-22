# Local verification

22 October 2026. The accepted attention architecture is the only supported head. Fresh research training has not started and no screening weights were promoted.

| Check | Result |
| --- | --- |
| Suite including pinned SPOTR/BioLORD integration | 161 passed, 2 deselected |
| Focused cache/checkpoint/fine-tuning/recovery tests after final identity guards | 28 passed |
| Supported head versus archived screened head, same tensors and weights | Bit-identical logits in evaluation and seeded training |
| Full signal encoder output versus the unchanged pretrained core | Bit-identical tokens; 15 valid keys of 16 for 1,250 samples |
| Fresh/cache logits using real frozen assets | Exact agreement |
| Attention order, padding masks, candidate permutation and frozen gradients | Passed |
| Interrupted versus uninterrupted CPU training | Matching learned weights, including step-based selection and dropout, with both cache formats |
| Fine-tuning initialization | Parent weights retained; fresh optimizer, temperature 1 and no inherited calibration |
| Calibration binding, dataset isolation and export/reload | Passed |
| Static checks, documentation links and whitespace checks | Passed |

The two existing real offline W&B tracking tests remain excluded because that service stalled in this environment during earlier verification. Mocked/disabled tracking checks pass. One existing NeuroKit/SciPy deprecation warning remains. No GPU training, GPU recovery, timing, fresh accuracy or ECE claim follows from these tests.

The archived MLP checkpoint remains byte-identical, SHA-256 `461b939b19ef59bfa05725c69438b2e8332e031b2586aeab9234fe848c633eba`, with its matching code, export and reports in ignored `retired/mlp-fusion/`. The package version remains unchanged. Current artifact status is recorded in `models/checkpoint.json`.
