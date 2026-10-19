# Architecture implementation for review

Implemented on 19 October 2026. This is the replacement architecture, with a fresh trainable router/head and frozen NormWear, MantisV2, and BioLORD. No old decision weights, feature caches, calibration results, or benchmark claims were carried into it.

## What changed

| Area | Implementation |
| --- | --- |
| Public contract | `paya/inputs.py`: separate `state`, `question`, and ordered `criteria`; strict nine-type enum; finite positive numeric per-channel rates; units, timestamps, shared intervals, and budgets. |
| Preparation | `paya/preprocessing.py`: deterministic NeuroKit filtering, physical unit conversion, independent prepared views, compatibility masks, complete real frames, and audit-only QC. All preparation finishes before routing. |
| Router | `paya/router.py`: shared waveform Conv1D layers, pooled waveform features, question projection, and a masked two-expert top-1 gate. No metadata vector or criterion enters the gate. |
| Frozen signals | `paya/encoders.py`: pinned NormWear/MantisV2 loaders and extraction. NormWear identity formatting; Mantis right zero-padding. No external learned projection. |
| Frozen text | `paya/text.py`: separately encoded question, criteria, and internal ending; normalized width 768; explicit rejection of text longer than the 128-token context. |
| Decision head | `paya/head.py`: shared text projection, role vectors, two pre-LayerNorm blocks of self-attention/cross-attention/feed-forward, and shared candidate scoring. No sensor metadata or encoder-specific learned weights. |
| Runtime | `paya/model.py`: shared batch handling and exactly one selected signal encoder per request at inference. Output has exactly `choice`, `probabilities`, and `confidence`. |
| Retraining | `paya/data.py`, `paya/train.py`: new JSONL schema, split isolation, candidate permutations, balanced head warmup, training-only expert targets, top-1 training, and resumable optimizer/RNG state. Only routed epochs can become `best.pt`. |
| Calibration | `paya/calibrate.py`: held-out outcome NLL fits one positive temperature under actual top-1 routing. Calibration binds the exact router/head weights. Training clears it. |
| Artifacts | `paya/features.py`, `paya/checkpoint.py`, `paya/provenance.py`, `paya/export.py`: new versioned FP32 caches, explicit checksums, strict asset/preparation identities, checkpoint rejection of old formats, and self-contained bundle loading. |
| Evaluation | `paya/evaluate.py`: same computation as inference, with variable-choice NLL/Brier/accuracy/ECE and reliability counts. No real-dataset evaluation has been run. |

## Internal shapes

`B` is batch size; `K` is candidate count after batch padding; `S=K+2`; `M` is packed signal-vector count. Padding is applied to encoded vectors, never to samples inside a frozen encoder.

| Layer | Input → output | Purpose |
| --- | --- | --- |
| Router shared Conv1D + GELU | `[R,1,L] → [R,16,ceil(L/2)]` | Extract local waveform features. |
| Router shared Conv1D + GELU | Previous → `[R,32,ceil(L/4)]` | Compress waveform features. |
| Temporal and frame pooling | Per-frame features → `[B,2,32] → [B,64]` | Represent the two available preparations. |
| Router question linear + GELU | `[B,768] → [B,128]` | Condition expert selection on the separate question. |
| Router MLP | `[B,192] → [B,64] → [B,2]` | Choose one compatible expert. |
| NormWear fixed adapter | `[M,768] → [M,768]` | Preserve its original coordinates. |
| Mantis fixed adapter | `[M,512] → [M,768]` | Append 256 zeros without learned weights. |
| Shared text linear + LayerNorm | `[B,S,768] → [B,S,256]` | Initialize question/criteria/ending states. |
| Self-attention, each of two blocks | `[B,S,256] → [B,S,256]`, four heads × 64 | Let text states interact bidirectionally. |
| Cross-attention, each block | Q `[B,S,256]`; signal K/V `[B,M,768]` projected to 256 → `[B,S,256]` | Read the selected encoded waveform. |
| Feed-forward, each block | `[B,S,256] → [B,S,1024] → [B,S,256]` | Refine the joint states. |
| Final LayerNorm, candidate rows, shared scorer | `[B,K,256] → [B,K,1] → [B,K]` | Produce one logit per original candidate. |
| Temperature and masked softmax | `[B,K] → [B,K]` | Produce probabilities in candidate order. |

The fixed adapters format dimensions; they do not align independently pretrained feature spaces. The ordinary shared attention projections learn the decision task during retraining.

The default router has 113,666 trainable parameters; the shared head has 2,830,081. The output adapters have zero parameters.

The actual frozen weights were exercised using a synthetic six-second, 500 Hz single-channel ECG window and three separate candidate strings:

| Route | Prepared frames | Native output | Formatted signal vectors | Router logits | Decision logits |
| --- | --- | --- | --- | --- | --- |
| NormWear | `[1,1,390]` | `[105,768]` | `[105,768]` | `[1,2]` | `[1,3]` |
| MantisV2 | `[11,1,512]` | `[11,512]` | `[11,768]` | `[1,2]` | `[1,3]` |

The head adds the request batch axis to those formatted vectors. BioLORD produced question `[1,768]`, criteria `[1,3,768]`, and ending `[1,768]`. The public probability list had length three and summed to one. The machine-readable report is `artifacts/architecture-shapes.json`.

## Verification

The check suite covers invalid/missing types and rates; all nine preparations; equivalent units; constant temperature preservation; real frame coverage; interval alignment; bounded resampling; separate text encoding; all layer widths; adapter exactness; stage order; single dispatch; variable candidate counts; masks; candidate/channel permutations; gradients and frozen weights; new dataset/split contracts; training-only routing targets; tiny synthetic training/resume; cache/checkpoint integrity; temperature binding; and export/load parity with small asset doubles.

The integration check uses the real pinned signal and text checkpoints and verifies both signal routes plus the gradient boundary. It performs no optimizer step on those checkpoints. Synthetic training/calibration unit fixtures use tiny frozen encoder doubles and temporary files only.

Result: **60 passed** (59 unit checks and one real-asset integration check), zero failures/skips. The integration check also exercised a mixed batch containing a two-channel NormWear request and a temperature Mantis request, with two and four candidates respectively; logits were `[2,4]` with the correct padding mask. `git diff --check` passed. NeuroKit emits one upstream `scipy.misc` deprecation warning; it does not affect these checks.

Checked on Python 3.12.12, PyTorch 2.14.1+cpu, NumPy 2.5.4, SciPy 1.18.1, NeuroKit 0.2.12, mantis-tsfm 1.1.0, sentence-transformers 5.7.0, transformers 4.57.6, and pytest 9.1.1. The test report is `artifacts/architecture-tests.xml`.

Run from the project root with its `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp cache/pytest-architecture-final --junitxml artifacts/architecture-tests.xml
```

## Retired implementation

The original `paya`, `tests`, `config`, `deploy`, `benchmarks`, and `examples` directories, plus the old `pyproject.toml`, are preserved at:

`C:\jev\retired\paya-20261019-architecture-v1`

The retained vendor encoder and BioLORD loader were copied back and connected to the new interfaces. Obsolete main-project callers, deployment scripts, and benchmark results are retired. Existing ignored datasets and old model files were left in place; the new configuration points to fresh dataset/cache/run paths and never auto-loads old decision artifacts.

## Review limits

- No real-dataset retraining, temperature fitting, accuracy/generalization testing, or latency benchmarking has started. The new router/head are untrained, so current output distributions do not represent validated confidence.
- Checks were run on CPU. GPU execution and peak-memory/latency budgets have not been measured.
- Preparation uses explicit generic NeuroKit low-pass filtering and modality-specific units/rates. Polyphase resampling uses SciPy's underlying operation with linear boundary extension to preserve physical levels. It does not silently apply NeuroKit's peak-detection filters or pass derived HR/HRV/QC tables into scoring.
- Temperature and raw EMG are unavailable to the NormWear branch; Mantis needs enough real coverage at its specified rate. No EMG-envelope route or third encoder was introduced.
- Free-form spatial/clinical question support, suitable lead/montage/coordinate conventions, acquisition bandwidth, routing quality, and representation sufficiency need task-level validation. The implementation does not claim that accepting a type establishes support for every question about it.
- Checkpoint, cache, and export contracts are new. Existing decision weights and calibration cannot be reused; the two selected signal encoders remain fixed.
