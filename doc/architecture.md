# Architecture

Paya combines frozen SPOTR signal tokens, frozen BioLORD sentence embeddings and a trainable attention head. At width 128 with four attention heads, the decision head has **480,257 parameters**. Only this head is optimized.

```text
raw samples + type/rate -> physiological processing --+
processed samples ------> validation/passthrough -----+-> frozen SPOTR -> signal tokens
                                                                           |
question + candidates -> frozen BioLORD -> shared projection + roles         |
                                                   |                       |
                                     question/candidate self-attention      |
                                                   |                       |
                                     cross-attention to signal tokens <-----+
                                                   |
                                     shared candidate MLP -> mask -> softmax
```

## Input and processing

`model.predict(state, question, criteria)` accepts an aligned recording and 2–64 distinct, nonempty candidate strings. The question is a separate nonempty string. BioLORD's tokenizer rejects inputs exceeding its 128-token context, including special tokens.

Each `state.signals` entry contains finite one-dimensional `samples` and a `metadata` object:


| Field              | Meaning                                                            |
| -------------------- | -------------------------------------------------------------------- |
| `sensor_type`      | `ppg`, `ecg`, `eeg` or `ieeg`; determines processing for raw input |
| `sampling_rate_hz` | Positive native rate; used for validation and raw processing       |
| `duration_s`       | Consistent with sample count divided by rate                       |
| `signal_condition` | `raw` or `processed`; defaults to `raw`                            |

One recording has 1–128 channels and 2–12,800 samples per channel. All channels must share sample count, rate and start time. Supplied timestamps must describe uniform sampling. Channel order is preserved. Over-budget, nonfinite and misaligned inputs are rejected without silent truncation.

Raw channels receive the existing signal-specific physiological processing at their native rate. Processed channels receive no additional cleaning or resampling. SPOTR retains its pretrained internal normalization and padding in both cases. Signal type, rate and audit identifiers are not head features. No physical sensor metadata projection, fixed ending text or extra answer slot is present.

## Signal tokens

SPOTR Base has width 512, 12 transformer blocks and eight attention heads. For `[B,C,L]` samples it normalizes each channel, pads internally to `Lp = 200 * ceil(L/200)`, and constructs `N = Lp/100` temporal positions. Channel and temporal summaries, plus the learned global token, pass through the pretrained transformer unchanged.

The wrapper retains **all `[B,1+C+N,512]` outputs**:

- Token 0: global representation.
- Next `C` tokens: channel summaries.
- Next `N` tokens: temporal summaries, in encoder order.

The global and channel tokens are valid. Temporal position `j` is valid when `100*j < L`; positions covering only added padding are masked at the decision head. This preserves the screened implementation, including the pretrained encoder's internal attention to its padded representation.

For a single-channel, 1,250-sample recording, SPOTR produces **16 tokens: one global, one channel and 14 temporal**. The final temporal position is fully padded, leaving 15 valid head keys. At 125 Hz the nominal patch stride is 0.8 seconds. These tokens are contextualized summaries, not isolated waveform crops; partial patches and convolutional/attention context cross nominal boundaries. No fixed rate is assumed.

## Text projection and roles

BioLORD encodes the question and each candidate independently into normalized 768-dimensional vectors. The same `Linear(768,128) -> LayerNorm(128)` projects all text. A learned question role and a shared candidate role are added. There are no candidate-position embeddings.

Signal tokens use `LayerNorm(512) -> Linear(512,128)`, forming the cross-attention memory. The QA sequence is `[question, candidate_1, ..., candidate_K]`. Padded candidate slots are masked as self-attention keys and in final logits.

## Attention blocks

The QA sequence first passes through self-attention. It then supplies queries to cross-attention whose keys and values are the projected signal tokens. The signal memory is read without being updated by the head.

Each block contains:

1. LayerNorm on the query sequence.
2. Four-head attention, dropout and residual addition.
3. `LayerNorm -> Linear(128,256) -> GELU -> Dropout -> Linear(256,128)`, followed by dropout and a residual addition.

For self-attention, normalized QA vectors supply queries, keys and values. For cross-attention, normalized QA vectors query the signal memory. A final LayerNorm follows both blocks. Self-attention permits candidates to affect one another's representations; it does not guarantee that particular physiological relationships are learned.

## Shared scoring and probabilities

Split the resulting QA sequence into question `q` and candidates `c_i`. Each candidate uses the same scorer:

```text
score_i = Linear(128,1)(Dropout(GELU(Linear(384,128)([q, c_i, q*c_i]))))
probabilities = softmax(masked_scores / temperature)
```

The Hadamard product `q*c_i` supplies feature interactions; its dimensions have no predefined meaning. The scorer supports new candidate sets without resizing a task-specific classifier. In evaluation mode, permuting candidates permutes their outputs within floating-point tolerance. Adding or changing a candidate can alter all scores because self-attention uses the supplied set.

The public result contains the selected `choice`, its `confidence`, and choice-keyed `probabilities`. Exact ties use lexicographic candidate text. Uncalibrated temperature is 1. A fitted temperature is bound to exact head weights and independent calibration outcomes. Fine-tuning clears that binding.

## Numerical and storage boundaries

Both encoders remain frozen in evaluation mode during training. Frozen signal extraction uses FP32, one recording per encoder call; repeated recordings are deduplicated within a batch. The head uses FP32. Batches pad token sequences with a separate validity mask.

File and SQLite caches store complete tokens and masks with checksums. Bindings include frozen encoder assets, representation and preparation, independently of head settings. A global-vector cache is not an input to this implementation. Checkpoints contain head weights, model identities, source and weight hashes, optional recovery state, and optional calibration provenance. Exports include pinned encoder assets, supported code, configuration and file checksums.
