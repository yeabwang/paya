# Paya architecture

Paya uses a two-expert top-1 router and one shared decision head. NormWear, MantisV2 and BioLORD are frozen; the router and decision head are trainable. Preprocessing and encoder-output adapters have no learned parameters.

```mermaid
flowchart TD
    S[Signal + metadata] --> P[Native-rate NeuroKit cleaning]
    P --> R[Trainable router]
    Q[Question] --> T[Frozen BioLORD]
    C[Ordered choices] --> T
    T -->|Question vector| R
    R --> F[Format the selected encoder input]
    F --> E[Frozen NormWear or MantisV2]
    E --> A[Fixed width adapter]
    A --> H[Trainable shared decision head]
    T -->|Separate text vectors| H
    H --> O[Choice + ordered probabilities + confidence]
    classDef frozen fill:#acdafa,stroke:#404040,color:#111;
    classDef trainable fill:#b2efb7,stroke:#404040,color:#111;
    class T,E frozen;
    class R,H trainable;
```

The public arguments are `state`, `question` and ordered `criteria`. Metadata determines native cleaning and encoder eligibility. One cleaned waveform per physical channel reaches the router; resampling and framing run after selection. Neither learned component receives a metadata vector. Quality observations and dataset identifiers remain audit fields. See [preprocessing](preprocessing.md) and [training](training.md).

## Dimensions

`B` is batch size; `K` is the maximum candidate count; `S = K+1` counts the question and candidates. `R_N` and `R_M` count real channel/frame pairs after formatting for the selected encoder. `M` is the maximum encoded signal-token count per request. Padding masks identify real candidates and encoded tokens; padding never enters a frozen signal encoder.

## Frozen encoders

### NormWear

Preparation produces 65 Hz, six-second frames. `W` is frame count and `C` is real channel count.

| Operation | Input | Output |
| --- | --- | --- |
| Waveform/difference Ricker transform | `[W,C,390]` | `[W*C,3,389,65]` |
| 9×5 strided patch projection | `[W*C,3,389,65]` | `[W*C,559,768]` |
| Append CLS and frozen positions | `[W*C,559,768]` | `[W*C,560,768]` |
| 12 transformer blocks and final LayerNorm | `[W*C,560,768]` | `[W*C,560,768]` |
| Pool 43 time rows to 8, retaining 13 scale bins and CLS | `[W,C,560,768]` | `[W,C,105,768]` |
| Pack and apply identity adapter | `[W,C,105,768]` | `[W*C*105,768]` |

Each block uses pre-LayerNorm, self-attention with 12 heads of width 64 (`768→2304` QKV, `768→768` output), and a GELU feed-forward layer `768→3072→768`, with residual additions. Every other block also applies these operations across the `C` channel CLS tokens. Pooling compresses temporal detail; it is part of the fixed representation.

### MantisV2

Each real frame contains 512 measurements. The selected output uses transformer layers 0–2 and concatenates CLS with mean patch embeddings.

| Operation | Input | Output |
| --- | --- | --- |
| Waveform/difference/statistics token generator | `[R_M,1,512]` | `[R_M,32,256]` |
| Append CLS | `[R_M,32,256]` | `[R_M,33,256]` |
| Three transformer blocks | `[R_M,33,256]` | `[R_M,33,256]` |
| Concatenate CLS and mean patch embedding | `[R_M,33,256]` | `[R_M,512]` |
| Append 256 zeros | `[R_M,512]` | `[R_M,768]` |

Each block uses RMSNorm, rotary self-attention with eight heads of width 32 (`256→768` QKV, `256→256` output), and RMSNorm followed by a gated feed-forward layer `256→1024→512→256` with SwiGLU and residual additions. The token generator retains a path for raw patch means and standard deviations. Zero-padding preserves every output coordinate; it does not align the encoders' feature meanings. Learned cross-attention handles their shared 768-wide format.

### BioLORD-2023

Each question and choice is independently tokenized and encoded. For `N` uncached strings and token length `L≤128`, MPNet maps `[N,L]` IDs/masks to `[N,L,768]`. Its 12 layers use 12 self-attention heads of width 64 and `768→3072→768` feed-forward layers. Masked mean pooling and L2 normalization return `[N,768]`; vectors are cached on CPU. Longer text is rejected rather than truncated.

## Trainable router

Purpose: choose an eligible encoder from native-rate cleaned waveforms and the question. Each channel is a one-dimensional array `[N_c]`. Equal-length channels are batched as `[G,1,N]`; unequal lengths are processed separately without padding, resampling or framing.

| Layer | Input | Output |
| --- | --- | --- |
| Conv1d, kernel 9, stride 2, padding 4; GELU | `[G,1,N]` | `[G,16,ceil(N/2)]` |
| Conv1d, kernel 5, stride 2, padding 2; GELU | Above | `[G,32,ceil(N/4)]` |
| Temporal mean | `[G,32,ceil(N/4)]` | `[G,32]` |
| Mean across physical channels belonging to each request | Per-channel 32-dimensional outputs and owner indices | `[B,32]` |
| Question linear + GELU | `[B,768]` | `[B,128]` |
| Concatenate waveform/question outputs | `[B,32]` and `[B,128]` | `[B,160]` |
| Linear + GELU | `[B,160]` | `[B,64]` |
| Expert logits; compatibility mask; argmax | `[B,64]` | `[B,2]` logits; `[B]` selected encoder |

`G` is the number of channels in a convolution batch and `N` is their native sample count. Convolution weights are shared across all channels. For one six-second ECG at 500 Hz, the convolution shapes are `[1,1,3000]→[1,16,1500]→[1,32,750]`, followed by temporal mean `[1,32]`.

Compatibility is calculated from declared signal types, coverage and budgets, without constructing encoder inputs. Unavailable encoder logits are masked to negative infinity. After argmax, only the selected encoder's resampling and framing run, followed by that frozen encoder. Hard routing has no answer-loss gradient through argmax; training supplies detached expert-loss routing targets. Choices, quality measurements and metadata vectors are not CNN/MLP features.

## Trainable shared decision head

Purpose: compare all choices against the encoded signal and question in one computation. Pack selected expert outputs as `[B,M,768]`. Default text-state width is 256 with four attention heads of width 64, two residual blocks and 2,829,825 trainable parameters.

| Layer | Input | Output |
| --- | --- | --- |
| Stack the separate question and candidates | `[B,768]`, `[B,K,768]` | `[B,S,768]` |
| Shared text linear + LayerNorm | `[B,S,768]` | `[B,S,256]` |
| Add question/candidate role vectors | `[B,S,256]`, `[2,256]` | `[B,S,256]` |
| Pre-LayerNorm, bidirectional self-attention + residual | `[B,S,256]`; Q/K/V `[B,4,S,64]` | `[B,S,256]` |
| Pre-LayerNorm and cross-attention query projection | `[B,S,256]` | Q `[B,4,S,64]` |
| Signal key/value projections, 768→256 each | `[B,M,768]` | K/V `[B,4,M,64]` |
| Add fixed sinusoidal relative-time encoding to keys | `[B,M]` relative times | `[B,M,256]` encoding, reshaped to `[B,4,M,64]` |
| Cross-attention, output projection and residual | Q/K/V; attention scores `[B,4,S,M]` | `[B,S,256]` |
| Pre-LayerNorm, linear, GELU, dropout, linear + residual | `[B,S,256]` | `[B,S,1024]→[B,S,256]` |
| Repeat the residual block once | `[B,S,256]` | `[B,S,256]` |
| Final LayerNorm and shared candidate linear scorer | Candidate states `[B,K,256]` | `[B,K,1]→[B,K]` logits |
| Masked softmax of logits divided by fitted temperature | `[B,K]` | `[B,K]` probabilities |

The default dropout is 0.1. Candidates share one role vector and scorer, with no candidate-position embeddings, preserving candidate permutation equivariance. Each signal embedding carries one relative time in `[0,1]`, used only in the key positional encoding. No scale coordinate, CLS/frame indicator or sensor label is appended. Each block costs `O(S²+S*M)` attention work rather than adding signal-token self-attention of `O(M²)`.

The returned choice is the highest-probability criterion; ties use the criterion text as a deterministic key. Probabilities retain the supplied order. Confidence equals the chosen probability. Calibration measures groups of predictions and does not guarantee an individual answer is correct.

## Identity and verification

[paya/provenance.py](../paya/provenance.py) defines Paya `v1` contracts and pinned upstream identities. Cleaned states, feature caches, checkpoints, calibration and exported bundles validate the relevant identity and checksums. Package metadata reads the same central version. Upstream model names and revisions are unchanged.

Unit tests cover validation, native processing, routing, masks, permutations, gradients, caching, checkpoint/calibration bindings and training tracking. Integration tests exercise the actual pinned encoders and output shapes. Generalization, outcome calibration and latency must be measured after shared training, on independent domains and recordings.

Sources: [NormWear](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear), [MantisV2](https://github.com/vfeofanov/mantis), [BioLORD](https://huggingface.co/FremyCompany/BioLORD-2023), [NeuroKit](https://neuropsychology.github.io/NeuroKit/).
