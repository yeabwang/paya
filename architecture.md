# Architecture

Paya is a non-autoregressive System 1 decision model for physiological signals. It scores the supplied candidate answers in one forward pass.

```text
Signal + metadata → validation / raw-signal cleaning → frozen SPOTR → 512-D signal vector
Question and candidates → frozen BioLORD-2023 → separate 768-D text vectors
Signal and text vectors → MLP fusion head → candidate logits → softmax probabilities
```

The signal encoder processes the complete aligned recording once. Channels retain their declared native sampling rate. Raw channels receive modality-specific NeuroKit cleaning; processed channels bypass it. SPOTR applies per-channel z-score normalization and internal padding. The question and each candidate are encoded separately; metadata is used for preparation, not as head features.

The trainable head has **264,065 parameters**, hidden width 128 and dropout 0.1. A shared text projection maps question/candidate embeddings to 128 dimensions. A signal projection maps the global signal vector to 128 dimensions. Two MLP stages combine concatenated vectors and element-wise (Hadamard) products: first signal/question, then context/candidate. A shared scalar scorer produces one logit per candidate. The encoders remain frozen.

The public API accepts `state`, `question` and at least two distinct `criteria`; it returns `choice`, `confidence` and choice-keyed `probabilities`. Confidence equals the selected probability. Inputs require sensor type, sampling rate and duration. Aligned recordings support 1–128 channels and 2–12,800 samples per channel. Accepted sensor types are PPG, ECG, EEG and iEEG; the reported trained-model results cover PPG only.

The release retains the selected MLP implementation. Historical attention heads and experiment tooling are archived outside the tracked source. The serialized head kind remains `interaction` for checkpoint identity compatibility; the model computation and trained tensors are unchanged.
