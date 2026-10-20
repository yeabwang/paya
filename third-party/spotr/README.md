---
license: mit
tags:
- physiological-signals
- self-supervised-learning
- foundation-model
- representation-learning
---

# SPOTR

SPOTR (**Spatio-temporal Pooling One-Token Reconstruction**) is a universal self-supervised learning framework for physiological signals, including electroencephalography (EEG), intracranial EEG (iEEG), electrocardiography (ECG), and photoplethysmography (PPG).

SPOTR compresses each physiological waveform into a single global representation and reconstructs the original signal using only this representation. The single-token information bottleneck encourages the model to learn compact and transferable global features instead of relying on local temporal continuity or cross-channel redundancy.

- **Paper:** *SPOTR: Spatio-temporal Pooling One-Token Reconstruction for Universal Physiological Signal Self-supervised Learning*
- **Conference:** IJCAI 2026
- **Code:** [https://github.com/5GYYYYY/SPOTR](https://github.com/5GYYYYY/SPOTR)
- **License:** MIT

## Model Description

SPOTR consists of three main components:

1. **ST Compactor:** Compresses the input waveform into compact temporal and spatial token sequences, reducing the sequence length from \(C \times N\) to \(C + N\).
2. **Latent Aggregator:** Aggregates the compact spatiotemporal tokens into a single global representation.
3. **Latent Renderer:** Reconstructs the input waveform from mask tokens conditioned only on the single global representation.

The pretrained representation can be used for linear probing, few-shot learning, full fine-tuning, and physiological signal feature extraction.

## Supported Signal Modalities

- Electroencephalography (EEG)
- Intracranial electroencephalography (iEEG)
- Electrocardiography (ECG)
- Photoplethysmography (PPG)

SPOTR supports inputs with different numbers of signal channels.

## Pretraining Data

SPOTR was pretrained on 20 physiological signal datasets:

- 6 EEG datasets
- 2 iEEG datasets
- 8 ECG datasets
- 3 PPG datasets
- 1 multimodal physiological signal dataset

The complete pretraining collection contains more than 17 million signal samples from over 450,000 subjects.

Please refer to the paper and repository for the complete dataset descriptions and preprocessing details. Users are responsible for obtaining the original datasets and complying with their respective licenses and data-use agreements.

## Preprocessing

The evaluation experiments reported in the paper used:

- A sampling frequency of 200 Hz
- A 50 Hz or 60 Hz notch filter, depending on the acquisition region
- Subject-independent training, validation, and test splits

Inputs should follow the preprocessing and tensor format specified in the official SPOTR repository. Differences in sampling frequency, filtering, normalization, channel configuration, or segment duration may affect model performance.

## Intended Use

SPOTR is intended for research on physiological signal representation learning, including:

- Physiological signal feature extraction
- Linear probing
- Few-shot classification
- Full-parameter fine-tuning
- Cross-dataset and cross-modality transfer
- Development of EEG, iEEG, ECG, and PPG analysis systems

## Out-of-Scope Use

SPOTR is not a clinically validated diagnostic system. The pretrained model should not be used as the sole basis for diagnosis, treatment, patient monitoring, or other clinical decisions.

Performance may vary across populations, acquisition devices, institutions, channel configurations, sampling conditions, and preprocessing pipelines. Independent validation is required before any clinical or safety-critical application.

## Evaluation

SPOTR was evaluated on 17 downstream datasets across EEG, iEEG, ECG, and PPG.

Under linear probing, SPOTR achieved the following average ROC-AUC results:

| Modality | Average ROC-AUC |
|:--|--:|
| iEEG | 88.22% |
| EEG | 88.10% |
| ECG | 87.39% |
| PPG | 64.14% |
| Overall across 17 datasets | 80.86% |

SPOTR exceeded the general-purpose time-series foundation model MOMENT by 19.11 percentage points in overall mean ROC-AUC under linear probing.

In the efficiency evaluation, SPOTR achieved approximately:

- 78.1% lower median inference latency
- 78.3% lower p95 inference latency
- 206.3% higher throughput
- 51.8% lower peak GPU memory

These efficiency results were measured against MOMENT under the experimental settings described in the paper and should not be interpreted as universal hardware-independent guarantees.

## Model Sizes

The paper evaluates three SPOTR variants:

| Variant | Parameters |
|:--|--:|
| SPOTR-Small | 3.69M |
| SPOTR-Medium | 13.93M |
| SPOTR-Base | 62.47M |

This repository provides the SPOTR-Base checkpoint.

## Usage

The model implementation, preprocessing pipeline, downstream adaptation code, and usage instructions are available in the official repository:

```bash
git clone https://github.com/5GYYYYY/SPOTR.git
cd SPOTR
```

Please ensure that the model configuration used in the code matches the uploaded checkpoint variant.

