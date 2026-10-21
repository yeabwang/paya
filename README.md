# Paya

A non-autoregressive System 1 decision model for physiological signals.

The selected model combines frozen signal/text encoders with a trained MLP fusion head. It scores supplied candidates directly and returns the selected choice, confidence and candidate probabilities.

## Use

Use Python 3.12 for the verified model runtime, then install Paya:

```bash
python -m pip install -r requirements-model.txt
python -m pip install -e ".[test]"
```

Download the private [model bundle](https://huggingface.co/yeabwang/Paya) to `models/export/` using your Hugging Face account, then:

```python
from paya.inference import load_model

model = load_model("models/export")
state = {"signals": [{"samples": ppg_samples, "metadata": {
    "sensor_type": "ppg", "sampling_rate_hz": 125,
    "duration_s": 10, "signal_condition": "processed",
}}]}
result = model.predict(state, "What is the heart rate category?",
                       ["bradycardia", "normal", "tachycardia"])
```

## Documentation

- [Architecture](architecture.md)
- [Experiments](experiments.md)
- [Fine-tuning](finetune.md)
- [Results and figures](results.md)
- [Model card](models/model.card.md)

Supported code is in `paya/`, tests in `tests/`, and the selected configuration in `config/paya.yaml`. Historical implementations and experiment tools are in ignored `retired/`. Checkpoints, local credentials, datasets and caches are not tracked.

```bash
python -m pytest -q -m "not integration"
python -m pytest -q -m integration
```

Reported results cover PPG; acceptance of other supported signal types does not establish their predictive accuracy.
