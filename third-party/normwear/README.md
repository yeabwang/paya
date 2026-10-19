# NormWear inference assets

Paya loads `config.json` and the encoder-only `model.safetensors`. `source.json` records the pinned upstream checkpoint and local asset checksums.

The encoder implementation lives in [paya/vendor/normwear.py](../../paya/vendor/normwear.py), with its transformer layers in [paya/vendor/layers.py](../../paya/vendor/layers.py). Upstream training scripts and baselines are not required.

See [source attribution](../../paya/vendor/SOURCE.md) and the retained [Apache-2.0 license](LICENSE).
