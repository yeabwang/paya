# NormWear source attribution

`normwear.py` and `layers.py` are modified versions of the NormWear encoder source
from `mosaic-laboratory/normwear`, pinned to revision
`c35a3d1ecda1cb971f1f89caaa3ee4b4c6ad8e84`:

- [Original normwear.py](https://huggingface.co/mosaic-laboratory/normwear/blob/c35a3d1ecda1cb971f1f89caaa3ee4b4c6ad8e84/normwear.py)
- [Original layers.py](https://huggingface.co/mosaic-laboratory/normwear/blob/c35a3d1ecda1cb971f1f89caaa3ee4b4c6ad8e84/layers.py)

Paya retains the waveform preprocessing, patch embedding and transformer
operations used by its signal encoder. Unused pretraining, reconstruction,
alignment and masking branches were removed. Patch-grid dimensions are computed
directly from convolution geometry.

The original copyright and reference notices are retained in both files.
The accompanying [LICENSE](LICENSE) is Apache-2.0, preserved from the
[official NormWear repository](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/blob/07517fcb13def8c89cb586128359cec02f86ec8d/LICENSE).
