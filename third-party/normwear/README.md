---
license: apache-2.0
tags:
- machine-learning
- deep-learning
- signal-processing
- digital-healthcare
---

<div align="center">
  
# 🚀 NormWear: A Foundation Model for Multivariate Wearable Sensing of Physiological Signals.

[![ACM Journal](https://img.shields.io/static/v1?label=ACM&message=10.1145/3803808&color=0B3D91&logo=acm)](https://dl.acm.org/doi/10.1145/3803808)
[![github-repo](https://img.shields.io/badge/GitHub-Code_Repository-black?logo=github&logoColor=white)](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)
[![ckpt-release](https://img.shields.io/badge/Github_Release-Model_Checkpoint-black?logo=github&logoColor=white)](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha)
[![downstream](https://img.shields.io/badge/Google_Drive-Downstream_Datasets-blue?logo=googledrive&logoColor=fff)](https://drive.google.com/file/d/1Mojf_iby8FnUydogwUE-b321fB1V6SGK/view?usp=sharing)
[![pretrain](https://img.shields.io/badge/Google_Drive-Pretrain_Datasets-blue?logo=googledrive&logoColor=fff)](https://drive.google.com/file/d/1WBlyweezkYm16PR3UFrO85XrZCKjqWmP/view?usp=sharing)

</div>

> [!IMPORTANT]
> 🚀 [News! Mar 2026] We are pleased to announce that our paper has been accepted by ACM Transactions on Computing for Healthcare (HEALTH) and is now In Press!


## ✨ Introduction
This is the official implementation of the paper [Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals](https://arxiv.org/abs/2412.09758).

<img src="./framework_overview.png" alt="Overview" width="1000"/> 

Time-series foundation models have the ability to run inference, mainly forecasting, on any type of time series data, thanks to the informative representations comprising waveform features. Wearable sensing data, on the other hand, contain more variability in both patterns and frequency bands of interest and generally emphasize more on the ability to infer healthcare-related outcomes. The main challenge of crafting a foundation model for wearable sensing physiological signals is to learn generalizable representations that support efficient adaptation across heterogeneous sensing configurations and applications. In this work, we propose NormWear, a step toward such a foundation model, aiming to extract generalized and informative wearable sensing representations. NormWear has been pretrained on a large set of physiological signals, including PPG, ECG, EEG, GSR, and IMU, from various public resources. For a holistic assessment, we perform downstream evaluation on 11 public wearable sensing datasets, spanning 18 applications in the areas of mental health, body state inference, vital sign estimations, and disease risk evaluations. We demonstrate that NormWear achieves a better performance improvement over competitive SoTA baselines with different modeling strategies. In addition, leveraging a novel representation-alignment-match-based method, we align physiological signals embeddings with text embeddings. This alignment enables our proposed foundation model to perform zero-shot inference, allowing it to generalize to previously unseen wearable signal-based health applications.

## 📈 Usage

Download: 

```sh
# Clone the repository
git clone git@github.com:Mobile-Sensing-and-UbiComp-Laboratory/NormWear.git
```
The python libraries dependencies are specified in `NormWear/dependencies.txt`, with an example bash script `NormWear/config_env.sh`. 

The pretrained model checkpoint can be found in [Release](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha).

### Extracting Encoder Embeddings

An example showing how to get signal embedding using NormWear:

```python
import torch
torch.uint64 = torch.int64
torch.uint32 = torch.int32
torch.uint16 = torch.int16

from transformers import AutoModel

model = AutoModel.from_pretrained(
    "mosaic-laboratory/normwear",
    trust_remote_code=True
)

# demo input/output
batch_num = 2 # bn
num_channel = 3 # nvar
sequence_length = 256 # L
x = torch.rand(batch_num, num_channel, sequence_length)

with torch.no_grad():
  outpack = model(x, return_spec=True, return_enc_out=True, 
                  return_dec_out=False, zero_shot_input_pack=None)

# log
for k in outpack:
  print(k, outpack[k].shape)
# above should output:
# spec torch.Size([2, 3, 3, 255, 65]), # bn, nvar, cwt_channels, L, num_scales
# enc_out torch.Size([2, 3, 365, 768]) # bn, nvar, num_patches, embed_size
# NOTE: the embedding of the 1st patch in enc_out is the [CLS] embedding
```

### Zero shot inference

```python
# NOTE: model loading same as above from Hugging Face for extracting embedding. 

### generate data ###############################################################
sampling_rate = 64
x = torch.rand(2, 3, sampling_rate*2).to(device) # test example: 2 samples, 3 sensor, sequence length of 2 seconds

# setup question and options
task_of_interest = [
    "Is the person doing good?"
]

options = [
    "The person is doing so so",
    "The person is under stress.",
    "The person is having fun.",
]

### Inference ###################################################################
# construct language model. We leverage a model pretrained on clinical related corpus. 
# Reference https://huggingface.co/muzammil-eds/tinyllama-2.5T-Clinical-v2
tokenizer = AutoTokenizer.from_pretrained("muzammil-eds/tinyllama-2.5T-Clinical-v2")
nlp_model = AutoModelForCausalLM.from_pretrained("muzammil-eds/tinyllama-2.5T-Clinical-v2")

with torch.no_grad():
  # text encoding
  inputs = tokenizer(task_of_interest+options, padding=True, return_tensors="pt").to(nlp_model.device)
  txt_embed = nlp_model(**inputs, output_hidden_states=True) # [4, 2048]
  query_embed, option_embed = txt_embed[:1, :], txt_embed[1:, :]

  # NormWear inference
  outpack = model(x, zero_shot_input_pack=(query_embed, option_embed, options))


# log
for k in outpack:
    if type(outpack[k]) is list:
        print(outpack[k])
        continue
    print(k, outpack[k].shape)
# above should output:
signal_embed torch.Size([2, 3, 365, 768]) # bn, nvar, num_patches, embed_size
msitf_fuse_out torch.Size([2, 2048]) # bn, text_embed_size
pred_prob torch.Size([2, 3]) # bn, len(options)
pred (2,) # bn
['The person is doing so so', 'The person is doing so so'] # selected choices
```

> [!TIP]  
> For the best performance, conduct prompt engineering on the query could be a good practice. 

## 🔥 Pre-training

> [!NOTE]
> The instruction below is for working with pipeline from Github repo [https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)

To run the evaluation on the downstream datasets, run the following command:
```sh
python3 -m NormWear.pretrain_main
```


| Required Parameter         | Type      | Default   | Description                                                     |
|---------------------|-----------|-----------|-----------------------------------------------------------------------|
| `--batch_size`      | `<int>`  | `4` | Number of input samples for each round of forward and backpropagation.       |
| `--epochs`          | `<int>`  | `400` | Number of epochs.                        |
| `--blr`             | `<float>`  | `1e-3` | base learning rate: absolute_lr = base_lr * total_batch_size / 256 |
| `--warmup_epochs`   | `<int>`  | `10` | Number of epochs to increase learning rate from a small number to `blr`.  |
| `--target_len`      | `<int>`  | `388` | Sequence length of the reconstructed time series |
| `--weight_decay`   | `<float>`  | `5e-2` | Regularization coefficient of L2 penalty on model parameters. |
| `--mask_scheme`   | `<str>`  | `random` | Masking scheme for pretext task (choose from 'random' and 'struc')  |
| `--mask__prob`   | `<float>`  | `0.8` | Masking ratio for random masking  |
| `--mask_t_prob`   | `<float>`  | `0.6` | Masking ratio along time axis of the scalogram. |
| `--mask_f_prob`   | `<float>`  | `0.5` | Masking ratio along scale axis of the scalogram.  |
| `--remark`   | `<string>`  | `""` | Name to mark the current experimental trail. |

An example bash command would be:
```sh
torchrun --nproc_per_node=4 --master_port=12345 -m NormWear.pretrain_main --batch_size 256 --epochs 100 --warmup_epochs 4 --remark test_run
```


## ❄️ Downstream Evaluation

To run the evaluation on the downstream datasets, run the following command:
```sh
python3 -m NormWear.downstream_main
```


| Required Parameter         | Type      | Default   | Description                                                                 |
|--------------------------|-----------|-----------|-----------------------------------------------------------------------------|
| `--model_name`           | `<string>`  | `normwear` | Supported models are [stats, chronos, clap, tfc, normwear]                                 |
| `--model_weight_dir`     | `<string>`  | `""` | Path to the model checkpoint, only `normwear` need this parameter.                           |
| `--group`                 | `<int>`  | `0` | Run a group of downstream tasks. The group can be customized in `NormWear/downstream_main.py` |
| `--data_path`   | `<string>`  | `../data` | Root path for where the downstream data is placed. |
| `--num_runs`   | `<int>`  | `1` | Number of repetition for running the evaluation of each task. |
| `--prepare_embed`   | `<int>`  | `1` | Run the inference and save the embeddings before sending them to evaluation. Set to `1` to overwrite previous saved embedding if there are any. |
| `--remark`   | `<string>`  | `""` | Name to mark the current experimental trail. By default, all the embeddings are stored under a subfolder named as `--model_name` under the folder of each downstream data. |

An example bash command would be:
```sh
CUDA_VISIBLE_DEVICES=0 python3 -m NormWear.downstream_main --model_name normwear --model_weight_dir data/results/normwear_last_checkpoint-15470-correct.pth --group 0 --data_path ../data --num_runs 1 --prepare_embed 1 --remark test_run
```

#### :floppy_disk: The processed clean downstream datasets can be downloaded from [here](https://drive.google.com/file/d/1Mojf_iby8FnUydogwUE-b321fB1V6SGK/view?usp=sharing). 

### ✏️ For adding a downstream dataset, please following the format:
```bash
name_of_new_dataset/
├── sample_for_downstream/
│   ├── name_of_sample_0.pkl
│   ├── name_of_sample_1.pkl
│   ├── name_of_sample_2.pkl
│   └── ...
└── train_test_split.json
```
where the content in each file requires content in the following format:
```python
# data content in name_of_sample_i.pkl
{
    "uid": "", # subject ID identifier
    "data": np.array(sensor_data_here).astype(np.float16), # float numpy array with shape of [num_channels, sequence_length]
    "sampling_rate": 64, # put the correct sampling rate of the sensor signal here. 
    "label": [ # label for each task on this dataset. 'class' for classification and 'reg' for regression.
        {"class": label}, 
        {"reg": label},
        ...
    ]
}

# data content in train_test_split.json
{
  'train': ["name_of_sample_i.pkl", "name_of_sample_j.pkl", "name_of_sample_k.pkl", ...],
  'test': ["name_of_sample_l.pkl", "name_of_sample_m.pkl", "name_of_sample_n.pkl", ...]
}
```

### 🔥 Finetuning model for downstream datasets
To run the evaluation on the downstream datasets, run the following command:
```sh
python -m main_finetune --ds_name <dataset_name> --checkpoint <weight_path>
```
For multiple gpus, run:
```sh
torchrun --nproc_per_node=<number of device> main_finetune.py --world_size <number of device> --ds_name <dataset_name> --checkpoint <weight_path>
```

## 📈 Summary of Results

<img src="./main_results.png" alt="Overview" width="1000"/> 

## 📝 Citation

If you find NormWear model useful for your research, please consider citing the associated [paper](https://dl.acm.org/doi/10.1145/3803808). 

```
@article{10.1145/3803808,
  author = {Luo, Yunfei and Chen, Yuliang and Salekin, Asif and Rahman, Tauhidur},
  title = {Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals},
  year = {2026},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3803808},
  doi = {10.1145/3803808},
  note = {Just Accepted},
  journal = {ACM Trans. Comput. Healthcare},
  month = mar,
  keywords = {Machine Learning, Deep Learning, Digital Health, Time Series Modeling, Signal Processing}
}
```

## 📃 License
This project is licensed under the [Apache License 2.0](LICENSE).