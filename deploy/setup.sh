#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp
PROJECT=$BASE/paya
PY=$BASE/paya-env/bin/python
cd "$PROJECT"
mkdir -p runs cache third-party benchmarks/results
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
export HF_HOME="$PROJECT/cache/huggingface" TOKENIZERS_PARALLELISM=false
if [ ! -x "$PY" ]; then /root/miniconda3/bin/python -m venv --system-site-packages "$BASE/paya-env"; fi
/root/miniconda3/bin/python -c "import torch; print('torch=='+torch.__version__)" > "$BASE/torch-constraint.txt"
"$PY" -m pip install --quiet --disable-pip-version-check -c "$BASE/torch-constraint.txt" \
  transformers==4.57.6 sentence-transformers==5.1.1 'safetensors>=0.5' 'scikit-learn>=1.5' \
  'scipy>=1.13' PyYAML pytest wandb==0.30.0 python-dotenv==1.2.4
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available()
print({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)},flush=True)
PY
"$PY" -m pytest -q -p no:cacheprovider --basetemp="$PROJECT/.test-tmp"
"$PY" -u -m paya.validate --data data/body-signals-v1 --output benchmarks/results/data-validation.json
"$PY" -u -m paya.text --texts data/body-signals-v1/texts.json --output cache/paya-text.pt --device cuda
if [ ! -f cache/paya-evidence.json ]; then
  "$PY" -u -m paya.features --data data/body-signals-v1 --assets third-party/normwear --output cache/paya-evidence.json --device cuda
fi
"$PY" -u -m paya.train --config config/paya.yaml --run-dir runs/diagnostic --max-steps 10
"$PY" -u -m paya.train --config config/paya.yaml --run-dir runs/paya >runs/baseline.log 2>&1 &
baseline=$!
"$PY" -u -m paya.train --config config/transfer-epilepsy.yaml --run-dir runs/transfer-epilepsy >runs/transfer-epilepsy.log 2>&1 &
epilepsy=$!
"$PY" -u -m paya.train --config config/transfer-fatigue.yaml --run-dir runs/transfer-fatigue >runs/transfer-fatigue.log 2>&1 &
fatigue=$!
failed=0
for job in "$baseline" "$epilepsy" "$fatigue"; do wait "$job" || failed=1; done
if [ "$failed" -ne 0 ]; then printf 'At least one training run failed; inspect runs/*.log\n'; exit 1; fi
"$PY" -u -m paya.evaluate_cached --config config/transfer-epilepsy.yaml --checkpoint runs/transfer-epilepsy/best.pt --split heldout_domain --only epilepsy --families eeg_delta_theta --output runs/transfer-epilepsy/heldout-domain.jsonl
"$PY" -u -m paya.evaluate_cached --config config/transfer-fatigue.yaml --checkpoint runs/transfer-fatigue/best.pt --split heldout_domain --only drive_fatigue --families eeg_delta_theta --output runs/transfer-fatigue/heldout-domain.jsonl
# All training is complete before measuring the deployed artifact on the idle GPU.
"$PY" -u -m deploy.finalize --config config/paya.yaml --run-dir runs/paya --output model \
  --transfer epilepsy runs/transfer-epilepsy --transfer drive_fatigue runs/transfer-fatigue \
  --transfer-family eeg_delta_theta
printf 'Training and development assessment complete\n'
