# SNN Ablation Pipeline

Config-driven PyTorch scaffold for four checkpoints:

1. `baseline`: LIF SNN with surrogate-gradient training.
2. `skip`: baseline plus Optuna-selected skip-connection settings.
3. `threshold`: skip architecture plus multi-threshold LIF neurons.
4. `full`: threshold model plus ANN-teacher distillation.

Install dependencies:

```bash
pip install -r requirements.txt
```

Colab with DVS128 Gesture stored in Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/nishant1479/snn-conn.git
%cd snn-conn
!python colab_runner.py install
!python colab_runner.py prepare-dvsgesture --drive-dataset-dir /content/drive/MyDrive/datasets/DVSGesture
```

The Drive folder should contain `ibmGestureTrain.tar.gz` and `ibmGestureTest.tar.gz`.

Example commands, when you are ready to train:

```bash
python train.py --config configs/default.yaml --stage baseline
python train.py --config configs/default.yaml --stage skip-search
python train.py --config configs/default.yaml --stage skip --init-checkpoint runs/snn_ablation/baseline/best.pt
python train.py --config configs/default.yaml --stage threshold --init-checkpoint runs/snn_ablation/skip/best.pt
python train.py --config configs/default.yaml --stage full --init-checkpoint runs/snn_ablation/threshold/best.pt
```

Results are appended to `runs/snn_ablation/ablation_results.csv`.

Temporal loss modes are controlled by `train.loss.method`:

- `mean_logits`: legacy behavior in this scaffold, averaging logits over time before cross entropy.
- `tet` or `original_tet`: uniform temporal loss, `(1/T) * sum_t CE(O_t, y)`.
- `gradnorm`: GradNorm-style adaptive temporal balancing, `sum_t w_t CE(O_t, y)`, using `train.loss.gradnorm.shared_parameters` to choose the shared layer for gradient norms.

Notes:

- `DVSGesture` is the default because Tonic provides an official train/test split. `CIFAR10-DVS` is also supported, but this scaffold creates a local random split because the dataset is commonly distributed without a canonical train/test split.
- Multi-threshold neurons output normalized multi-level spike values, so they can change activation scale. If accuracy drops, tune `model.multi_threshold.thresholds`, `train.lr`, and distillation weights rather than assuming the stacked method is beneficial.
- The skip search objective fine-tunes from a checkpoint when `optuna.pretrained_checkpoint` is set. Concatenation skips add projection layers, so checkpoint loading is intentionally non-strict.
