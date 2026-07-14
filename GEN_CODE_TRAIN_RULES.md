# GEN_CODE_TRAIN_RULES.md

> Bộ rule bắt buộc dành cho AI coding agent khi xây dựng, chỉnh sửa và duy trì pipeline training Machine Learning/Deep Learning trong repository này. Pipeline phải chạy được trên **Kaggle Notebook với 2x NVIDIA Tesla T4**.

---

## 1. Repository Architecture Rules

Cấu trúc repo bắt buộc:

```
project/
├── README.md
├── TRAINING_RULES.md
├── KAGGLE_TRAIN_RULES.md
├── requirements.txt
├── environment.yml
├── configs/
│   ├── train.yaml
│   └── eval.yaml
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── logger.py
│   └── utils.py
├── checkpoints/
├── logs/
├── outputs/
└── notebooks/
    └── kaggle_train.ipynb
```

**Rule:**
- Notebook **không được** chứa training logic.
- Toàn bộ model, dataset, training loop phải nằm trong `src/`.
- Notebook chỉ là **execution controller** (gọi script, không định nghĩa logic).

---

## 2. Kaggle Environment Rules

**Target:**
- Kaggle Notebook
- NVIDIA Tesla T4 x2
- CUDA enabled
- Python >= 3.10

**Yêu cầu cài đặt:**
- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `pyyaml`
- `tqdm`
- `tensorboard`
- `wandb` (optional)

**Notebook phải có các bước:**
1. `git clone` repository
2. `pip install -r requirements.txt`
3. Lưu lại environment bằng `pip freeze`

```bash
pip freeze > outputs/environment.txt
```

---

## 3. Configuration Management Rules

Tất cả hyperparameter **KHÔNG** được hardcode trong code.

Mọi thay đổi phải thông qua:
- `configs/train.yaml`
- `configs/eval.yaml`

Notebook phải có **một khu vực duy nhất** để chỉnh config:

```python
TRAIN_CONFIG = {
    "experiment_name": "exp001",
    "model_name": "model_name",
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "gradient_accumulation_steps": 8,
    "mixed_precision": "fp16",
    "save_steps": 500,
    "eval_steps": 500,
    "seed": 42
}
```

Sau đó notebook phải tự động ghi:

```
TRAIN_CONFIG  -->  configs/train.yaml
EVAL_CONFIG   -->  configs/eval.yaml
```

---

## 4. Accelerate Multi-GPU Training Rules

Training **bắt buộc** sử dụng HuggingFace `accelerate`.

**Cấm** tự quản lý GPU thủ công:

```python
# CẤM
model.cuda(0)
model.cuda(1)
```

**Phải dùng:**

```python
from accelerate import Accelerator
accelerator = Accelerator()
```

**Training command:**

```bash
accelerate launch \
  --num_processes 2 \
  src/train.py \
  --config configs/train.yaml
```

Hoặc dùng config file:

```bash
accelerate launch \
  --config_file accelerate_config.yaml \
  src/train.py \
  --config configs/train.yaml
```

Phải tạo file `accelerate_config.yaml`:

```yaml
distributed_type: MULTI_GPU
num_processes: 2
mixed_precision: fp16
```

---

## 5. Training Entry Point Rules

File: `src/train.py`

Bắt buộc có:

```python
def train(config):
    pass

if __name__ == "__main__":
    train(config)
```

**Training phải:**
1. Load config yaml
2. Initialize accelerator
3. Prepare model, optimizer, dataloader
4. Train
5. Save checkpoint
6. Write logs

---

## 6. Evaluation Entry Point Rules

File: `src/evaluate.py`

Bắt buộc có:

```python
def evaluate(config):
    pass

if __name__ == "__main__":
    evaluate(config)
```

**Evaluation phải:**
1. Load checkpoint
2. Inference
3. Calculate metrics
4. Save result

---

## 7. Checkpoint Rules

Training bắt buộc lưu checkpoint.

**Structure:**

```
checkpoints/
├── checkpoint-500/
├── checkpoint-1000/
└── latest/
```

Mỗi checkpoint phải chứa:
- Model weights
- Optimizer state
- Scheduler state
- Training state
- Config

```
checkpoint-XXXX/
├── model.safetensors
├── optimizer.pt
├── scheduler.pt
├── trainer_state.json
└── config.yaml
```

---

## 8. Resume Training Rules

Pipeline phải tự động kiểm tra checkpoint:

```python
if os.path.exists("checkpoints/latest"):
    resume_training = True
```

Nếu có checkpoint, training command phải hỗ trợ `--resume_from_checkpoint`:

```bash
accelerate launch \
  --num_processes 2 \
  src/train.py \
  --config configs/train.yaml \
  --resume_from_checkpoint checkpoints/latest
```

---

## 9. Logging Rules

Mọi experiment phải có log.

**Structure:**

```
logs/
├── train/
│   ├── train.log
│   └── metrics.json
└── evaluate/
    ├── eval.log
    └── results.json
```

**Training log phải ghi:**
- Timestamp
- Experiment name
- Epoch
- Step
- Loss
- Learning rate
- GPU memory
- Checkpoint path

**Evaluation log phải ghi:**
- Checkpoint sử dụng
- Dataset size
- Metrics
- Timestamp

---

## 10. Experiment Tracking Rules

Mỗi lần train phải tạo experiment id, ví dụ:

```
2026_07_14_llama_exp001
```

**Output structure:**

```
outputs/
└── experiment_id/
    ├── config/
    │   ├── train.yaml
    │   └── eval.yaml
    ├── checkpoints/
    ├── logs/
    ├── results/
    ├── git_commit.txt
    └── environment.txt
```

---

## 11. Reproducibility Rules

Mỗi experiment phải lưu:
- `train.yaml`
- `eval.yaml`
- Git commit hash
- Python environment
- Random seed

```bash
git rev-parse HEAD > git_commit.txt
pip freeze > environment.txt
```

---

## 12. GPU Optimization Rules

Cho Tesla T4, ưu tiên:
- FP16 mixed precision
- Gradient accumulation
- Gradient clipping
- Efficient dataloader

```yaml
gradient_accumulation_steps: 8
max_grad_norm: 1.0
mixed_precision: fp16
```

---

## 13. Kaggle Notebook Workflow Rules

Notebook workflow bắt buộc theo đúng thứ tự:

1. Clone repo
2. Install dependencies
3. Generate `train.yaml`
4. Generate `eval.yaml`
5. Check checkpoint
6. Run accelerate training
7. Monitor logs
8. Run evaluation
9. Backup outputs

---

## 14. Backup Rules

Trước khi kết thúc Kaggle session, phải backup:
- `outputs/`
- `logs/`
- `checkpoints/`

```bash
tar -czf experiment_backup.tar.gz \
  outputs logs checkpoints
```

---

## 15. Git Rules

**Không commit:**

```
checkpoints/
logs/
outputs/
*.pt
*.bin
*.safetensors
```

Phải có file `.gitignore` khai báo đầy đủ các mục trên.

**Rule tạo file `.gitignore`:**
- Nếu repo **chưa có** file `.gitignore` ở root, AI coding agent **bắt buộc phải tự tạo** file này trước khi thực hiện bất kỳ thay đổi nào khác liên quan đến training pipeline.
- Nếu file `.gitignore` đã tồn tại, agent phải kiểm tra và **bổ sung** các mục còn thiếu, không được ghi đè hoặc xóa các rule không liên quan đã có sẵn.
- Nội dung tối thiểu bắt buộc của `.gitignore`:

```gitignore
# Checkpoints & model weights
checkpoints/
*.pt
*.bin
*.safetensors
*.ckpt

# Logs
logs/
*.log

# Outputs & experiment artifacts
outputs/
experiment_backup.tar.gz

# Environment
__pycache__/
*.pyc
.ipynb_checkpoints/
venv/
.env

# Kaggle
kaggle.json
```

- Agent phải chạy kiểm tra `git status` sau khi tạo `.gitignore` để xác nhận `checkpoints/`, `logs/`, `outputs/` không còn được git track.

---

## 16. AI Coding Agent Rules

Khi AI chỉnh sửa repo, bắt buộc:
- Đọc cấu trúc repo trước khi thay đổi.
- Không tạo file thừa.
- Không hardcode config.
- Giữ backward compatibility.
- Không phá checkpoint cũ.
- Kiểm tra accelerate compatibility.
- Kiểm tra resume compatibility.
- Test trước khi hoàn thành.

---

## 17. Definition of Done

Pipeline chỉ được coi là hoàn thành khi đạt:

- [x] Can run on Kaggle T4 x2
- [x] Supports accelerate distributed training
- [x] Saves checkpoints
- [x] Produces training logs
- [x] Produces evaluation logs
- [x] Can resume training
- [x] Can reproduce experiment from config
- [x] Can run from GitHub clone
- [x] Automatically creates train.yaml
- [x] Automatically creates eval.yaml
- [x] Saves git commit information
- [x] Saves environment information
- [x] Supports experiment tracking