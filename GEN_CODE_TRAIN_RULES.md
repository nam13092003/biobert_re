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
    "seed": 42,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": false
}
```

Sau đó notebook phải tự động ghi:

```
TRAIN_CONFIG  -->  configs/train.yaml
EVAL_CONFIG   -->  configs/eval.yaml
```

**Lưu ý:** `metric_for_best_model` và `greater_is_better` bắt buộc phải được khai báo trong `train.yaml` vì đây là điều kiện dùng để xác định "best checkpoint" ở mục 7a.

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
5. Đánh giá trên tập validation ở mỗi `eval_steps`, so sánh với `metric_for_best_model` để cập nhật best checkpoint (xem mục 7a)
6. Save checkpoint (định kỳ + best)
7. Write logs

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
1. Load checkpoint — **mặc định phải load `checkpoints/best`** (xem mục 7a), trừ khi người dùng chỉ định rõ checkpoint khác qua `--checkpoint_path`
2. Inference
3. Calculate metrics
4. Lưu kết quả prediction **cho từng bản ghi** (xem mục 6a)
5. Save result (metrics tổng hợp)

### 6a. Per-record Prediction Output Rules

Ngoài metrics tổng hợp, evaluation **bắt buộc** phải lưu lại output dự đoán chi tiết cho **từng bản ghi (mỗi sample)** trong tập eval. Đây là yêu cầu bắt buộc, không tuỳ chọn.

**File output:** `logs/evaluate/predictions.jsonl` (định dạng JSON Lines, mỗi dòng là một bản ghi).

Mỗi dòng phải chứa tối thiểu các trường sau:

```json
{
  "id": "sample_0001",
  "checkpoint": "checkpoints/best",
  "input": "...",
  "label": "...",
  "prediction": "...",
  "score": 0.87,
  "correct": true
}
```

- `id`: định danh duy nhất của bản ghi (lấy từ dataset nếu có, nếu không phải tự sinh theo index).
- `checkpoint`: đường dẫn checkpoint dùng để dự đoán bản ghi này, để đảm bảo truy vết được.
- `input` / `label`: dữ liệu gốc và nhãn thật (ground truth), nếu có.
- `prediction`: output mô hình dự đoán.
- `score`: điểm số/metric riêng của bản ghi đó nếu áp dụng được (ví dụ confidence, loss, similarity...), có thể để `null` nếu không áp dụng.
- `correct`: đánh giá đúng/sai ở mức bản ghi nếu bài toán cho phép (classification, exact-match...), có thể để `null` với bài toán generation tự do.

**Quy tắc bắt buộc:**
- Không được ghi đè `predictions.jsonl` của lần evaluate trước một cách âm thầm; mỗi lần evaluate phải ghi vào thư mục riêng theo `experiment_id`/`checkpoint` (xem cấu trúc dưới).
- Việc ghi predictions phải streaming theo batch (ghi từng batch ngay khi có kết quả), không được giữ toàn bộ predictions trong RAM rồi ghi một lần, để tránh OOM trên Kaggle T4.

**Cấu trúc log evaluate cập nhật:**

```
logs/
└── evaluate/
    ├── eval.log
    ├── results.json          # metrics tổng hợp
    └── predictions.jsonl     # prediction chi tiết từng bản ghi
```

Trong `outputs/experiment_id/results/`, mỗi lần evaluate cũng phải lưu bản sao gắn với checkpoint đã dùng:

```
outputs/experiment_id/results/
└── <checkpoint_name>/
    ├── results.json
    └── predictions.jsonl
```

---

## 7. Checkpoint Rules

Training bắt buộc lưu checkpoint.

**Structure:**

```
checkpoints/
├── checkpoint-500/
├── checkpoint-1000/
├── best/
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

### 7a. Best Checkpoint Rules

Ngoài các checkpoint định kỳ theo `save_steps`, pipeline **bắt buộc** phải theo dõi và lưu lại checkpoint tốt nhất riêng biệt tại `checkpoints/best/`.

**Yêu cầu bắt buộc:**
- Config `train.yaml` phải khai báo `metric_for_best_model` (ví dụ `eval_loss`, `f1`, `accuracy`...) và `greater_is_better` (`true`/`false`) để xác định thế nào là "tốt hơn".
- Sau mỗi lần chạy validation (`eval_steps`), training loop phải so sánh giá trị metric hiện tại với giá trị tốt nhất đã ghi nhận trước đó.
- Nếu metric hiện tại tốt hơn, ghi đè `checkpoints/best/` bằng checkpoint hiện tại (đầy đủ model weights + config; optimizer/scheduler state có thể lưu tuỳ nhu cầu resume, nhưng model weights + `config.yaml` là bắt buộc).
- Phải ghi lại một file `checkpoints/best/best_metric.json` chứa:

```json
{
  "metric_name": "eval_loss",
  "metric_value": 0.4123,
  "step": 1500,
  "epoch": 3,
  "checkpoint_source": "checkpoint-1500",
  "timestamp": "2026-07-15T10:00:00"
}
```

- Việc cập nhật `checkpoints/best/` phải được ghi log rõ ràng vào `logs/train/train.log` (best metric cũ, best metric mới, step cập nhật).
- `checkpoints/best/` không được xoá/ghi đè bởi cơ chế rotate/cleanup của các checkpoint định kỳ (`checkpoint-XXXX/`); nó là thư mục độc lập, luôn giữ bản tốt nhất tính đến thời điểm hiện tại.
- **Evaluation (`src/evaluate.py`) mặc định phải sử dụng `checkpoints/best/`** để chạy đánh giá cuối cùng, trừ khi người dùng override bằng `--checkpoint_path` (xem mục 6).

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

Khi resume, trạng thái best checkpoint (`checkpoints/best/best_metric.json`) cũng phải được load lại để tránh việc một checkpoint tệ hơn ghi đè nhầm lên best hiện có.

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
    ├── results.json
    └── predictions.jsonl
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
- Trạng thái best checkpoint (có cập nhật hay không, metric value)

**Evaluation log phải ghi:**
- Checkpoint sử dụng (mặc định là `checkpoints/best`)
- Dataset size
- Metrics
- Đường dẫn file `predictions.jsonl`
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
    │   └── best/
    ├── logs/
    ├── results/
    │   └── <checkpoint_name>/
    │       ├── results.json
    │       └── predictions.jsonl
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
6. Run accelerate training (bao gồm cập nhật `checkpoints/best/` theo mục 7a)
7. Monitor logs
8. Run evaluation **trên `checkpoints/best/`**, lưu `predictions.jsonl` cho từng bản ghi
9. Backup outputs

---

## 14. Backup Rules

Trước khi kết thúc Kaggle session, phải backup:
- `outputs/`
- `logs/`
- `checkpoints/` (bao gồm `checkpoints/best/`)

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

# Predictions
predictions.jsonl

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
- Không phá `checkpoints/best/` đã có (chỉ được ghi đè khi metric mới thực sự tốt hơn theo mục 7a).
- Kiểm tra accelerate compatibility.
- Kiểm tra resume compatibility.
- Test trước khi hoàn thành.

---

## 17. Definition of Done

Pipeline chỉ được coi là hoàn thành khi đạt:

- [x] Can run on Kaggle T4 x2
- [x] Supports accelerate distributed training
- [x] Saves checkpoints
- [x] Saves best checkpoint (`checkpoints/best/`) dựa trên `metric_for_best_model`
- [x] Produces training logs
- [x] Produces evaluation logs
- [x] Evaluation runs on best checkpoint by default
- [x] Saves per-record prediction output (`predictions.jsonl`) during evaluate
- [x] Can resume training
- [x] Can reproduce experiment from config
- [x] Can run from GitHub clone
- [x] Automatically creates train.yaml
- [x] Automatically creates eval.yaml
- [x] Saves git commit information
- [x] Saves environment information
- [x] Supports experiment tracking