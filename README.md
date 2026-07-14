# Medical Relation Extraction on n2c2 2010 (English & Vietnamese)

Dự án này tập trung vào bài toán **Trích xuất Quan hệ Y khoa (Medical Relation Extraction)** trên bộ dữ liệu **n2c2 2010** cho cả hai ngôn ngữ **Tiếng Anh** và **Tiếng Việt**.

---

## 📌 Mục đích dự án

Mục đích chính của repository này là phân loại mối quan hệ giữa các thực thể y khoa (như *Treatment* - Phương pháp điều trị, *Problem* - Bệnh/Triệu chứng, *Test* - Xét nghiệm) được tìm thấy trong hồ sơ bệnh án y khoa. 

Các quan hệ cần trích xuất bao gồm:
*   **TrAP**: Treatment Applied for Problem (Điều trị cho bệnh)
*   **TrWP**: Treatment Worsened Problem (Điều trị làm tệ hơn bệnh)
*   **TrCP**: Treatment Caused Problem (Điều trị gây ra bệnh phụ)
*   **TrIP**: Treatment Improved Problem (Điều trị làm giảm bệnh)
*   **TrNAP**: Treatment Not Administered for Problem (Điều trị không áp dụng cho bệnh)
*   **TeRP**: Test Revealed Problem (Xét nghiệm phát hiện bệnh)
*   **TeCP**: Test Caused Problem (Xét nghiệm gây ra biến chứng/bệnh)
*   **PIP**: Problem Influence Problem (Bệnh này ảnh hưởng bệnh kia)
*   **false**: Không có quan hệ

---

## 📂 Cấu trúc thư mục

```text
vi_medical_re/
├── configs/                     # Chứa các file cấu hình huấn luyện và đánh giá (.yaml)
│   ├── train.yaml               # Cấu hình train Tiếng Anh (mặc định: BioBERT)
│   ├── eval.yaml                # Cấu hình đánh giá Tiếng Anh (mặc định: BioBERT)
│   ├── train_vi.yaml            # Cấu hình train Tiếng Việt (mặc định: XLM-RoBERTa)
│   └── eval_vi.yaml             # Cấu hình đánh giá Tiếng Việt (mặc định: XLM-RoBERTa)
├── src/                         # Mã nguồn chính của dự án (Modularized)
│   ├── train.py                 # File thực thi huấn luyện chính
│   ├── evaluate.py              # File thực thi đánh giá chính
│   ├── model.py                 # Khởi tạo kiến trúc mô hình (Hugging Face AutoModel)
│   ├── dataset.py               # Tiền xử lý dữ liệu và tạo PyTorch Dataset (Tự động nhận diện EN/VI)
│   ├── trainer.py               # Lớp Trainer quản lý vòng lặp huấn luyện bằng Accelerate
│   ├── logger.py                # Thiết lập ghi log tập trung
│   └── utils.py                 # Các hàm bổ trợ (seed, thống kê bộ nhớ GPU)
├── n2c2 2010/                   # Thư mục chứa dữ liệu JSONL (Đã được gitignore ngoại trừ các tệp dữ liệu mẫu)
├── notebooks/                   # File notebook chạy trên Kaggle
│   └── kaggle_train.ipynb       # Execution controller quản lý huấn luyện trên Kaggle (NVIDIA T4 x2)
├── accelerate_config.yaml       # Cấu hình phân phối Multi-GPU của Hugging Face Accelerate
├── requirements.txt             # Các thư viện phụ thuộc của dự án
└── README.md                    # File hướng dẫn này
```

---

## 🛠️ Cài đặt môi trường

Yêu cầu hệ thống: **Python >= 3.10**

Cài đặt tất cả các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn chạy từng bước

### Bước 1: Chuẩn bị cấu hình
Các tham số huấn luyện (như epochs, batch size, learning rate, paths) đều được quản lý tập trung thông qua các file YAML trong thư mục `configs/`.

*   Để huấn luyện **Tiếng Anh**: Sử dụng [configs/train.yaml](file:///c:/lab/vi_medical_re/configs/train.yaml) (sử dụng mô hình gốc `dmis-lab/biobert-base-cased-v1.2`).
*   Để huấn luyện **Tiếng Việt**: Sử dụng [configs/train_vi.yaml](file:///c:/lab/vi_medical_re/configs/train_vi.yaml) (sử dụng mô hình đa ngôn ngữ `xlm-roberta-base`).

---

### Bước 2: Huấn luyện mô hình (Training)

#### Chạy trên môi trường Single-GPU / CPU:
Sử dụng trực tiếp lệnh Python thông thường:
*   **Huấn luyện Tiếng Anh:**
    ```bash
    python src/train.py --config configs/train.yaml
    ```
*   **Huấn luyện Tiếng Việt:**
    ```bash
    python src/train.py --config configs/train_vi.yaml
    ```

#### Chạy trên môi trường Multi-GPU (Ví dụ: 2x Tesla T4 trên Kaggle):
Sử dụng thư viện `accelerate` để phân phối tải tự động trên các card đồ họa:
*   **Huấn luyện Tiếng Anh:**
    ```bash
    accelerate launch --num_processes 2 src/train.py --config configs/train.yaml
    ```
*   **Huấn luyện Tiếng Việt:**
    ```bash
    accelerate launch --num_processes 2 src/train.py --config configs/train_vi.yaml
    ```

> [!TIP]
> Hệ thống sẽ tự động tạo thư mục `checkpoints/latest` để lưu trạng thái huấn luyện. Nếu quá trình huấn luyện bị gián đoạn, bạn có thể tiếp tục huấn luyện bằng cách thêm cờ `--resume_from_checkpoint` vào cuối lệnh.

---

### Bước 3: Đánh giá mô hình (Evaluation)

Sau khi huấn luyện hoàn tất, các checkpoint sẽ được lưu lại. Bạn chạy lệnh đánh giá để tính toán các chỉ số Precision, Recall và F1-score trên tập kiểm thử (Test set):

*   **Đánh giá Tiếng Anh:**
    ```bash
    python src/evaluate.py --config configs/eval.yaml
    ```
*   **Đánh giá Tiếng Việt:**
    ```bash
    python src/evaluate.py --config configs/eval_vi.yaml
    ```

Kết quả báo cáo phân loại chi tiết (Classification Report) và các chỉ số sẽ được ghi nhận tại file log `logs/evaluate/eval.log` và xuất ra tệp JSON tại `outputs/<experiment_name>/results.json`.

---

### 📓 Chạy trên Kaggle Notebook

Dự án cung cấp file notebook điều khiển [kaggle_train.ipynb](file:///c:/lab/vi_medical_re/notebooks/kaggle_train.ipynb) để có thể import trực tiếp vào Kaggle và chạy trên môi trường **2x NVIDIA Tesla T4**.

Quy trình hoạt động trong notebook:
1.  Cài đặt môi trường từ [requirements.txt](file:///c:/lab/vi_medical_re/requirements.txt).
2.  Tự động ghi đè hoặc chỉnh sửa các tệp cấu hình YAML từ giao diện code notebook.
3.  Tự động kiểm tra checkpoint cũ để kích hoạt tính năng chạy nối tiếp (Resume training) nếu có.
4.  Kích hoạt huấn luyện song song bằng lệnh `accelerate launch`.
5.  Thực hiện đánh giá trên tập kiểm thử và đóng gói toàn bộ kết quả (`outputs/`, `logs/`, `checkpoints/`) thành tệp tin `experiment_backup.tar.gz` để tải xuống.
