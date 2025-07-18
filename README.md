# 🖋️ Ứng Dụng Nhận Diện Chữ Ký AI

Ứng dụng AI nhận diện và xác minh chữ ký sử dụng Siamese Network và OpenCV.

## 🎯 Mục tiêu

- **Phân biệt chữ ký thật và giả** với độ chính xác cao
- **Giao diện thân thiện** với Streamlit
- **Quản lý dữ liệu hiệu quả** với SQLite
- **Tính năng vẽ chữ ký trực tiếp** trên web

## 🚀 Tính năng chính

### 👤 Quản lý người dùng
- Đăng ký người dùng mới
- Chọn người dùng để làm việc
- Xem danh sách người dùng

### 📝 Đăng ký chữ ký mẫu
- Upload ảnh chữ ký mẫu
- Xử lý ảnh tự động (grayscale, crop, resize)
- Trích xuất đặc trưng
- Lưu trữ vào database

### 🔍 Xác minh chữ ký
- Upload ảnh chữ ký cần kiểm tra
- So sánh với các mẫu đã lưu
- Hiển thị kết quả với % tin cậy
- Lưu lịch sử xác minh

### 🎨 Vẽ chữ ký trực tiếp
- Canvas tương tác để vẽ chữ ký
- Lưu làm mẫu hoặc xác minh ngay
- Hỗ trợ chuột và bút cảm ứng

### 🤖 Huấn luyện mô hình AI
- Siamese Network với TensorFlow/Keras
- Tạo dataset demo tự động
- Theo dõi quá trình training
- Lưu và load mô hình

### 📊 Thống kê và báo cáo
- Dashboard tổng quan
- Lịch sử xác minh chi tiết
- Biểu đồ thống kê
- Xuất báo cáo

## 🏗️ Cấu trúc dự án

```
signature_ai_app/
├── model/                  # Mô hình AI
│   ├── siamese_network.py  # Kiến trúc Siamese Network
│   ├── trainer.py          # Huấn luyện mô hình
│   └── signature_model.h5  # Mô hình đã train (sẽ tạo sau)
├── data/                   # Dữ liệu
│   ├── signatures/         # Ảnh chữ ký mẫu
│   ├── test/              # Ảnh test
│   └── database.db        # SQLite database
├── utils/                  # Tiện ích
│   ├── image_processor.py  # Xử lý ảnh
│   └── database.py        # Quản lý database
├── ui/                    # Giao diện (dự phòng)
├── app.py                 # Ứng dụng Streamlit chính
├── requirements.txt       # Dependencies
└── README.md             # Tài liệu này
```

## 🛠️ Cài đặt và chạy

### 1. Cài đặt Python dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt packages
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

## 📖 Hướng dẫn sử dụng

### Bước 1: Tạo người dùng
1. Vào menu "👤 Quản Lý Người Dùng"
2. Thêm người dùng mới với tên và email
3. Chọn người dùng để làm việc

### Bước 2: Đăng ký chữ ký mẫu
1. Vào menu "📝 Đăng Ký Chữ Ký"
2. Upload ảnh chữ ký rõ nét (PNG/JPG)
3. Xem ảnh đã xử lý và lưu làm mẫu
4. Có thể đăng ký nhiều mẫu cho 1 người

### Bước 3: Xác minh chữ ký
1. Vào menu "🔍 Xác Minh Chữ Ký"
2. Upload ảnh chữ ký cần kiểm tra
3. Xem kết quả: Hợp lệ/Không hợp lệ với % tin cậy
4. Kết quả được lưu vào lịch sử

### Bước 4: Vẽ chữ ký trực tiếp
1. Vào menu "🎨 Vẽ Chữ Ký"
2. Vẽ chữ ký bằng chuột trên canvas
3. Lưu làm mẫu hoặc xác minh ngay

### Bước 5: Xem thống kê
1. Vào menu "📊 Thống Kê & Lịch Sử"
2. Xem dashboard tổng quan
3. Xem lịch sử xác minh chi tiết
4. Phân tích biểu đồ thống kê

## 🤖 Về mô hình AI

### Kiến trúc Siamese Network
- **Input**: Cặp ảnh chữ ký (128x128 grayscale)
- **Base Network**: CNN với 4 lớp Conv2D + BatchNorm + MaxPool
- **Feature Extraction**: Dense layers (512 → 256 → 128)
- **Similarity**: Euclidean distance + Sigmoid
- **Output**: Xác suất hai chữ ký giống nhau (0-1)

### Quy trình xử lý ảnh
1. **Grayscale**: Chuyển sang ảnh xám
2. **Threshold**: Tách nền trắng/chữ đen
3. **Contour Detection**: Tìm viền chữ ký
4. **Crop & Padding**: Cắt vùng chữ ký + thêm viền
5. **Resize**: Chuẩn hóa về 128x128
6. **Normalize**: Giá trị pixel về [0,1]

### Đặc trưng được trích xuất
- Raw pixels (128x128 = 16,384 features)
- Histogram của gradient magnitude (32 bins)
- Histogram của gradient direction (32 bins)
- Thống kê cơ bản (mean, std, min, max, pixel ratio)

## 📊 Đánh giá mô hình

### Metrics sử dụng
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Tỷ lệ chữ ký thật được dự đoán đúng
- **Recall**: Tỷ lệ chữ ký thật được nhận diện
- **F1-Score**: Trung bình điều hòa của Precision và Recall

### Ngưỡng quyết định
- **Mặc định**: 70% similarity
- **Có thể điều chỉnh** trong Settings
- **Khuyến nghị**: 60-80% tùy yêu cầu

## 🗄️ Database Schema

### Bảng `users`
- `id`: Primary key
- `name`: Tên người dùng (unique)
- `email`: Email
- `created_at`: Thời gian tạo

### Bảng `signatures`
- `id`: Primary key
- `user_id`: Foreign key đến users
- `image_path`: Đường dẫn file ảnh
- `features`: JSON string của features vector
- `is_template`: True nếu là mẫu, False nếu là test
- `created_at`: Thời gian tạo

### Bảng `verifications`
- `id`: Primary key
- `user_id`: Foreign key đến users
- `template_signature_id`: ID chữ ký mẫu
- `test_signature_id`: ID chữ ký test
- `similarity_score`: Điểm tương đồng (0-1)
- `is_genuine`: True nếu hợp lệ
- `verification_time`: Thời gian xác minh

## 🔧 Customization

### Thay đổi tham số mô hình
Chỉnh sửa trong `model/siamese_network.py`:
```python
# Thay đổi input size
input_shape = (256, 256, 1)  # Ảnh 256x256

# Thay đổi kiến trúc
x = layers.Conv2D(64, (3, 3), activation='relu')(x)  # Thêm filters
```

### Thay đổi threshold
Trong `app.py`, tìm dòng:
```python
threshold = 0.7  # Ngưỡng 70%
```

### Thêm features mới
Trong `utils/image_processor.py`, method `extract_features()`:
```python
# Thêm đặc trưng mới
new_features = [...]
features = np.concatenate([features, new_features])
```

## 🚀 Triển khai Production

### 1. Streamlit Cloud
```bash
# Push code lên GitHub
git init
git add .
git commit -m "Initial commit"
git push origin main

# Deploy trên Streamlit Cloud: https://share.streamlit.io
```

### 2. Heroku
```bash
# Tạo Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### 3. Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## ⚠️ Lưu ý quan trọng

### Bảo mật
- **Không upload** chữ ký thật lên server public
- **Mã hóa** database trong production
- **Sử dụng HTTPS** cho web app
- **Backup** dữ liệu định kỳ

### Hiệu suất
- **GPU** khuyến nghị cho training
- **CPU** đủ cho inference
- **RAM**: Tối thiểu 4GB
- **Storage**: Tùy số lượng ảnh

### Độ chính xác
- **Dataset lớn** → Độ chính xác cao hơn
- **Nhiều mẫu/người** → Kết quả tốt hơn
- **Chất lượng ảnh** quan trọng
- **Fine-tune** threshold theo use case

## 🐛 Troubleshooting

### Lỗi thường gặp

**1. Import Error**
```bash
ModuleNotFoundError: No module named 'tensorflow'
```
**Giải pháp**: Cài đặt lại packages
```bash
pip install -r requirements.txt
```

**2. Database Error**
```bash
sqlite3.OperationalError: database is locked
```
**Giải pháp**: Đóng tất cả kết nối database

**3. Memory Error**
```bash
ResourceExhaustedError: OOM when allocating tensor
```
**Giải pháp**: Giảm batch_size hoặc input_size

**4. Canvas không hoạt động**
**Giải pháp**: Update streamlit và streamlit-drawable-canvas

## 📚 Tài liệu tham khảo

### Papers
- [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Signature Verification using Deep Learning](https://arxiv.org/abs/1705.05787)

### Datasets
- [CEDAR Signature Database](http://www.cedar.buffalo.edu/NIJ/data/)
- [MCYT Signature Database](http://atvs.ii.uam.es/atvs/signatures.html)

### Libraries
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Tác giả

**AI Assistant** - Phát triển bởi AI

Project Link: [https://github.com/your-username/signature_ai_app](https://github.com/your-username/signature_ai_app)

---

⭐ **Hãy cho dự án này một star nếu nó hữu ích cho bạn!** ⭐
