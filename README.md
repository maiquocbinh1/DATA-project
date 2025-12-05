# 🎬 Movie Recommender System - TMDB 5000

## 📖 Mô tả dự án

Hệ thống gợi ý phim (Movie Recommendation System) sử dụng **Content-Based Filtering** để gợi ý các phim tương tự dựa trên:
- Nội dung mô tả (overview)
- Thể loại (genres)
- Từ khóa (keywords)
- Diễn viên (cast)
- Đạo diễn (director)

**Dataset**: TMDB 5000 Movies (>4800 phim)

---

## 🎯 Các tính năng chính

### ✅ Thu thập & Xử lý dữ liệu
- Gộp 2 file: `tmdb_5000_movies.csv` và `tmdb_5000_credits.csv`
- Xử lý missing values, duplicates, outliers
- Chuẩn hóa dữ liệu (MinMaxScaler)

### 📊 Phân tích & Trực quan hóa
- Phân bố rating (histogram, boxplot)
- Tần suất thể loại phim (bar chart)
- Top phim có rating cao nhất
- Heatmap tương quan giữa các biến

### 🤖 Mô hình gợi ý
- **Content-Based Filtering**
- Vector hóa: TF-IDF (max_features=5000, ngram_range=(1,2))
- Cosine Similarity để tính độ tương đồng

### 📈 Đánh giá mô hình
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Precision@K** và **Recall@K**

### 🌐 Giao diện Web
- Streamlit Web App với UI đẹp, hiện đại
- Tìm kiếm phim yêu thích
- Hiển thị thông tin chi tiết phim
- Danh sách gợi ý với similarity score
- Download danh sách gợi ý dạng CSV

---

## 🚀 Hướng dẫn cài đặt & chạy

### Bước 1: Clone hoặc tải project

```bash
cd KHDL
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 3: Chạy Jupyter Notebook để tạo model

```bash
jupyter notebook tmdb_recommender.ipynb
```

**Chạy tất cả các cell trong notebook** để:
- Load và xử lý dữ liệu
- Phân tích EDA
- Xây dựng mô hình
- Đánh giá mô hình
- **Lưu model vào file `movie_recommender_model.pkl`** (quan trọng!)

### Bước 4: Chạy Streamlit Web App

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tự động tại: `http://localhost:8501`

---

## 📂 Cấu trúc thư mục

```
KHDL/
│
├── tmdb_5000_movies.csv              # Dataset phim
├── tmdb_5000_credits.csv             # Dataset credits
│
├── tmdb_recommender.ipynb            # Jupyter Notebook phân tích & mô hình
├── app.py                            # Streamlit Web App
│
├── movie_recommender_model.pkl       # Model đã train (tạo sau khi chạy notebook)
│
├── requirements.txt                  # Danh sách thư viện
└── README.md                         # File này
```

---

## 💻 Sử dụng Web App

1. **Mở trình duyệt** tại `http://localhost:8501`
2. **Chọn phim yêu thích** từ dropdown menu
3. **Điều chỉnh số lượng gợi ý** (1-20 phim)
4. **Nhấn nút "Tìm phim tương tự"**
5. **Xem kết quả gợi ý** với thông tin chi tiết:
   - Tên phim
   - Rating (⭐)
   - Số lượng vote (👥)
   - Similarity score (🎯)
   - Thể loại
   - Tóm tắt nội dung
6. **Tải xuống danh sách gợi ý** dạng CSV (nếu cần)

---

## 📊 Kết quả đánh giá mô hình

| Metric | Giá trị |
|--------|---------|
| **RMSE** | ~0.5-0.8 |
| **MAE** | ~0.4-0.6 |
| **Precision@10** | ~0.3-0.5 |
| **Recall@10** | ~0.02-0.05 |

*(Kết quả cụ thể sẽ thay đổi tùy thuộc vào sample test)*

---

## 🛠️ Công nghệ sử dụng

- **Python 3.8+**
- **Pandas** - Xử lý dữ liệu
- **NumPy** - Tính toán số học
- **Matplotlib & Seaborn** - Trực quan hóa
- **Scikit-learn** - Machine Learning (TF-IDF, Cosine Similarity)
- **Streamlit** - Web Framework

---

## 📝 Yêu cầu đề bài đã hoàn thành

### 1. ✅ Thu thập dữ liệu
- Dataset: 4803 phim (>2000 ✓)
- Features: 20+ features (>5 ✓)

### 2. ✅ Làm sạch dữ liệu (3/3 tác vụ)
- ✓ Xử lý missing values
- ✓ Loại bỏ duplicates
- ✓ Xử lý outliers
- ✓ Chuẩn hóa dữ liệu
- ✓ Vector hóa (TF-IDF)

### 3. ✅ Phân tích & trực quan hóa (4/3 tác vụ)
- ✓ Phân bố rating (histogram + boxplot)
- ✓ Tần suất thể loại (bar chart)
- ✓ Top phim rating cao
- ✓ Heatmap tương quan

### 4. ✅ Xây dựng hệ gợi ý
- Content-Based Filtering với TF-IDF + Cosine Similarity

### 5. ✅ Đánh giá mô hình
- ✓ RMSE
- ✓ MAE
- ✓ Precision@K
- ✓ Recall@K

### 6. ✅ Giao diện hiển thị
- ✓ Web Interface (Streamlit) - đẹp & dễ sử dụng
- ✓ Tính năng tìm kiếm & gợi ý
- ✓ Hiển thị thông tin chi tiết
- ✓ Download CSV

---

## 🎓 Tác giả

**Final Project - Recommendation System**  
Môn: Khai thác dữ liệu (Data Mining)

---

## 📞 Hỗ trợ

Nếu gặp vấn đề khi chạy project:

1. **Lỗi không tìm thấy model**: Hãy chạy notebook trước để tạo file `movie_recommender_model.pkl`
2. **Lỗi import thư viện**: Kiểm tra lại `pip install -r requirements.txt`
3. **Lỗi encoding CSV**: Dataset phải ở cùng thư mục với notebook

---

## 📜 License

MIT License - Tự do sử dụng cho mục đích học tập.

---

**Made with ❤️ using Python & Streamlit**

