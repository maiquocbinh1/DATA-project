# 📄 HƯỚNG DẪN VIẾT BÁO CÁO (8-12 TRANG)

## Cấu trúc báo cáo đề xuất

---

### 1. TRANG BÌA (1 trang)
- Tên trường/khoa
- Tên môn học: **KHAI THÁC DỮ LIỆU**
- Đề tài: **XÂY DỰNG HỆ THỐNG GỢI Ý PHIM VỚI TMDB 5000**
- Họ tên sinh viên + MSSV
- Giảng viên hướng dẫn
- Ngày nộp

---

### 2. MỤC LỤC (1 trang)
- Danh sách các chương, mục với số trang

---

### 3. CHƯƠNG 1: GIỚI THIỆU (1-1.5 trang)

#### 1.1. Bối cảnh & Động lực
- Vai trò của hệ thống gợi ý trong thời đại số
- Ứng dụng trong các nền tảng như Netflix, Amazon
- Tại sao chọn đề tài gợi ý phim?

#### 1.2. Mục tiêu dự án
- Xây dựng hệ thống gợi ý phim dựa trên nội dung (content-based)
- Phân tích và trực quan hóa dữ liệu TMDB
- Đánh giá hiệu quả mô hình
- Xây dựng giao diện web thân thiện

#### 1.3. Phạm vi & Giới hạn
- Dataset: TMDB 5000 (4803 phim)
- Phương pháp: Content-Based Filtering
- Không bao gồm collaborative filtering (do thiếu dữ liệu user-item interaction)

---

### 4. CHƯƠNG 2: CƠ SỞ LÝ THUYẾT (1.5-2 trang)

#### 2.1. Hệ thống gợi ý (Recommendation System)
- Định nghĩa
- Các loại hệ thống gợi ý:
  - Content-Based Filtering
  - Collaborative Filtering
  - Hybrid Methods

#### 2.2. Content-Based Filtering
- Nguyên lý hoạt động
- Ưu điểm & Nhược điểm
- Ứng dụng

#### 2.3. TF-IDF (Term Frequency - Inverse Document Frequency)
- Công thức toán học
- Ý nghĩa các thành phần
- Ứng dụng trong vector hóa văn bản

#### 2.4. Cosine Similarity
- Công thức tính độ tương đồng
- Giá trị từ 0 đến 1
- Ứng dụng trong so sánh vector

#### 2.5. Các metrics đánh giá
- RMSE & MAE (cho rating prediction)
- Precision@K & Recall@K (cho ranking)

---

### 5. CHƯƠNG 3: PHƯƠNG PHÁP & DỮ LIỆU (2-2.5 trang)

#### 3.1. Dataset TMDB 5000
- Nguồn: TMDB (The Movie Database)
- 2 files: `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`
- Số lượng: 4803 phim
- Features chính:
  - Movies: title, overview, genres, keywords, vote_average, vote_count, popularity, budget, revenue, runtime, release_date
  - Credits: cast, crew

**Bảng mô tả features** (có thể thêm bảng)

| Feature | Mô tả | Kiểu dữ liệu |
|---------|-------|--------------|
| title | Tên phim | String |
| overview | Nội dung tóm tắt | String |
| genres | Thể loại (JSON) | String |
| ... | ... | ... |

#### 3.2. Quy trình xử lý dữ liệu

**Sơ đồ quy trình** (vẽ flowchart đơn giản):
```
Nạp dữ liệu → Merge 2 file → Xử lý missing → Loại duplicate → 
Xử lý outliers → Chuẩn hóa → Vector hóa TF-IDF → Tính Cosine Similarity
```

##### 3.2.1. Missing values
- Text columns: thay bằng chuỗi rỗng
- Numeric columns: thay bằng median

##### 3.2.2. Duplicate removal
- Loại bỏ phim trùng lặp theo title

##### 3.2.3. Outliers
- Clip vote_count ở percentiles 1%-99%

##### 3.2.4. Chuẩn hóa
- MinMaxScaler cho vote_average, popularity, vote_count

##### 3.2.5. Feature Engineering
- Parse JSON: genres, keywords, cast, crew
- Tạo combined_features = overview + genres + keywords + cast + director

##### 3.2.6. Vector hóa TF-IDF
- max_features = 5000
- ngram_range = (1, 2)
- stop_words = 'english'

#### 3.3. Xây dựng mô hình
- Tính cosine similarity matrix (4803 × 4803)
- Hàm `get_recommendations(title, top_n)`
- Workflow:
  1. Tìm index của phim
  2. Lấy similarity scores
  3. Sắp xếp giảm dần
  4. Trả về top N phim

---

### 6. CHƯƠNG 4: KẾT QUẢ & ĐÁNH GIÁ (2-3 trang)

#### 4.1. Phân tích dữ liệu (EDA)

##### 4.1.1. Phân bố Rating
- **Chèn hình ảnh histogram & boxplot**
- Nhận xét:
  - Mean rating: ~6.x
  - Phần lớn phim có rating 5-7
  - Một số phim có rating rất cao (>9)

##### 4.1.2. Tần suất thể loại
- **Chèn hình bar chart top 15 genres**
- Nhận xét:
  - Drama là thể loại phổ biến nhất (~2500 phim)
  - Tiếp theo là Comedy, Thriller, Action

##### 4.1.3. Top phim rating cao
- **Chèn bảng top 10 phim**
- Phân tích đặc điểm chung

##### 4.1.4. Heatmap tương quan
- **Chèn hình heatmap**
- Nhận xét:
  - vote_count và popularity có tương quan dương mạnh
  - budget và revenue tương quan mạnh
  - vote_average có tương quan yếu với các biến khác

#### 4.2. Kết quả gợi ý

##### Ví dụ: Gợi ý cho phim "Avatar"
**Chèn bảng kết quả:**

| # | Tên phim | Similarity | Rating |
|---|----------|------------|--------|
| 1 | Guardians of the Galaxy | 0.45 | 7.9 |
| 2 | Star Wars | 0.42 | 8.1 |
| ... | ... | ... | ... |

**Phân tích:**
- Các phim được gợi ý đều là sci-fi/adventure
- Similarity score từ 0.3-0.5
- Phù hợp với nội dung và thể loại của Avatar

#### 4.3. Đánh giá mô hình

**Bảng tổng hợp metrics:**

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| RMSE | 0.XX | Sai số dự đoán rating |
| MAE | 0.XX | Sai số tuyệt đối trung bình |
| Precision@10 | 0.XX | Tỷ lệ phim relevant trong top 10 |
| Recall@10 | 0.XX | Tỷ lệ phim relevant được tìm thấy |

**Phân tích:**
- RMSE và MAE thấp → mô hình dự đoán rating tốt
- Precision cao → gợi ý chính xác
- Recall thấp → do số lượng relevant items lớn

#### 4.4. Giao diện Web App

**Chèn screenshot giao diện Streamlit:**
- Trang chủ
- Kết quả gợi ý
- Thông tin chi tiết phim

**Mô tả tính năng:**
- Tìm kiếm phim theo tên
- Điều chỉnh số lượng gợi ý
- Hiển thị thông tin đầy đủ
- Download CSV

---

### 7. CHƯƠNG 5: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (1 trang)

#### 5.1. Kết luận
- Đã xây dựng thành công hệ thống gợi ý phim
- Content-based filtering hiệu quả với dữ liệu TMDB
- Giao diện web thân thiện, dễ sử dụng
- Đạt đầy đủ yêu cầu đề bài

#### 5.2. Ưu điểm
- Không cần dữ liệu user (cold start problem)
- Gợi ý dựa trên nội dung chất lượng
- Giải thích được kết quả (thể loại, diễn viên, đạo diễn)

#### 5.3. Hạn chế
- Không khai thác được sở thích người dùng
- Không phát hiện được phim "bất ngờ" (serendipity)
- Chỉ gợi ý phim tương tự, thiếu đa dạng

#### 5.4. Hướng phát triển (tính điểm cộng)
- **Hybrid System**: Kết hợp collaborative filtering
- **Embeddings nâng cao**: Sử dụng BERT, Word2Vec
- **Context-aware**: Gợi ý theo thời gian, thể loại trending
- **User profile**: Lưu lịch sử xem và sở thích
- **Deploy cloud**: Heroku, AWS, Azure

---

### 8. TÀI LIỆU THAM KHẢO (0.5 trang)

```
[1] Dataset: TMDB 5000 Movie Dataset, Kaggle
[2] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: 
    History and Context. ACM Transactions on Interactive Intelligent Systems
[3] Scikit-learn Documentation: TF-IDF Vectorizer
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
[4] Streamlit Documentation
    https://docs.streamlit.io
[5] Pazzani, M.J., Billsus, D. (2007). Content-Based Recommendation Systems. 
    The Adaptive Web. Lecture Notes in Computer Science, vol 4321.
```

---

## 📝 LƯU Ý KHI VIẾT BÁO CÁO

### ✅ Nên làm:
1. **Chèn hình ảnh, biểu đồ** từ notebook (screenshot hoặc export)
2. **Giải thích công thức** một cách đơn giản, dễ hiểu
3. **Phân tích kết quả** thay vì chỉ liệt kê số liệu
4. **Trích dẫn nguồn** đầy đủ
5. **Format đẹp**: Times New Roman 13, căn lề, số trang
6. **Kiểm tra chính tả** kỹ lưỡng

### ❌ Không nên:
1. Copy-paste code dài vào báo cáo (chỉ nên giải thích logic)
2. Chèn quá nhiều bảng số liệu
3. Viết quá dài dòng, lan man
4. Thiếu phân tích, chỉ mô tả

### 💡 Tips:
- Mỗi hình ảnh/bảng phải có **chú thích** (Figure 1, Table 1, ...)
- Sử dụng **bullet points** cho dễ đọc
- Highlight **keywords** quan trọng
- Thêm phần **Appendix** nếu cần (code, bảng đầy đủ)

---

## 📊 CHECKLIST HOÀN THÀNH

- [ ] Chạy notebook, lưu tất cả hình ảnh EDA
- [ ] Chụp screenshot giao diện Streamlit
- [ ] Viết phần lý thuyết TF-IDF, Cosine Similarity
- [ ] Tạo bảng metrics đánh giá
- [ ] Viết phần phân tích kết quả
- [ ] Kiểm tra format, chính tả
- [ ] Xuất PDF 8-12 trang
- [ ] Đóng gói file: `TenSV_MaSV_finalProject.zip`

---

**Chúc bạn viết báo cáo thành công! 🎓**

