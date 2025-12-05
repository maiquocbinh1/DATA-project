@echo off
chcp 65001 >nul
echo ================================================================
echo   🎬 TMDB MOVIE RECOMMENDER SYSTEM
echo ================================================================
echo.

echo [Bước 1/3] Kiểm tra thư viện...
pip show pandas >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Chưa cài đặt thư viện! Đang cài đặt...
    pip install -r requirements.txt
) else (
    echo ✓ Thư viện đã được cài đặt
)

echo.
echo [Bước 2/3] Training model từ dữ liệu TMDB...
python train_model.py
if %errorlevel% neq 0 (
    echo ❌ Lỗi khi train model!
    pause
    exit /b 1
)

echo.
echo [Bước 3/3] Khởi động Streamlit Web App...
echo.
echo ================================================================
echo   🌐 Trình duyệt sẽ tự động mở tại: http://localhost:8501
echo   📌 Nhấn Ctrl+C để dừng server
echo ================================================================
echo.
streamlit run app.py

pause

