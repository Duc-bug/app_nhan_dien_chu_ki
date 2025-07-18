@echo off
chcp 65001 >nul
echo ========================================
echo   🚀 DEPLOY TO RAILWAY - Full App
echo ========================================
echo.

echo 📦 Chuẩn bị files cho Railway deployment...

REM Backup requirements hiện tại
if exist "requirements.txt" (
    copy "requirements.txt" "requirements_backup.txt"
    echo ✅ Backup requirements.txt gốc
)

REM Sử dụng requirements tối ưu cho Railway
copy "requirements_railway.txt" "requirements.txt"
echo ✅ Sử dụng requirements_railway.txt

echo.
echo 📤 Push lên GitHub...
git add railway.toml requirements_railway.txt Dockerfile requirements.txt
git status

echo.
echo Commit deployment files...
git commit -m "Add Railway deployment config - Full app with all features"

echo.
echo Push lên GitHub...
git push origin main

if errorlevel 0 (
    echo.
    echo ========================================
    echo   ✅ PUSH THÀNH CÔNG!
    echo ========================================
    echo.
    echo 🌟 TIẾP THEO - Deploy trên Railway:
    echo.
    echo 1. Truy cập: https://railway.app/
    echo 2. Đăng ký/Đăng nhập bằng GitHub
    echo 3. Click "New Project"
    echo 4. Chọn "Deploy from GitHub repo"
    echo 5. Chọn repository: Duc-bug/app_nhan_dien_chu_ki
    echo 6. Railway sẽ tự động detect và deploy!
    echo.
    echo 🔧 Cấu hình tự động:
    echo   - Build: NIXPACKS (auto-detect)
    echo   - Start command: streamlit run app.py
    echo   - Environment: Production ready
    echo.
    echo 🌐 URL demo sẽ có dạng:
    echo    https://app-nhan-dien-chu-ki-production.up.railway.app/
    echo.
    echo 📊 Railway Features:
    echo   ✅ 500h/tháng miễn phí
    echo   ✅ Auto-restart khi crash
    echo   ✅ Persistent storage
    echo   ✅ Custom domains
    echo   ✅ Metrics và logs
    echo.
) else (
    echo.
    echo ❌ Push thất bại!
    echo Kiểm tra connection và thử lại
)

REM Restore requirements gốc
if exist "requirements_backup.txt" (
    copy "requirements_backup.txt" "requirements.txt"
    del "requirements_backup.txt"
    echo 🔄 Khôi phục requirements.txt gốc
)

echo.
pause
