@echo off
chcp 65001 >nul
echo ========================================
echo   🔧 FIX RAILWAY BUILD ERRORS
echo ========================================
echo.

echo 🔄 Sửa lỗi build với requirements tối ưu...

REM Backup requirements hiện tại
if exist "requirements.txt" (
    copy "requirements.txt" "requirements_backup.txt"
)

REM Thử version minimal trước
echo 📦 Sử dụng requirements minimal...
copy "requirements_minimal_railway.txt" "requirements.txt"

echo.
echo 📤 Push fixes lên GitHub...
git add .
git commit -m "Fix Railway build errors - Use minimal stable requirements"
git push origin main

if errorlevel 0 (
    echo.
    echo ========================================
    echo   ✅ PUSH THÀNH CÔNG!
    echo ========================================
    echo.
    echo 🔧 Đã sửa lỗi build:
    echo   ✅ Downgrade TensorFlow → 2.13.0
    echo   ✅ Downgrade OpenCV → 4.8.1.78
    echo   ✅ Sử dụng pip upgrade
    echo   ✅ Thêm build tools
    echo.
    echo 🚀 Railway sẽ auto-rebuild với config mới
    echo.
    echo 📋 Nếu vẫn lỗi, thử:
    echo   1. Sử dụng NIXPACKS thay vì Docker
    echo   2. Deploy với app_demo.py thay vì app.py
    echo   3. Contact Railway support
    echo.
    echo 🌐 Monitor build tại: https://railway.app/
    echo.
) else (
    echo ❌ Push thất bại!
)

REM Restore requirements
if exist "requirements_backup.txt" (
    copy "requirements_backup.txt" "requirements.txt"
    del "requirements_backup.txt"
)

pause
