@echo off
chcp 65001 >nul
echo 🚀 PUSH MINIMAL VERSION TO GITHUB - CHỈ PUSH FILE CƠ BẢN
echo ================================================

echo.
echo ⚠️ LOẠI BỎ CÁC FILE LỚN KHÔNG CẦN THIẾT...
echo.

REM Xóa tạm thời các file model lớn
if exist model\signature_model.h5 (
    echo 📁 Backup model file...
    move model\signature_model.h5 model\signature_model.h5.bak
)

REM Xóa tạm thời data demo để giảm size
if exist data\demo_genuine (
    echo 📁 Backup demo data...
    ren data\demo_genuine demo_genuine_bak
)

if exist data\demo_forged (
    ren data\demo_forged demo_forged_bak
)

echo.
echo ✅ COMMIT VÀ PUSH CHỈ SOURCE CODE...
git add .
git status
echo.
echo 📤 Push to GitHub...
git commit -m "Add lightweight version - Source code only"
git push origin main

echo.
echo 🔄 KHÔI PHỤC CÁC FILE ĐÃ BACKUP...

REM Khôi phục model
if exist model\signature_model.h5.bak (
    move model\signature_model.h5.bak model\signature_model.h5
    echo ✅ Model file restored
)

REM Khôi phục demo data
if exist data\demo_genuine_bak (
    ren data\demo_genuine_bak demo_genuine
    echo ✅ Demo genuine data restored
)

if exist data\demo_forged_bak (
    ren data\demo_forged_bak demo_forged
    echo ✅ Demo forged data restored
)

echo.
echo 🎉 HOÀN THÀNH! Repository đã được push với version nhẹ
echo 📋 Kiểm tra tại: https://github.com/Duc-bug/app_nhan_dien_chu_ki
echo.
pause
