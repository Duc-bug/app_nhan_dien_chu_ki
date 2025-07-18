@echo off
chcp 65001 >nul
echo ========================================
echo   FIX LARGE FILES - Sua Loi File Lon
echo ========================================
echo.

echo Van de: Thu muc venv/ qua lon cho GitHub
echo Giai phap: Loai bo venv va chi push source code
echo.

echo Buoc 1: Xoa venv khoi git cache...
git rm -r --cached venv/ 2>nul

echo Buoc 2: Cap nhat .gitignore...
echo venv/ >> .gitignore
echo *.h5 >> .gitignore
echo *.pkl >> .gitignore

echo Buoc 3: Add lai files (khong co venv)...
git add .

echo Buoc 4: Kiem tra files se duoc push...
echo.
git status --porcelain

echo.
echo Buoc 5: Commit...
git commit -m "Remove venv and large files, add proper gitignore"

echo Buoc 6: Push len GitHub...
git push origin main

if errorlevel 0 (
    echo.
    echo THANH CONG!
    echo Da loai bo cac file lon
    echo Chi push source code len GitHub
    echo.
    echo Repository: https://github.com/Duc-bug/app_nhan_dien_chu_ki
    echo San sang deploy len Streamlit Cloud!
    echo.
) else (
    echo.
    echo Van con loi!
    echo Kiem tra output ben tren
    echo.
)

pause
