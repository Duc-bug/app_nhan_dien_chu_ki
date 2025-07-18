@echo off
chcp 65001 >nul
echo ========================================
echo   PUSH ESSENTIAL FILES - Push Tu Tung Phan
echo ========================================
echo.

echo Chien luoc: Chi push source code, loai bo file nang
echo.

REM Backup current state
echo Backup current state...
git stash push -m "Backup before essential push"

REM Xoa tat ca file khoi staging
git reset

REM Chi add nhung file can thiet
echo Adding essential files...

REM Core Python files
git add *.py
git add *.md
git add *.txt
git add *.bat

REM Add utils va model (chi code, khong data)
git add utils/*.py
git add model/*.py

REM Add gitignore
git add .gitignore

echo.
echo Files duoc add:
git status --porcelain

echo.
set /p confirm="Tiep tuc push essential files? (y/N): "
if /i "%confirm%" NEQ "y" (
    echo Huy bo
    git reset
    git stash pop
    pause
    exit /b 0
)

echo.
echo Committing essential files...
git commit -m "Add essential source code files only"

echo.
echo Pushing to GitHub...
git push origin main

if errorlevel 0 (
    echo.
    echo THANH CONG!
    echo Da push source code len GitHub
    echo.
    echo Repository: https://github.com/Duc-bug/app_nhan_dien_chu_ki
    echo.
    echo Tiep theo:
    echo 1. Kiem tra repo tren GitHub
    echo 2. Deploy len Streamlit Cloud
    echo 3. App se tu tai model khi can
    echo.
    
    REM Restore stashed files
    git stash pop
) else (
    echo.
    echo CO LOI!
    echo Restore lai state cu...
    git reset --hard HEAD~1
    git stash pop
)

pause
