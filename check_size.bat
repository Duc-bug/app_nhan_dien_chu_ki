@echo off
chcp 65001 >nul
echo ========================================
echo   KIEM TRA KICH THUOC - Size Check
echo ========================================
echo.

echo Kiem tra kich thuoc cac thu muc:
echo.

for /d %%d in (*) do (
    if exist "%%d" (
        echo Checking %%d...
        dir "%%d" /s /-c 2>nul | find "File(s)" | find /v "Dir(s)"
    )
)

echo.
echo Files lon nhat:
forfiles /m *.* /s /c "cmd /c if @fsize gtr 10485760 echo @path - @fsize bytes" 2>nul

echo.
echo Thu muc co the gay van de:
echo - venv/ (virtual environment)
echo - data/ (neu co file model lon)
echo - model/ (neu co file .h5 lon)
echo.

pause
