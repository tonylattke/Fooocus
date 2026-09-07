@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "fooocus_env\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

echo Starting Fooocus...
echo Browser should open at http://127.0.0.1:7865
echo Close this window to stop Fooocus.
echo.

"fooocus_env\Scripts\python.exe" entry_with_update.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
    echo Fooocus exited with code %EXIT_CODE%.
)
pause
endlocal & exit /b %EXIT_CODE%
