@echo off
setlocal
cd /d %~dp0

set PYTHON_EXE=..\.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python venv not found at %PYTHON_EXE%
  exit /b 1
)

"%PYTHON_EXE%" export_wow_addon.py
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" test_addon_offline.py
if errorlevel 1 exit /b 1

echo.
echo [OK] Addon offline test complete.
