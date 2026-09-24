@echo off
cd /d "%~dp0"
if not exist .venv (
  echo شغل install.bat أولا.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
pause
