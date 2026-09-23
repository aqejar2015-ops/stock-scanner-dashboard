@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo افتح ملف .env وضع مفتاح Polygon ثم اضغط Enter.
pause
streamlit run app.py
