@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo تم تشغيل النسخة الجديدة.
echo ضع مفتاح Polygon داخل خانة API Key في لوحة التحكم.
echo.
streamlit run app.py
