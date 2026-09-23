# US Stock Pattern Scanner

لوحة Streamlit تفحص الأسهم الأمريكية وتقارن آخر نمط حركة لها مع GRML وIMCC وWHLR باستخدام بيانات Polygon/Massive.

## التشغيل على Windows

1. نزّل المشروع من **Code > Download ZIP** وفك الضغط.
2. انسخ `.env.example` وسمّ النسخة `.env`.
3. افتح `.env` وضع مفتاحك:

```dotenv
POLYGON_API_KEY=ضع_مفتاحك_هنا
```

4. شغّل `start_windows.bat`.
5. افتح الرابط الذي يظهر، عادةً `http://localhost:8501`.

البرنامج يثبت المتطلبات تلقائياً ويتحدث كل دقيقتين.

> لا ترفع ملف `.env` إلى GitHub ولا تشارك مفتاح API. توفر بيانات الدقيقتين يعتمد على اشتراك Polygon/Massive. الأداة تحليلية وليست توصية مالية.
