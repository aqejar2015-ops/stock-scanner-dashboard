import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
REFERENCE_SYMBOLS = ["GRML", "IMCC", "WHLR"]
DEFAULT_UNIVERSE = "AAPL,MSFT,NVDA,AMD,TSLA,AMZN,META,GOOGL,NFLX,PLTR,SOFI,HOOD,SMCI,MARA,RIOT,COIN,AMC,GME,NIO,XPEV,LCID,RIVN,SPY,QQQ,IWM"
REFRESH_SECONDS = 120
BAR_MINUTES = 2
LOOKBACK_DAYS = 7
PATTERN_LENGTH = 30

st.set_page_config(page_title="US Stock Pattern Scanner", page_icon="📈", layout="wide")
st.title("📈 US Stock Pattern Scanner")
st.caption("تحليل حركة GRML و IMCC و WHLR ومقارنة نمطها مع أسهم أمريكية أخرى")


def parse_symbols(text):
    return list(dict.fromkeys(s.strip().upper() for s in text.replace("\n", ",").split(",") if s.strip()))


def date_range():
    end = datetime.now()
    return (end - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def clean_bars(results):
    rows = []
    for item in results or []:
        if all(k in item for k in ("o", "h", "l", "c", "v", "t")):
            rows.append({
                "Timestamp": pd.to_datetime(item["t"], unit="ms", utc=True).tz_convert("America/New_York"),
                "Open": float(item["o"]), "High": float(item["h"]),
                "Low": float(item["l"]), "Close": float(item["c"]), "Volume": float(item["v"]),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Timestamp").sort_index()
    df = df[(df.index.time >= pd.Timestamp("09:30").time()) & (df.index.time <= pd.Timestamp("16:00").time())]
    if df.empty:
        return df
    df["Return"] = df["Close"].pct_change()
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]
    return df.replace([np.inf, -np.inf], np.nan).dropna()


@st.cache_data(ttl=100, show_spinner=False)
def fetch_bars(symbol):
    if not API_KEY:
        return pd.DataFrame()
    start, end = date_range()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{BAR_MINUTES}/minute/{start}/{end}"
    try:
        response = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}, timeout=30)
        if response.status_code != 200:
            return pd.DataFrame()
        return clean_bars(response.json().get("results", []))
    except requests.RequestException:
        return pd.DataFrame()


def download_data(symbols):
    data = {}
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        frame = fetch_bars(symbol)
        if not frame.empty:
            data[symbol] = frame
        progress.progress((i + 1) / len(symbols))
    progress.empty()
    return data


def pattern(df):
    if len(df) < PATTERN_LENGTH + 1:
        return None
    values = np.nan_to_num(df["Return"].tail(PATTERN_LENGTH).to_numpy(float))
    std = values.std()
    return np.zeros(PATTERN_LENGTH) if std == 0 else (values - values.mean()) / std


def stats(df):
    first, last = float(df.Close.iloc[0]), float(df.Close.iloc[-1])
    recent = df.tail(PATTERN_LENGTH)
    average_volume = df.Volume.tail(50).mean()
    return {
        "price": last,
        "week_return": (last / first - 1) * 100,
        "recent_return": (recent.Close.iloc[-1] / recent.Close.iloc[0] - 1) * 100 if len(recent) > 1 else 0,
        "volatility": float(df.Return.std() * np.sqrt(195) * 100),
        "volume_ratio": float(df.Volume.iloc[-1] / average_volume) if average_volume else 0,
        "bars": len(df),
    }


def similarity(candidate, references):
    values = []
    for reference in references:
        if candidate is not None and reference is not None:
            corr = np.corrcoef(candidate, reference)[0, 1]
            if not np.isnan(corr):
                values.append(corr)
    return float(np.mean(values)) if values else np.nan


def ranking(data, universe):
    references = [pattern(data[s]) for s in REFERENCE_SYMBOLS if s in data]
    rows = []
    for symbol in universe:
        if symbol in REFERENCE_SYMBOLS or symbol not in data:
            continue
        candidate = pattern(data[symbol])
        if candidate is None:
            continue
        item = stats(data[symbol])
        score = similarity(candidate, references)
        if np.isnan(score):
            continue
        score += min(max(item["volume_ratio"] - 1, -1), 1) * 0.05
        rows.append({"الرمز": symbol, "النتيجة": score, "التشابه %": score * 100, "السعر": item["price"], "تغير الأسبوع %": item["week_return"], "تغير آخر 30 شمعة %": item["recent_return"], "التذبذب %": item["volatility"], "الحجم مقارنة بالمتوسط": item["volume_ratio"], "عدد الشموع": item["bars"]})
    return pd.DataFrame(rows).sort_values("النتيجة", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


if not API_KEY:
    st.error("ضع مفتاح Polygon في ملف .env باسم POLYGON_API_KEY ثم أعد تشغيل البرنامج.")
    st.stop()

with st.sidebar:
    st.header("⚙️ الإعدادات")
    universe_text = st.text_area("الأسهم المراد فحصها", DEFAULT_UNIVERSE, height=220)
    top_n = st.slider("عدد النتائج", 3, 10, 3)
    st.info(f"شموع: {BAR_MINUTES} دقيقة | الفترة: {LOOKBACK_DAYS} أيام | التحديث: كل دقيقتين")

st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
universe = parse_symbols(universe_text)
symbols = list(dict.fromkeys(REFERENCE_SYMBOLS + universe))

with st.spinner("جاري جلب البيانات وتحليلها..."):
    data = download_data(symbols)

if not data:
    st.error("لم تصل بيانات. تحقق من المفتاح والاشتراك واتصال الإنترنت.")
    st.stop()

st.caption("آخر تحديث: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
cols = st.columns(3)
for col, symbol in zip(cols, REFERENCE_SYMBOLS):
    with col:
        st.subheader(symbol)
        if symbol not in data:
            st.warning("لا توجد بيانات")
            continue
        item = stats(data[symbol])
        st.metric("السعر", f"${item['price']:.4f}")
        st.metric("تغير الأسبوع", f"{item['week_return']:.2f}%")
        st.caption(f"الحجم الحالي: {item['volume_ratio']:.2f}x المتوسط")

st.divider()
st.header("🏆 أفضل الأسهم المشابهة")
result = ranking(data, universe)
if result.empty:
    st.warning("لا توجد نتائج كافية. أضف أسهماً أخرى أو تحقق من توفر البيانات.")
else:
    top = result.head(top_n)
    st.success("المرشحون الحاليون: " + ", ".join(top["الرمز"].tolist()))
    st.dataframe(top.style.format({"النتيجة": "{:.3f}", "التشابه %": "{:.2f}%", "السعر": "${:.4f}", "تغير الأسبوع %": "{:.2f}%", "تغير آخر 30 شمعة %": "{:.2f}%", "التذبذب %": "{:.2f}%", "الحجم مقارنة بالمتوسط": "{:.2f}x"}), use_container_width=True, hide_index=True)
    selected = st.selectbox("اختر سهماً للرسم", top["الرمز"].tolist())
    chart = data[selected][["Close"]].rename(columns={"Close": "السعر"})
    st.line_chart(chart, use_container_width=True)

st.warning("هذه أداة تحليل وليست توصية مالية. التشابه التاريخي لا يضمن الحركة المستقبلية.")
