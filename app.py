import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

ENV_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ENV_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()
FEED = os.getenv("ALPACA_FEED", "iex").strip() or "iex"
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "120"))
MIN_PRICE = float(os.getenv("MIN_PRICE", "1"))
MIN_DOLLAR_VOLUME = float(os.getenv("MIN_DOLLAR_VOLUME", "100000"))
MAX_SPREAD_PERCENT = float(os.getenv("MAX_SPREAD_PERCENT", "1.5"))
DATA_URL = "https://data.alpaca.markets"

st.set_page_config(page_title="NASDAQ Liquidity Scanner", page_icon="📈", layout="wide")
st.markdown("<style>html,body,[class*=css]{direction:rtl;text-align:right}.card{background:#101827;border:1px solid #263449;border-radius:12px;padding:12px}</style>", unsafe_allow_html=True)


def symbols_from_text(text):
    return list(dict.fromkeys(x.strip().upper() for x in text.replace(",", "\n").replace("،", "\n").splitlines() if x.strip()))[:100]


def load_symbols():
    path = BASE_DIR / "symbols.txt"
    return symbols_from_text(path.read_text(encoding="utf-8")) if path.exists() else []


def api_get(path, params, api_key, secret_key):
    response = requests.get(
        DATA_URL + path,
        headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key},
        params=params,
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Alpaca API {response.status_code}: {response.text[:300]}")
    return response.json()


def indicators(df):
    df = df.copy()
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["close", "volume"])
    typical = (df.high + df.low + df.close) / 3
    df["vwap_calc"] = (typical * df.volume).cumsum() / df.volume.replace(0, np.nan).cumsum()
    df["ema9"] = df.close.ewm(span=9, adjust=False).mean()
    df["ema20"] = df.close.ewm(span=20, adjust=False).mean()
    delta = df.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    df["avg_volume"] = df.volume.rolling(20, min_periods=3).mean()
    df["rvol"] = (df.volume / df.avg_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["dollar_volume"] = df.close * df.volume
    return df.dropna(subset=["close"])


def fetch_bars(symbols, api_key, secret_key):
    now = datetime.now(timezone.utc)
    payload = api_get(
        "/v2/stocks/bars",
        {"symbols": ",".join(symbols), "timeframe": "2Min", "start": (now - timedelta(hours=18)).isoformat(), "end": now.isoformat(), "feed": FEED, "adjustment": "raw", "limit": 10000, "sort": "asc"},
        api_key,
        secret_key,
    )
    output = {}
    for symbol, rows in payload.get("bars", {}).items():
        if len(rows) < 2:
            continue
        frame = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if all(column in frame for column in required):
            output[symbol] = indicators(frame)
    return output


def fetch_quotes(symbols, api_key, secret_key):
    payload = api_get("/v2/stocks/quotes/latest", {"symbols": ",".join(symbols), "feed": FEED}, api_key, secret_key)
    output = {}
    for symbol, quote in payload.get("quotes", {}).items():
        bid, ask = float(quote.get("bp") or 0), float(quote.get("ap") or 0)
        output[symbol] = {"spread": ((ask - bid) / ((ask + bid) / 2) * 100) if bid and ask else np.nan}
    return output


def score(symbol, frame, quote):
    if len(frame) < 2:
        return None
    previous, current = frame.iloc[-2], frame.iloc[-1]
    price = float(current.close)
    price_change = (price / float(previous.close) - 1) * 100 if previous.close else 0
    volume_growth = (float(current.volume) / float(previous.volume) - 1) * 100 if previous.volume else 0
    rvol, vwap, rsi = float(current.rvol), float(current.vwap_calc), float(current.rsi)
    spread = float(quote.get("spread", np.nan))
    total = 0.0
    reasons = []
    if price_change > 0: total += min(25, price_change * 8); reasons.append("السعر يرتفع")
    else: total += max(-15, price_change * 4)
    if volume_growth > 0: total += min(25, volume_growth / 4); reasons.append("الحجم يتزايد")
    else: total += max(-10, volume_growth / 8)
    if rvol >= 1.5: total += 15; reasons.append("RVOL مرتفع")
    elif rvol >= 1: total += 7
    if price > vwap: total += 10; reasons.append("فوق VWAP")
    if price > current.ema9 > current.ema20: total += 10; reasons.append("اتجاه EMA إيجابي")
    if 50 <= rsi <= 75: total += 8; reasons.append("RSI صحي")
    elif rsi > 85: total -= 8; reasons.append("تشبع محتمل")
    if np.isnan(spread) or spread <= MAX_SPREAD_PERCENT: total += 7
    else: total -= 15; reasons.append("سبريد مرتفع")
    if price_change <= 0 and volume_growth > 20: total -= 20; reasons.append("ضغط بيع محتمل")
    status = "إيجابي" if total >= 55 else "مراقبة" if total >= 35 else "ضعيف"
    return {"الرمز": symbol, "السعر": price, "تغير_دقيقتين": price_change, "قيمة_التداول": float(current.dollar_volume), "نمو_الحجم": volume_growth, "RVOL": rvol, "VWAP": vwap, "RSI": rsi, "السبريد": spread, "الدرجة": max(0, min(100, total)), "الحالة": status, "السبب": "، ".join(reasons)}


def scan(symbols, api_key, secret_key):
    bars = fetch_bars(symbols, api_key, secret_key)
    quotes = fetch_quotes(symbols, api_key, secret_key)
    rows = [score(symbol, frame, quotes.get(symbol, {})) for symbol, frame in bars.items()]
    rows = [row for row in rows if row and row["السعر"] >= MIN_PRICE and row["قيمة_التداول"] >= MIN_DOLLAR_VOLUME]
    return pd.DataFrame(rows).sort_values(["الدرجة", "قيمة_التداول"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


st.title("📈 ماسح السيولة والإيجابية - NASDAQ")
st.caption("مقارنة السعر والسيولة مع آخر شمعة مدتها دقيقتان. هذه أداة تحليل وليست توصية مالية.")

with st.sidebar:
    st.header("الإعدادات")
    st.subheader("مفاتيح Alpaca")
    st.caption("يمكنك إدخال المفاتيح هنا مباشرة. لا يتم حفظها في GitHub.")
    api_key = st.text_input("ALPACA API Key", value=ENV_API_KEY, type="password")
    secret_key = st.text_input("ALPACA Secret Key", value=ENV_SECRET_KEY, type="password")
    if api_key and secret_key:
        st.success("تم إدخال المفاتيح")
    else:
        st.warning("أدخل مفتاحي Alpaca أولًا")
    upload = st.file_uploader("ارفع ملف الأسهم TXT أو CSV", type=["txt", "csv"])
    symbols = symbols_from_text(upload.read().decode("utf-8")) if upload else load_symbols()
    st.write(f"عدد الأسهم: {len(symbols)} / 100")
    st.write(f"مصدر البيانات: Alpaca {FEED.upper()}")
    st.write(f"التحديث: كل {REFRESH_SECONDS} ثانية")
    st.info("IEX المجاني لا يمثل كامل السوق الأمريكي وقد تختلف تغطيته عن Nasdaq SIP.")

if not api_key or not secret_key:
    st.error("أدخل ALPACA API Key وALPACA Secret Key في الشريط الجانبي.")
    st.stop()
if not symbols:
    st.warning("أضف رموز الأسهم في symbols.txt.")
    st.stop()

st_autorefresh(interval=REFRESH_SECONDS * 1000, key="market_refresh")
try:
    with st.spinner("جاري جلب البيانات وتحليل السيولة..."):
        result = scan(symbols, api_key, secret_key)
except Exception as exc:
    st.error(f"حدث خطأ: {exc}")
    st.stop()

st.caption("آخر تحديث: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
if result.empty:
    st.warning("لا توجد بيانات كافية. تحقق من المفاتيح والسوق والرموز.")
    st.stop()

st.subheader("أفضل 5 أسهم")
for column, (_, row) in zip(st.columns(5), result.head(5).iterrows()):
    column.markdown(f"<div class='card'><h3>{row['الرمز']}</h3><b>{row['الحالة']}</b><br>الدرجة: {row['الدرجة']:.1f}<br>السعر: ${row['السعر']:.2f}<br>التغير: {row['تغير_دقيقتين']:+.2f}%<br>RVOL: {row['RVOL']:.2f}x<br>السيولة: ${row['قيمة_التداول']:,.0f}</div>", unsafe_allow_html=True)

view = result.copy()
for column in ["السعر", "VWAP", "RSI", "الدرجة", "RVOL", "تغير_دقيقتين", "نمو_الحجم"]:
    view[column] = view[column].round(2)
st.subheader("ترتيب جميع الأسهم")
st.dataframe(view, use_container_width=True, hide_index=True)
st.download_button("تحميل النتائج CSV", result.to_csv(index=False).encode("utf-8-sig"), "market_scan.csv", "text/csv")
