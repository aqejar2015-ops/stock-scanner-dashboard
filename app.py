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
BINANCE_URL = "https://api.binance.com"

st.set_page_config(page_title="Market Liquidity Scanner", page_icon="📈", layout="wide")
st.markdown("<style>html,body,[class*=css]{direction:rtl;text-align:right}.card{background:#101827;border:1px solid #263449;border-radius:12px;padding:12px}</style>", unsafe_allow_html=True)


def symbols_from_text(text):
    return list(dict.fromkeys(x.strip().upper() for x in text.replace(",", "\n").replace("،", "\n").splitlines() if x.strip()))[:100]


def load_symbols(filename):
    path = BASE_DIR / filename
    return symbols_from_text(path.read_text(encoding="utf-8")) if path.exists() else []


def indicators(df):
    df = df.copy()
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["close", "volume"])
    typical = (df.high + df.low + df.close) / 3
    cumulative_volume = df.volume.replace(0, np.nan).cumsum()
    df["vwap_calc"] = (typical * df.volume).cumsum() / cumulative_volume
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


def alpaca_get(path, params, api_key, secret_key):
    response = requests.get(
        DATA_URL + path,
        headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key},
        params=params,
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Alpaca API {response.status_code}: {response.text[:300]}")
    return response.json()


def fetch_stock_bars(symbols, api_key, secret_key):
    now = datetime.now(timezone.utc)
    payload = alpaca_get(
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


def fetch_stock_quotes(symbols, api_key, secret_key):
    payload = alpaca_get("/v2/stocks/quotes/latest", {"symbols": ",".join(symbols), "feed": FEED}, api_key, secret_key)
    output = {}
    for symbol, quote in payload.get("quotes", {}).items():
        bid, ask = float(quote.get("bp") or 0), float(quote.get("ap") or 0)
        output[symbol] = {"spread": ((ask - bid) / ((ask + bid) / 2) * 100) if bid and ask else np.nan}
    return output


def fetch_crypto_bars(symbols):
    output = {}
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 8 * 60 * 60 * 1000
    for symbol in symbols:
        symbol = symbol.replace("/", "").replace("-", "")
        try:
            response = requests.get(
                f"{BINANCE_URL}/api/v3/klines",
                params={"symbol": symbol, "interval": "1m", "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                timeout=20,
            )
            if response.status_code != 200:
                continue
            rows = response.json()
            if len(rows) < 4:
                continue
            frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
            frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
            frame = frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            for column in ["open", "high", "low", "close", "volume"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            # Binance spot has 1m candles; aggregate completed candles into the same 2m rhythm.
            frame = frame.resample("2min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            current_bucket = pd.Timestamp.now(tz="UTC").floor("2min")
            frame = frame[frame.index < current_bucket]
            if len(frame) >= 2:
                output[symbol] = indicators(frame.reset_index())
        except (requests.RequestException, ValueError, TypeError):
            continue
    return output


def score(symbol, frame, quote=None, market="stock"):
    if len(frame) < 2:
        return None
    previous, current = frame.iloc[-2], frame.iloc[-1]
    price = float(current.close)
    price_change = (price / float(previous.close) - 1) * 100 if previous.close else 0
    volume_growth = (float(current.volume) / float(previous.volume) - 1) * 100 if previous.volume else 0
    rvol, vwap, rsi = float(current.rvol), float(current.vwap_calc), float(current.rsi)
    spread = float((quote or {}).get("spread", np.nan))
    total = 0.0
    reasons = []
    if price_change > 0:
        total += min(25, price_change * 8); reasons.append("السعر يرتفع")
    else:
        total += max(-15, price_change * 4)
    if volume_growth > 0:
        total += min(25, volume_growth / 4); reasons.append("الحجم يتزايد")
    else:
        total += max(-10, volume_growth / 8)
    if rvol >= 1.5:
        total += 15; reasons.append("RVOL مرتفع")
    elif rvol >= 1:
        total += 7
    if price > vwap:
        total += 10; reasons.append("فوق VWAP")
    if price > current.ema9 > current.ema20:
        total += 10; reasons.append("اتجاه EMA إيجابي")
    if 50 <= rsi <= 75:
        total += 8; reasons.append("RSI صحي")
    elif rsi > 85:
        total -= 8; reasons.append("تشبع محتمل")
    if market == "stock":
        if np.isnan(spread) or spread <= MAX_SPREAD_PERCENT:
            total += 7
        else:
            total -= 15; reasons.append("سبريد مرتفع")
    if price_change <= 0 and volume_growth > 20:
        total -= 20; reasons.append("ضغط بيع محتمل")
    status = "إيجابي" if total >= 55 else "مراقبة" if total >= 35 else "ضعيف"
    return {"الرمز": symbol, "السعر": price, "تغير_دقيقتين": price_change, "قيمة_التداول": float(current.dollar_volume), "نمو_الحجم": volume_growth, "RVOL": rvol, "VWAP": vwap, "RSI": rsi, "السبريد": spread, "الدرجة": max(0, min(100, total)), "الحالة": status, "السبب": "، ".join(reasons)}


def scan_stocks(symbols, api_key, secret_key):
    bars = fetch_stock_bars(symbols, api_key, secret_key)
    quotes = fetch_stock_quotes(symbols, api_key, secret_key)
    rows = [score(symbol, frame, quotes.get(symbol, {}), "stock") for symbol, frame in bars.items()]
    rows = [row for row in rows if row and row["السعر"] >= MIN_PRICE and row["قيمة_التداول"] >= MIN_DOLLAR_VOLUME]
    return pd.DataFrame(rows).sort_values(["الدرجة", "قيمة_التداول"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def scan_crypto(symbols):
    bars = fetch_crypto_bars(symbols)
    rows = [score(symbol, frame, {}, "crypto") for symbol, frame in bars.items()]
    return pd.DataFrame(rows).sort_values(["الدرجة", "قيمة_التداول"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def display_results(result, positive_title, negative_title):
    if result.empty:
        st.warning("لا توجد بيانات كافية لهذه القائمة أو لا توجد رموز صحيحة.")
        return
    positive = result[result["تغير_دقيقتين"] > 0].head(5)
    negative = result[result["تغير_دقيقتين"] < 0].sort_values(["الدرجة", "تغير_دقيقتين"], ascending=[True, True]).head(5)
    st.subheader(positive_title)
    for column, (_, row) in zip(st.columns(5), positive.iterrows()):
        column.markdown(f"<div class='card'><h3>{row['الرمز']}</h3><b>{row['الحالة']}</b><br>الدرجة: {row['الدرجة']:.1f}<br>السعر: ${row['السعر']:.6g}<br>التغير: {row['تغير_دقيقتين']:+.2f}%<br>RVOL: {row['RVOL']:.2f}x<br>السيولة: ${row['قيمة_التداول']:,.0f}</div>", unsafe_allow_html=True)
    st.subheader(negative_title)
    for column, (_, row) in zip(st.columns(5), negative.iterrows()):
        column.markdown(f"<div class='card'><h3>{row['الرمز']}</h3><b>ضغط سلبي</b><br>الدرجة: {row['الدرجة']:.1f}<br>السعر: ${row['السعر']:.6g}<br>التغير: {row['تغير_دقيقتين']:+.2f}%<br>RVOL: {row['RVOL']:.2f}x<br>السيولة: ${row['قيمة_التداول']:,.0f}</div>", unsafe_allow_html=True)
    view = result.copy()
    for column in ["السعر", "VWAP", "RSI", "الدرجة", "RVOL", "تغير_دقيقتين", "نمو_الحجم"]:
        view[column] = view[column].round(4)
    st.subheader("ترتيب جميع النتائج")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("تحميل النتائج CSV", result.to_csv(index=False).encode("utf-8-sig"), "market_scan.csv", "text/csv", key=f"download_{positive_title}")


st.title("📈 ماسح السيولة والإيجابية - الأسهم والعملات الرقمية")
st.caption("تحديث افتراضي كل دقيقتين. النتائج ترتيب تحليلي وليست توصية مالية أو إشارة مضمونة.")

with st.sidebar:
    st.header("الإعدادات")
    st.subheader("مفاتيح Alpaca للأسهم")
    st.caption("أدخل المفاتيح هنا. بيانات Binance العامة لا تحتاج مفتاحًا.")
    api_key = st.text_input("ALPACA API Key", value=ENV_API_KEY, type="password")
    secret_key = st.text_input("ALPACA Secret Key", value=ENV_SECRET_KEY, type="password")
    if api_key and secret_key:
        st.success("تم إدخال مفاتيح Alpaca")
    stock_upload = st.file_uploader("قائمة الأسهم TXT/CSV", type=["txt", "csv"], key="stock_upload")
    crypto_upload = st.file_uploader("قائمة العملات TXT/CSV", type=["txt", "csv"], key="crypto_upload")
    stock_symbols = symbols_from_text(stock_upload.read().decode("utf-8")) if stock_upload else load_symbols("symbols.txt")
    crypto_symbols = symbols_from_text(crypto_upload.read().decode("utf-8")) if crypto_upload else load_symbols("crypto_symbols.txt")
    st.write(f"الأسهم: {len(stock_symbols)} / 100")
    st.write(f"العملات: {len(crypto_symbols)} / 100")
    st.write(f"التحديث: كل {REFRESH_SECONDS} ثانية")
    st.info("Binance Spot API عامة ولا تحتاج مفتاحًا. استخدم رموز USDT مثل BTCUSDT وETHUSDT.")

st_autorefresh(interval=REFRESH_SECONDS * 1000, key="market_refresh")
tab_stocks, tab_crypto = st.tabs(["🇺🇸 الأسهم الأمريكية", "₿ العملات الرقمية"])

with tab_stocks:
    if not api_key or not secret_key:
        st.error("أدخل مفاتيح Alpaca في الشريط الجانبي لتحليل الأسهم.")
    elif not stock_symbols:
        st.warning("أضف رموز الأسهم في symbols.txt أو ارفع قائمة الأسهم.")
    else:
        try:
            with st.spinner("جاري تحليل الأسهم الأمريكية..."):
                stock_result = scan_stocks(stock_symbols, api_key, secret_key)
            st.caption("آخر تحديث: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            display_results(stock_result, "أفضل 5 أسهم إيجابية", "5 أسهم باتجاه سلبي")
        except Exception as exc:
            st.error(f"خطأ في بيانات الأسهم: {exc}")

with tab_crypto:
    if not crypto_symbols:
        st.warning("أضف رموز العملات في crypto_symbols.txt أو ارفع قائمة العملات.")
    else:
        with st.spinner("جاري تحليل العملات من Binance..."):
            crypto_result = scan_crypto(crypto_symbols)
        st.caption("آخر تحديث Binance: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
        display_results(crypto_result, "أفضل 5 عملات إيجابية", "أفضل 5 عملات سلبية للشورت")
