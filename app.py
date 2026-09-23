import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

ENV_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
DEFAULT_REFERENCES = "GRML,IMCC,WHLR"
DEFAULT_UNIVERSE = "BENF,IPDN,VTGN,DCOY,CAPS,BFRG,ZONE,TLSI,WAFU,IONQ,EDVA,SOS,SQFT,NNVC,EDIT,QNT,RTB,PAAI,SRFM,INFQ,ACRS,QBTS,AGPU,HAO,MYSE,VGAS,ASTC"
REFRESH_SECONDS = 120
BAR_MINUTES = 2
LOOKBACK_DAYS = 7
PATTERN_LENGTH = 30

st.set_page_config(page_title="US Stock Pre-Rise Scanner", page_icon="📈", layout="wide")
st.title("📈 US Stock Pre-Rise Scanner")
st.caption("يقارن نمط آخر 30 شمعة مع الأنماط التي سبقت ارتفاعات تاريخية في الأسهم المرجعية")


def parse_symbols(text):
    return list(dict.fromkeys(s.strip().upper() for s in text.replace("\n", ",").replace("،", ",").split(",") if s.strip()))


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
def fetch_bars(symbol, api_key):
    start, end = date_range()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{BAR_MINUTES}/minute/{start}/{end}"
    try:
        response = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key},
            timeout=30,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code != 200:
            message = payload.get("error") or payload.get("message") or response.text[:300]
            return pd.DataFrame(), f"{symbol}: HTTP {response.status_code} - {message}"
        frame = clean_bars(payload.get("results", []))
        if frame.empty:
            return pd.DataFrame(), f"{symbol}: لا توجد شموع في الفترة المطلوبة"
        return frame, ""
    except requests.RequestException as error:
        return pd.DataFrame(), f"{symbol}: خطأ اتصال - {error}"


def download_data(symbols, api_key):
    data, errors = {}, []
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        frame, error = fetch_bars(symbol, api_key)
        if not frame.empty:
            data[symbol] = frame
        if error:
            errors.append(error)
        progress.progress((i + 1) / len(symbols))
    progress.empty()
    return data, errors


def zscore(values):
    values = np.nan_to_num(np.asarray(values, dtype=float))
    std = values.std()
    return np.zeros(len(values)) if std == 0 else (values - values.mean()) / std


def pattern_from_window(df):
    if len(df) < PATTERN_LENGTH + 1:
        return None
    returns = np.diff(np.log(df["Close"].tail(PATTERN_LENGTH + 1).to_numpy(float)))
    return zscore(returns)


def correlation(a, b):
    if a is None or b is None or len(a) != len(b):
        return np.nan
    value = np.corrcoef(a, b)[0, 1]
    return float(value) if not np.isnan(value) else np.nan


def detect_pre_rise_patterns(df, min_rise_pct, forward_bars, max_patterns=20):
    patterns = []
    if len(df) < PATTERN_LENGTH + forward_bars + 1:
        return patterns
    closes = df["Close"].to_numpy(float)
    for end in range(PATTERN_LENGTH, len(df) - forward_bars):
        start = end - PATTERN_LENGTH
        before = closes[end]
        future_high = np.max(closes[end + 1:end + forward_bars + 1])
        rise_pct = (future_high / before - 1) * 100
        current_return = (before / closes[start] - 1) * 100
        if rise_pct >= min_rise_pct and current_return < min_rise_pct * 0.75:
            candidate = pattern_from_window(df.iloc[start:end + 1])
            if candidate is not None:
                patterns.append({"pattern": candidate, "rise_pct": rise_pct, "time": df.index[end]})
    patterns.sort(key=lambda x: x["rise_pct"], reverse=True)
    selected = []
    for item in patterns:
        if all(abs((item["time"] - old["time"]).total_seconds()) > 60 * BAR_MINUTES * 8 for old in selected):
            selected.append(item)
        if len(selected) >= max_patterns:
            break
    return selected


def find_best_pre_rise_match(candidate_df, reference_data, min_rise_pct, forward_bars):
    candidate = pattern_from_window(candidate_df)
    if candidate is None:
        return np.nan, np.nan, None, 0
    matches = []
    for symbol, df in reference_data.items():
        for item in detect_pre_rise_patterns(df, min_rise_pct, forward_bars):
            score = correlation(candidate, item["pattern"])
            if not np.isnan(score):
                matches.append((score, item["rise_pct"], symbol, item["time"]))
    if not matches:
        return np.nan, np.nan, None, 0
    matches.sort(reverse=True)
    best = matches[0]
    confidence = max(0, min(100, (best[0] + 1) * 50))
    return best[0], confidence, f"{best[2]} ({best[3].strftime('%Y-%m-%d %H:%M')})", len(matches)


def stats(df):
    first, last = float(df.Close.iloc[0]), float(df.Close.iloc[-1])
    recent = df.tail(PATTERN_LENGTH)
    average_volume = df.Volume.tail(50).mean()
    return {
        "price": last,
        "week_return": (last / first - 1) * 100,
        "recent_return": (recent.Close.iloc[-1] / recent.Close.iloc[0] - 1) * 100 if len(recent) > 1 else 0,
        "volume_ratio": float(df.Volume.iloc[-1] / average_volume) if average_volume else 0,
    }


def rank_candidates(data, universe, references, min_rise_pct, forward_bars):
    reference_data = {s: data[s] for s in references if s in data}
    rows = []
    for symbol in universe:
        if symbol in references or symbol not in data:
            continue
        item = stats(data[symbol])
        corr, confidence, example, match_count = find_best_pre_rise_match(data[symbol], reference_data, min_rise_pct, forward_bars)
        if np.isnan(corr):
            continue
        score = corr + min(max(item["volume_ratio"] - 1, -1), 1) * 0.03
        rows.append({"الرمز": symbol, "الثقة التقريبية %": confidence, "التشابه": score, "النمط المشابه": example, "السعر": item["price"], "تغير الفترة %": item["week_return"], "تغير آخر 30 شمعة %": item["recent_return"], "الحجم مقارنة بالمتوسط": item["volume_ratio"], "عدد المطابقات": match_count})
    return pd.DataFrame(rows).sort_values(["الثقة التقريبية %", "عدد المطابقات"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


with st.sidebar:
    st.header("⚙��� الإعدادات")
    st.subheader("🔐 مفتاح Polygon/Massive")
    st.caption("يمكنك لصق المفتاح هنا مباشرة. لا يتم حفظه في GitHub.")
    api_key = st.text_input("API Key", value=ENV_API_KEY, type="password", help="الصق مفتاح Polygon/Massive هنا")
    if api_key:
        st.success("تم إدخال المفتاح")
    else:
        st.warning("أدخل المفتاح أولاً")
    reference_text = st.text_input("الأسهم الأساسية الثلاثة", DEFAULT_REFERENCES)
    references = parse_symbols(reference_text)
    if len(references) != 3:
        st.warning("اكتب ثلاثة رموز بالضبط، مفصولة بفاصلة.")
    universe_text = st.text_area("الأسهم المراد فحصها", DEFAULT_UNIVERSE, height=220)
    top_n = st.slider("عدد النتائج", 3, 10, 3)
    min_rise_pct = st.slider("الارتفاع التاريخي المطلوب بعد النمط %", 3.0, 30.0, 8.0, 0.5)
    forward_bars = st.slider("عدد شموع قياس الارتفاع", 3, 30, 10)
    st.info("يقارن آخر 30 شمعة للمرشح مع نوافذ سبقت ارتفاعاً في الأسهم الأساسية.")

if not api_key:
    st.error("الصق مفتاح Polygon/Massive في خانة API Key على اليسار.")
    st.stop()

st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
universe = parse_symbols(universe_text)
if len(references) != 3:
    st.stop()

symbols = list(dict.fromkeys(references + universe))
with st.spinner("جاري جلب البيانات واكتشاف أنماط ما قبل الارتفاع..."):
    data, errors = download_data(symbols, api_key)

if errors:
    with st.expander("تفاصيل جلب البيانات"):
        st.write("\n".join(errors[:20]))

if not data:
    st.error("لم تصل أي بيانات. تحقق من المفتاح والخطة واتصال الإنترنت.")
    st.stop()

st.caption("آخر تحديث: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
cols = st.columns(3)
for col, symbol in zip(cols, references):
    with col:
        st.subheader(symbol)
        if symbol not in data:
            st.warning("لا توجد بيانات")
            continue
        item = stats(data[symbol])
        st.metric("السعر", f"${item['price']:.4f}")
        st.metric("تغير الفترة", f"{item['week_return']:.2f}%")
        st.caption(f"الحجم الحالي: {item['volume_ratio']:.2f}x المتوسط")

st.divider()
st.header("🏆 الأسهم التي تشبه نمط ما قبل الارتفاع")
result = rank_candidates(data, universe, references, min_rise_pct, forward_bars)
if result.empty:
    st.warning("لم نجد نمطاً مطابقاً. خفّض حد الارتفاع أو زِد فترة البيانات/عدد الأسهم.")
else:
    top = result.head(top_n)
    st.success("المرشحون الحاليون: " + ", ".join(top["الرمز"].tolist()))
    st.dataframe(top.style.format({"الثقة التقريبية %": "{:.1f}%", "التشابه": "{:.3f}", "السعر": "${:.4f}", "تغير الفترة %": "{:.2f}%", "تغير آخر 30 شمعة %": "{:.2f}%", "الحجم مقارنة بالمتوسط": "{:.2f}x"}), use_container_width=True, hide_index=True)
    selected = st.selectbox("اختر سهماً للرسم", top["الرمز"].tolist())
    st.line_chart(data[selected][["Close"]].rename(columns={"Close": "السعر"}), use_container_width=True)

st.warning("هذا مؤشر تشابه تاريخي وليس ضماناً لارتفاع السهم أو توصية مالية.")
