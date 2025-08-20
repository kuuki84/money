
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="KRW FX Dashboard (Streamlit Cloud)", layout="wide")
st.title("KRW 다중 통화 대시보드 (USD, JPY, EUR, CNY, CAD, AUD, NOK, SEK, GBP)")

st.caption("• 데이터 소스: Yahoo Finance (^DXY, KRW=X 등). 실패 시 동봉된 sample_data.csv를 사용합니다. "
           "• 적정환율은 간단 모델(USDKRW vs DXY의 1년 중앙값 비율)로 산출한 뒤 교차환율로 확장합니다. 학습/리서치용 예시입니다.")

import datetime as dt

# -----------------------------
# Data
# -----------------------------
@st.cache_data(show_spinner=True)
def fetch_data():
    try:
        import yfinance as yf
        ticks = {
            "USDKRW": "KRW=X",   # KRW per USD
            "USDJPY": "JPY=X",
            "USDCAD": "CAD=X",
            "USDNOK": "NOK=X",
            "USDSEK": "SEK=X",
            "USDCNY": "CNY=X",
            "EURUSD": "EURUSD=X",  # USD per EUR
            "GBPUSD": "GBPUSD=X",  # USD per GBP
            "AUDUSD": "AUDUSD=X",  # USD per AUD
            "DXY": "^DXY",
        }
        frames = {}
        for k, t in ticks.items():
            df = yf.download(t, period="2y", interval="1d", progress=False)
            if not df.empty:
                frames[k] = df["Adj Close"].tz_localize(None)
        if not frames:
            raise RuntimeError("No data from yfinance")
        idx = sorted(set().union(*[s.index for s in frames.values()]))
        out = pd.DataFrame(index=idx)
        for k, s in frames.items():
            out[k] = s.reindex(idx)
        out = out.dropna(how="any")
        out.index.name = "date"
        return out.reset_index()
    except Exception as e:
        # Fallback to bundled CSV
        df = pd.read_csv("sample_data.csv", parse_dates=["date"])
        df = df.dropna(how="any").sort_values("date")
        return df.reset_index(drop=True)

df = fetch_data()
if df.empty:
    st.error("데이터 수집 실패")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    period = st.selectbox("기간", ["1년", "6개월", "3개월", "YTD", "전체"], index=0)
    tolerance = st.slider("고/저평가 임계값(%)", 1.0, 10.0, 3.0, 0.5)
    st.markdown("---")
    st.write("표시 단위: **KRW/1 단위 외화**")
    st.write("EUR/GBP/AUD은 USD per X (XUSD), CAD/NOK/SEK/CNY/JPY는 USD per 1 USD (USDX)")

# -----------------------------
# Period filter
# -----------------------------
today = pd.to_datetime(df["date"]).max()
if period == "1년":
    start = today - pd.DateOffset(years=1)
elif period == "6개월":
    start = today - pd.DateOffset(months=6)
elif period == "3개월":
    start = today - pd.DateOffset(months=3)
elif period == "YTD":
    start = pd.Timestamp(f"{today.year}-01-01")
else:
    start = df["date"].min()

d = df[pd.to_datetime(df["date"]) >= start].copy().reset_index(drop=True)

# -----------------------------
# Fair value model (USDKRW vs DXY ratio, 1y median)
# -----------------------------
def fair_usdkrw_by_dxy(full_df: pd.DataFrame) -> float:
    one_year_start = pd.to_datetime(full_df["date"]).max() - pd.DateOffset(years=1)
    base = full_df[pd.to_datetime(full_df["date"]) >= one_year_start].copy()
    ratio_med = (base["USDKRW"] / base["DXY"]).median()
    dxy_now = float(full_df["DXY"].iloc[-1])
    return float(ratio_med * dxy_now)

fair_usdkrw = fair_usdkrw_by_dxy(df)
usdkrw_now = float(d["USDKRW"].iloc[-1])

# -----------------------------
# KRW per currency (live & fair) via cross
# -----------------------------
def krw_per_currency_row(row):
    krw = {}
    usdkrw = row["USDKRW"]
    krw["USD"] = usdkrw
    krw["JPY"] = usdkrw / row["USDJPY"]
    for k, col in [("CAD","USDCAD"), ("NOK","USDNOK"), ("SEK","USDSEK"), ("CNY","USDCNY")]:
        krw[k] = usdkrw / row[col]
    for k, col in [("EUR","EURUSD"), ("GBP","GBPUSD"), ("AUD","AUDUSD")]:
        krw[k] = usdkrw * row[col]
    return krw

latest = d.iloc[-1]
krw_now = krw_per_currency_row(latest)

def krw_fair_from_fair_usdkrw(row, fair_usdkrw):
    out = {}
    out["USD"] = fair_usdkrw
    out["JPY"] = fair_usdkrw / row["USDJPY"]
    for k, col in [("CAD","USDCAD"), ("NOK","USDNOK"), ("SEK","USDSEK"), ("CNY","USDCNY")]:
        out[k] = fair_usdkrw / row[col]
    for k, col in [("EUR","EURUSD"), ("GBP","GBPUSD"), ("AUD","AUDUSD")]:
        out[k] = fair_usdkrw * row[col]
    return out

krw_fair = krw_fair_from_fair_usdkrw(latest, fair_usdkrw)

# -----------------------------
# Gap & Score
# -----------------------------
def score_from_gap(gap_pct):
    x = -gap_pct  # 저평가(음수 gap)일수록 큰 점수
    base = 50 + (x * 4)  # 1% ≈ 4점
    return float(np.clip(base, 0, 100))

order = ["USD","JPY","EUR","CNY","CAD","AUD","NOK","SEK","GBP"]
records = []
for ccy in order:
    now_v = krw_now[ccy]
    fair_v = krw_fair[ccy]
    gap = (now_v / fair_v - 1) * 100.0
    records.append({
        "통화": ccy,
        "현재 환율 (KRW/1)": round(now_v, 2),
        "적정 환율 (KRW/1)": round(fair_v, 2),
        "괴리율 (%)": round(gap, 2),
        "단기 점수 (0~100)": round(score_from_gap(gap), 0),
    })

tbl = pd.DataFrame(records).set_index("통화").loc[order]

st.subheader("현재 환율 vs 적정 환율 vs 괴리율 (%)")
st.dataframe(tbl, use_container_width=True)

st.subheader("단기 환차익 점수 순위")
st.dataframe(tbl.sort_values("단기 점수 (0~100)", ascending=False), use_container_width=True)

# -----------------------------
# Charts
# -----------------------------
st.subheader("원/달러 vs 달러지수 (참고)")
fig, ax = plt.subplots(figsize=(9,3))
ax.plot(pd.to_datetime(d["date"]), d["USDKRW"], label="USDKRW (KRW per USD)")
ax.set_ylabel("KRW")
ax2 = ax.twinx()
ax2.plot(pd.to_datetime(d["date"]), d["DXY"], label="DXY", linestyle="--")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
st.pyplot(fig)

st.caption("※ 본 앱은 교육/연구용 예시입니다. 투자 책임은 이용자에게 있습니다.")
