
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="KRW FX Dashboard v2 (Per-Currency Fair)", layout="wide")
st.title("KRW 다중 통화 대시보드 v2 (통화별 적정환율)")

st.caption("• 통화별 적정환율: 지난 1년간 **KRW/통화 ÷ DXY** 비율의 중앙값 × 현재 DXY.\n"
           "• 데이터: Yahoo Finance. 실패 시 sample_data.csv 사용. 연구/학습용 예시입니다.")

@st.cache_data(show_spinner=True)
def fetch():
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
            raise RuntimeError("no data")
        idx = sorted(set().union(*[s.index for s in frames.values()]))
        out = pd.DataFrame(index=idx)
        for k, s in frames.items():
            out[k] = s.reindex(idx)
        out.index.name = "date"
        out = out.dropna(how="any").reset_index()
        return out
    except Exception:
        df = pd.read_csv("sample_data.csv", parse_dates=["date"])
        return df.dropna(how="any").sort_values("date").reset_index(drop=True)

df = fetch()
if df.empty:
    st.error("데이터 수집 실패")
    st.stop()

# Sidebar
with st.sidebar:
    period = st.selectbox("기간", ["1년", "6개월", "3개월", "YTD", "전체"], index=0)
    tolerance = st.slider("고/저평가 임계값(%)", 1.0, 10.0, 3.0, 0.5)
    st.markdown("---")
    st.write("표시 단위: KRW / 1단위 외화")

# Filter period
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

# Build KRW per currency series across the dataframe
def build_krw_series(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    # KRW per 1 unit foreign currency
    out = pd.DataFrame({"date": x["date"]})
    out["USD"] = x["USDKRW"]
    out["JPY"] = x["USDKRW"] / x["USDJPY"]
    for k, col in [("CAD","USDCAD"), ("NOK","USDNOK"), ("SEK","USDSEK"), ("CNY","USDCNY")]:
        out[k] = x["USDKRW"] / x[col]
    for k, col in [("EUR","EURUSD"), ("GBP","GBPUSD"), ("AUD","AUDUSD")]:
        out[k] = x["USDKRW"] * x[col]
    out["DXY"] = x["DXY"]
    return out

krw_full = build_krw_series(df)
krw_view = build_krw_series(d)

# Per-currency fair value from last 1y ratio (KRW/ccy ÷ DXY)
def fair_per_currency(full_krw_df: pd.DataFrame) -> dict:
    end = pd.to_datetime(full_krw_df["date"]).max()
    start = end - pd.DateOffset(years=1)
    base = full_krw_df[pd.to_datetime(full_krw_df["date"]) >= start].copy()
    dxy_now = float(full_krw_df["DXY"].iloc[-1])
    fair = {}
    for ccy in ["USD","JPY","EUR","CNY","CAD","AUD","NOK","SEK","GBP"]:
        ratio_median = np.median(base[ccy].values / base["DXY"].values)
        fair[ccy] = float(ratio_median * dxy_now)
    return fair

fair_map = fair_per_currency(krw_full)

# Latest row for now values
latest = krw_view.iloc[-1]

def score_from_gap(gap_pct):
    x = -gap_pct
    base = 50 + x * 4
    return float(np.clip(base, 0, 100))

records = []
order = ["USD","JPY","EUR","CNY","CAD","AUD","NOK","SEK","GBP"]
for ccy in order:
    now_v = float(latest[ccy])
    fair_v = float(fair_map[ccy])
    gap = (now_v / fair_v - 1) * 100.0
    records.append({
        "통화": ccy,
        "현재 환율 (KRW/1)": round(now_v, 2),
        "적정 환율 (KRW/1)": round(fair_v, 2),
        "괴리율 (%)": round(gap, 2),
        "단기 점수 (0~100)": round(score_from_gap(gap), 0),
    })

tbl = pd.DataFrame(records).set_index("통화").loc[order]

st.subheader("현재 vs 적정 vs 괴리율 (통화별)")
st.dataframe(tbl, use_container_width=True)

st.subheader("단기 환차익 점수 순위")
st.dataframe(tbl.sort_values("단기 점수 (0~100)", ascending=False), use_container_width=True)

# Simple ref chart
st.subheader("참고: 원/달러 vs DXY")
fig, ax = plt.subplots(figsize=(9,3))
ax.plot(pd.to_datetime(d["date"]), d["USDKRW"], label="USDKRW")
ax.set_ylabel("KRW")
ax2 = ax.twinx()
ax2.plot(pd.to_datetime(d["date"]), d["DXY"], label="DXY", linestyle="--")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
st.pyplot(fig)

st.caption("※ 계산 로직: 통화별 (KRW/통화 ÷ DXY)의 1년 중앙값을 사용하여 적정환율 산정.")
