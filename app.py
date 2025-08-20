
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="KRW FX Dashboard v4 (Mobile Compact)", layout="wide")
st.title("KRW 다중 통화 대시보드 v4")

# ---- Mobile-friendly style ----
st.markdown('''
<style>
/* reduce paddings */
.block-container { padding-top: 0.8rem; padding-bottom: 0.5rem; }
thead tr th { font-size: 14px !important; }
tbody tr td { font-size: 15px !important; padding-top: 6px !important; padding-bottom: 6px !important; }
</style>
''', unsafe_allow_html=True)

st.caption("• 통화별 적정환율 = 지난 1년간 **KRW/통화 ÷ DXY** 비율의 중앙값 × 현재 DXY.\n"
           "• 점수 = 모든 통화의 괴리율(%)을 0~100으로 정규화(저평가↑, 고평가↓).\n"
           "• 데이터: Yahoo Finance. 실패 시 sample_data.csv 사용. 연구/학습용.")

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
    st.subheader("옵션")
    period = st.selectbox("기간", ["1년", "6개월", "3개월", "YTD", "전체"], index=0)
    mobile_only = st.toggle("모바일 간단 보기", value=True)
    st.markdown("---")
    st.write("단위: KRW / 1단위 외화")

# Period filter
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
latest = krw_view.iloc[-1]

# Compute gaps & normalized score
order = ["USD","JPY","EUR","CNY","CAD","AUD","NOK","SEK","GBP"]
rows, gaps = [], []
for ccy in order:
    now_v = float(latest[ccy])
    fair_v = float(fair_map[ccy])
    gap = (now_v / fair_v - 1) * 100.0
    gaps.append(gap)
    rows.append([ccy, now_v, fair_v, gap])

gmin, gmax = float(np.min(gaps)), float(np.max(gaps))
den = (gmax - gmin) if gmax != gmin else 1.0

def norm_score(gap):
    return float(np.clip(100.0 * (gmax - gap) / den, 0, 100))

def fmt_krw(v: float) -> str:
    return f"{v:,.2f}" if v < 100 else f"{v:,.0f}"

records = []
for ccy, now_v, fair_v, gap in rows:
    records.append({
        "통화": ccy,
        "점수": int(round(norm_score(gap), 0)),
        "현재환율": fmt_krw(now_v),
        "적정환율": fmt_krw(fair_v),
        "_gap": gap,
    })

compact = pd.DataFrame(records).set_index("통화").loc[order].reset_index()

# ---- Mobile compact table (one screen) ----
st.subheader("모바일 간단 보기 (통화 | 점수 | 현재환율 | 적정환율)")
st.dataframe(
    compact.sort_values("점수", ascending=False)[["통화","점수","현재환율","적정환율"]],
    use_container_width=True, hide_index=True, height=420
)

if not mobile_only:
    # Detailed table & chart in non-mobile mode
    detailed = pd.DataFrame({
        "통화": [r[0] for r in rows],
        "현재 환율 (KRW/1)": [round(r[1], 2) for r in rows],
        "적정 환율 (KRW/1)": [round(r[2], 2) for r in rows],
        "괴리율 (%)": [round(r[3], 2) for r in rows],
        "정규화 점수 (0~100)": [round(norm_score(r[3]), 1) for r in rows],
    }).set_index("통화").loc[order]

    st.subheader("상세 표")
    st.dataframe(detailed.sort_values("정규화 점수 (0~100)", ascending=False),
                 use_container_width=True, height=500)

    st.subheader("참고: 원/달러 vs DXY")
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(pd.to_datetime(d["date"]), d["USDKRW"], label="USDKRW")
    ax.set_ylabel("KRW")
    ax2 = ax.twinx()
    ax2.plot(pd.to_datetime(d["date"]), d["DXY"], label="DXY", linestyle="--")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    st.pyplot(fig)

st.caption("※ 아이폰 한 화면에 맞춘 4열 테이블. 사이드바에서 '모바일 간단 보기' on/off 가능.")
