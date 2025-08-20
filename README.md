
# KRW Multi-Currency FX Dashboard (Streamlit Cloud)

**통화**: USD, JPY, EUR, CNY, CAD, AUD, NOK, SEK, GBP  
**기능**: 현재 환율, 적정 환율(간단 모델), 괴리율(%), 단기 점수(0~100), 차트

## How to deploy (Streamlit Cloud)
1. GitHub에 `app.py`, `requirements.txt`, `sample_data.csv` 업로드
2. Streamlit Cloud에서 GitHub 연결 → Deploy
3. 앱 접속 URL을 아이폰/Safari에서 열면 끝

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
