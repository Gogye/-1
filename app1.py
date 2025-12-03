import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
import altair as alt
import ta
import os 
import uuid

# --------- Naver news crawler dependencies
import re
import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import ReadTimeout, ConnectTimeout, Timeout, RequestException
from urllib.parse import urlsplit, urlunsplit, urlencode

from st_clickable_images import clickable_images



# =========================
# 1. 페이지 설정 & 전역 스타일
# =========================
st.set_page_config(
    page_title="투자위키 - InvestWiki",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* ==========================================================================
       1. 전체 페이지 레이아웃 & 테마
       ========================================================================== */
    body { background-color: #f8f9fa; }
    
    .main-logo-text {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #004aad, #cb6ce6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        margin-top: 2rem;
    }
    
    /* ==========================================================================
       2. 사이드바 스타일 (다크 테마)
       ========================================================================== */
    [data-testid="stSidebar"] {
        background-color: #2B2D3E;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] input {
        color: #000000 !important;
    }

    /* ==========================================================================
       사이드바 버튼 스타일 (강력한 강제 적용 버전)
       ========================================================================== */
    
    /* 1. [선택 안 된 버튼] (Secondary) 스타일 */
    /* 버튼 컨테이너, 내부 div, 텍스트 모두 타겟팅 */
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] button[kind="secondary"] > div,
    section[data-testid="stSidebar"] button[kind="secondary"] p {
        background-color: #FFFFFF !important; /* 배경: 흰색 */
        color: #000000 !important;            /* 글자: 검정색 */
        border-color: #E0E0E0 !important;     /* 테두리: 연회색 */
    }
    
    /* Secondary 버튼 자체에만 border 적용 (중복 방지) */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        border: 1px solid #E0E0E0 !important;
    }

    /* 마우스 올렸을 때 (Hover) */
    section[data-testid="stSidebar"] button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] button[kind="secondary"]:hover > div,
    section[data-testid="stSidebar"] button[kind="secondary"]:hover p {
        background-color: #F5F5F5 !important;
        color: #000000 !important;
        border-color: #BDBDBD !important;
    }

    /* -------------------------------------------------------------------------- */

    /* 2. [선택된 버튼] (Primary) 스타일 */
    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="primary"] > div,
    section[data-testid="stSidebar"] button[kind="primary"] p {
        background-color: #2E86C1 !important; /* 배경: 파란색 */
        color: #FFFFFF !important;            /* 글자: 흰색 */
        border: none !important;
    }

    /* 마우스 올렸을 때 (Hover) */
    section[data-testid="stSidebar"] button[kind="primary"]:hover,
    section[data-testid="stSidebar"] button[kind="primary"]:hover > div,
    section[data-testid="stSidebar"] button[kind="primary"]:hover p {
        background-color: #1B4F72 !important; /* 더 진한 파란색 */
        color: #FFFFFF !important;
    }
    
    /* 버튼 공통 크기 설정 */
    section[data-testid="stSidebar"] button {
        width: 100%;
        border-radius: 8px !important;
        height: auto !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 2. 헬퍼 함수 (이미지 로드, 데이터 로드)
# =========================
@st.cache_data
def get_image_base64_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            encoded_string = base64.b64encode(response.content).decode()
            return f"data:image/png;base64,{encoded_string}"
    except:
        pass
    return None

pinpoints_df = pd.DataFrame({
    "Date": ["2024-06-05", "2024-10-10"],
    "Event": ["Vision Pro 발표", "신제품 출시"],
    "Content": ["Apple이 Vision Pro를 발표했습니다.", "Apple이 새로운 제품을 출시했습니다."],
    "Link": [
        "https://www.apple.com/newsroom/2024/06/apple-unveils-vision-pro-revolutionary-spatial-computing-platform/",
        "https://www.apple.com/newsroom/2024/10/apple-announces-new-products/",
    ],
})

# 인기 종목 리스트 (전역 변수)
ALL_POPULAR_STOCKS = [
    ("삼성전자", "005930"), ("셀트리온", "068270"), ("HMM", "011200"),
    ("애플", "AAPL"), ("마이크로소프트", "MSFT"), ("알파벳 A", "GOOGL"),
    ("알파벳 C", "GOOG"), ("아마존", "AMZN"), ("엔비디아", "NVDA"),
    ("메타", "META"), ("TSMC", "TSM"), ("테슬라", "TSLA"),
    ("현대차", "005380"), ("LG에너지솔루션", "373220"), ("SK하이닉스", "000660"),
    ("기아", "000270"), ("POSCO홀딩스", "005490"), ("KB금융", "105560"),
    ("신한지주", "055550"), ("카카오", "035720"), ("NAVER", "035420")
]

if "popular_indices" not in st.session_state:
    st.session_state.popular_indices = list(range(len(ALL_POPULAR_STOCKS)))

@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        df = fdr.DataReader(ticker, start_date, end_date)
        df = df.dropna()
        if df.empty: return None
        return df.copy()
    except: return None

# =========================
# 3. 알고리즘 함수들
# =========================
def apply_smoothing_and_phase(df, window_length, polyorder):
    df = df.copy()
    if len(df) < window_length:
        df["Smooth"] = df["Close"]
    else:
        df["Smooth"] = savgol_filter(df["Close"], window_length=window_length, polyorder=polyorder)
    df["Slope"] = np.gradient(df["Smooth"])
    df["Phase"] = df["Slope"].apply(lambda s: "상승" if s > 0 else "하락")
    return df

def apply_box_range(df, min_hits, window):
    df = df.copy()
    if df.empty: return df
    p_min, p_max = df["Close"].min(), df["Close"].max()
    limit = (p_max - p_min) / 25
    diffs = df["Close"].diff().abs()
    min_step = diffs[diffs > 0].min()
    if pd.isna(min_step): min_step = 10
    exponent = int(math.floor(math.log10(min_step)))
    step = 10 ** exponent if exponent >= 1 else 10

    for k in np.arange(p_min, p_max, step):
        crossings = [False] * len(df)
        for i in range(1, len(df)):
            y0, y1 = df["Close"].iloc[i-1], df["Close"].iloc[i]
            if (y0 - k) * (y1 - k) <= 0:
                crossings[i-1] = True; crossings[i] = True
        if len(crossings) <= window: continue
        for i in range(1, len(crossings) - window):
            if sum(crossings[i:i+window]) >= min_hits:
                if abs(df["Close"].iloc[i+window] - df["Close"].iloc[i]) <= limit:
                    df.loc[df.index[i:i+min_hits], "Phase"] = "박스권"
    
    if len(df) <= window: return df
    for i in range(len(df) - window):
        window_prices = df["Close"].iloc[i:i+window]
        window_mean = window_prices.mean()
        upper = window_mean + limit
        lower = window_mean - limit
        if window_prices.max() <= upper and window_prices.min() >= lower:
            df.loc[df.index[i:i+window], "Phase"] = "박스권"
    return df

def merge_short_phases(df, min_days):
    df = df.copy()
    if "Phase" not in df.columns or df.empty: return df
    df["group_id"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    df["group_size"] = df.groupby("group_id")["Phase"].transform("size")
    unique_ids = df["group_id"].unique()
    if len(unique_ids) < 2: return df
    min_gid = df["group_id"].min(); max_gid = df["group_id"].max()
    for gid in unique_ids:
        mask = df["group_id"] == gid
        size = df.loc[mask, "group_size"].iloc[0]
        if size <= min_days and gid > min_gid:
            if gid == max_gid: continue
            g_min, g_max = df.loc[mask, "Close"].min(), df.loc[mask, "Close"].max()
            if g_max - g_min >= (df["Close"].max() - df["Close"].min()) / 5: continue
            prev_phase = df.loc[df["group_id"] == gid - 1, "Phase"].iloc[0]
            next_phase = df.loc[df["group_id"] == gid + 1, "Phase"].iloc[0]
            if prev_phase != "박스권": df.loc[mask, "Phase"] = prev_phase
            elif next_phase != "박스권": df.loc[mask, "Phase"] = next_phase
    return df

def adjust_change_points(df, adjust_window):
    df = df.copy()
    if "Phase" not in df.columns or df.empty or len(df) < adjust_window: return df
    df["group_id"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    change_points = df.index[df["Phase"] != df["Phase"].shift()]
    if len(change_points) < 2: return df
    for cp in change_points:
        cp_idx = df.index.get_loc(cp)
        if cp_idx == 0: continue
        current_phase = df.loc[cp, "Phase"]
        prev_phase = df.loc[df.index[cp_idx - 1], "Phase"]
        start_win = max(0, cp_idx - adjust_window)
        end_win = min(len(df), cp_idx + adjust_window + 1)
        window_data = df.iloc[start_win:end_win]
        if window_data.empty: continue
        if current_phase == "상승":
            local_min_idx = window_data["Close"].idxmin()
            local_min_pos = df.index.get_loc(local_min_idx)
            if local_min_pos > cp_idx: df.loc[df.index[cp_idx:local_min_pos], "Phase"] = prev_phase
            elif local_min_pos < cp_idx: df.loc[df.index[local_min_pos:cp_idx], "Phase"] = "상승"
        elif current_phase == "하락":
            local_max_idx = window_data["Close"].idxmax()
            local_max_pos = df.index.get_loc(local_max_idx)
            if local_max_pos > cp_idx: df.loc[df.index[cp_idx:local_max_pos], "Phase"] = prev_phase
            elif local_max_pos < cp_idx: df.loc[df.index[local_max_pos:cp_idx], "Phase"] = "하락"
    return df

def detect_market_phases(df, window_length, polyorder, min_days1, min_days2, adjust_window, min_hits, box_window):
    df_res = df.copy()
    df_res = apply_smoothing_and_phase(df_res, window_length, polyorder)
    df_res = apply_box_range(df_res, min_hits, box_window)
    df_res = merge_short_phases(df_res, min_days1)
    df_res = adjust_change_points(df_res, adjust_window)
    df_res = merge_short_phases(df_res, min_days2)
    return df_res

# =========================
# 4. 시각화 함수들
# =========================
@st.cache_data
def get_stock_name(ticker):
    """
    티커(종목코드)를 입력받아 종목명(한글/영어)을 반환하는 함수
    1. 인기 종목 리스트에서 먼저 검색
    2. 없으면 KRX 전체 리스트에서 검색
    3. 그래도 없으면 티커 그대로 반환
    """
    ticker = ticker.upper().strip() # 대문자 변환 및 공백 제거

    ALL_POPULAR_STOCKS = [("삼성전자", "005930"), ("HMM", "011200"), ('셀트리온',"068270")]
    for name, code in ALL_POPULAR_STOCKS:
        if code == ticker:
            return name
    
    # 2. KRX(한국시장) 전체 리스트에서 찾기 (캐싱됨)
    try:
        df_krx = fdr.StockListing('KRX')
        # Code가 일치하는 행 찾기
        row = df_krx[df_krx['Code'] == ticker]
        if not row.empty:
            return row.iloc[0]['Name']
    except:
        pass

    # 3. 미국 주식 등 못 찾은 경우 그냥 티커 반환
    return ticker

def display_metrics(df):
    if len(df) < 2: return
    latest = df.iloc[-1]; prev = df.iloc[-2]
    diff = latest["Close"] - prev["Close"]
    pct = (diff / prev["Close"]) * 100
    
    st.markdown(f"""
    <div style="padding:15px; background:white; border-radius:10px; border:1px solid #ddd; display:flex; gap:20px; align-items:center; margin-bottom:20px;">
        <div>
            <span style="color:#666; font-size:0.9rem;">현재 주가</span><br>
            <span style="font-size:1.8rem; font-weight:bold;">{latest['Close']:,.0f}원</span>
        </div>
        <div style="color:{'red' if diff > 0 else 'blue'};">
            <span style="font-size:1.2rem; font-weight:bold;">{diff:,.0f}원 ({pct:+.2f}%)</span>
        </div>
        <div style="margin-left:auto; text-align:right;">
             <span style="color:#666; font-size:0.8rem;">거래량</span> <span style="font-weight:bold;">{latest['Volume']:,.0f}</span><br>
             <span style="color:#666; font-size:0.8rem;">RSI(14)</span> <span style="font-weight:bold;">{ta.momentum.RSIIndicator(df["Close"]).rsi().iloc[-1]:.1f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def visualize_candlestick(df):
    df_r = df.reset_index().rename(columns={"index":"Date"})
    base = alt.Chart(df_r).encode(x=alt.X("Date:T", axis=alt.Axis(format="%Y-%m-%d")))
    rule = base.mark_rule().encode(
        y=alt.Y("Low:Q", scale=alt.Scale(zero=False)), y2="High:Q",
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff"))
    )
    bar = base.mark_bar().encode(
        y="Open:Q", y2="Close:Q",
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff")),
        tooltip=["Date:T", "Open", "Close", "High", "Low"]
    )
    return (rule + bar).properties(height=350).interactive()

def visualize_technical_indicators(df):
    df = df.copy()
    if len(df) < 30: return alt.Chart(pd.DataFrame()).mark_text(text="데이터 부족")
    
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_h"] = bb.bollinger_hband(); df["bb_l"] = bb.bollinger_lband()
    rsi = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
    df["rsi"] = rsi
    
    df_r = df.reset_index().rename(columns={"index":"Date"})
    base = alt.Chart(df_r).encode(x="Date:T")
    
    bb_c = (base.mark_line(color="black").encode(y=alt.Y("Close:Q", scale=alt.Scale(zero=False))) + 
            base.mark_area(opacity=0.2).encode(y="bb_l:Q", y2="bb_h:Q")).properties(height=200, title="볼린저 밴드")
    
    rsi_c = (base.mark_line(color="purple").encode(y=alt.Y("rsi:Q", scale=alt.Scale(domain=[0,100]))) +
             alt.Chart(pd.DataFrame({'y':[70]})).mark_rule(color='red').encode(y='y') +
             alt.Chart(pd.DataFrame({'y':[30]})).mark_rule(color='blue').encode(y='y')).properties(height=150, title="RSI")
             
    return alt.vconcat(bb_c, rsi_c).resolve_scale(x='shared').interactive()

def visualize_return_analysis(df):
    df = df.copy()
    df["Cum_Ret"] = (1 + df["Close"].pct_change()).cumprod() - 1
    df_r = df.dropna().reset_index().rename(columns={"index":"Date"})
    return alt.Chart(df_r).mark_area(
        line={'color':'green'},
        color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='green', offset=1)], x1=1, x2=1, y1=1, y2=0)
    ).encode(
        x="Date:T", y=alt.Y("Cum_Ret:Q", axis=alt.Axis(format="%"), title="누적 수익률"),
        tooltip=["Date:T", alt.Tooltip("Cum_Ret:Q", format=".2%")]
    ).properties(height=300).interactive()

def visualize_phases_altair(df, pinpoints_df=None):
    if df.empty: return alt.Chart(pd.DataFrame()).mark_text()
    df_r = df.reset_index().rename(columns={"index":"Date"})
    
    bg = alt.Chart(pd.DataFrame()).mark_text()
    if "Phase" in df_r.columns:
        df_p = df_r.copy()
        df_p["gid"] = (df_p["Phase"] != df_p["Phase"].shift()).cumsum()
        blocks = df_p.groupby("gid").agg(s=("Date","min"), e=("Date","max"), p=("Phase","first")).reset_index()
        dom = ["상승","하락","박스권"]; rng = ["#ff9999","#aaccff","#d9d9d9"]
        bg = alt.Chart(blocks).mark_rect(opacity=0.4).encode(
            x="s:T", x2="e:T", color=alt.Color("p:N", scale=alt.Scale(domain=dom, range=rng))
        )

    line = alt.Chart(df_r).mark_line(color="gray").encode(x="Date:T", y=alt.Y("Close:Q", scale=alt.Scale(zero=False)))
    return (bg + line).properties(height=400).interactive()

# 챗봇 함수
def render_floating_chatbot():
    st.markdown("""
    <style>
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        width: 80px !important;  
        z-index: 999999 !important;
    }
    div[data-testid="stPopover"] > button {
        width: 100% !important;
        height: 100% !important;
        min-height: unset !important;
        min-width: unset !important;
        border-radius: 50% !important;
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background-color: #1d4ed8 !important;
        transform: scale(1.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.popover("💬"):
        st.markdown("### 🤖 투자 비서")
        st.caption("궁금한 점을 물어보세요.")
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]
        
        msgs = st.container(height=300)
        with msgs:
            for m in st.session_state.messages:
                st.chat_message(m["role"]).write(m["content"])
        
        if prompt := st.chat_input("질문 입력..."):
            st.session_state.messages.append({"role":"user", "content":prompt})
            msgs.chat_message("user").write(prompt)
            # 더미 응답
            ans = f"'{prompt}'에 대한 정보입니다. (AI 연결 필요)"
            st.session_state.messages.append({"role":"assistant", "content":ans})
            msgs.chat_message("assistant").write(ans)


# =========================
# 5. 뉴스 크롤러 (실시간 복구)
# =========================
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]
REFERER_POOL = ["https://news.naver.com/", "https://search.naver.com/"]
JITTER_RANGE = (0.2, 0.8)
CONNECT_TIMEOUT = 3
READ_TIMEOUT = 5
RESULTS_PER_PAGE = 10

def normalize_url(u: str) -> str:
    if not u: return ""
    u = u.strip()
    parts = urlsplit(u)
    scheme = parts.scheme or "https"
    netloc = parts.netloc
    path = parts.path
    if not netloc:
        if u.startswith(("news.naver.com", "n.news.naver.com")):
            pieces = u.split("/", 1)
            netloc = pieces[0]; path = "/" + pieces[1] if len(pieces) > 1 else "/"
        else: return re.sub(r"(\?.*|#.*)$", "", u)
    if netloc in ("news.naver.com", "m.news.naver.com"): netloc = "n.news.naver.com"
    return urlunsplit((scheme, netloc, path, "", ""))

def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter); s.mount("http://", adapter)
    s.headers.update({"User-Agent": random.choice(UA_POOL), "Referer": random.choice(REFERER_POOL)})
    return s

def get_with_backoff(session, url, **kwargs):
    time.sleep(random.uniform(*JITTER_RANGE))
    try:
        resp = session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if 200 <= resp.status_code < 300: return resp.text
    except: pass
    return None

def extract_news_content(url: str, session: requests.Session) -> Tuple[str, str, str, str]:
    html = get_with_backoff(session, normalize_url(url))
    if not html: raise Exception("No HTML")
    soup = BeautifulSoup(html, "html.parser")
    
    # 언론사
    company = "정보 없음"
    img = soup.select_one("a.media_end_head_top_logo img")
    if img: company = img.get("title") or img.get("alt") or company
    
    # 제목
    title = "정보 없음"
    for sel in ["h2#title_area", "div.media_end_head_title", "h1#newsct_article_title"]:
        t = soup.select_one(sel)
        if t: title = t.get_text(strip=True); break
        
    # 날짜
    date = "정보 없음"
    d = soup.select_one("span.media_end_head_info_datestamp_time")
    if d: date = d.get("data-date-time") or d.get_text(strip=True)
    
    return company, title, "", date

@st.cache_data(ttl=600)
def get_popular_news() -> List[Dict[str, str]]:
    query = "증시"
    today = datetime.today().strftime("%Y.%m.%d")
    session = make_session()
    
    # 검색 URL 빌드 (네이버 뉴스 검색)
    base = "https://search.naver.com/search.naver"
    params = {"where": "news", "query": query, "sm": "tab_opt", "sort": "1", "ds": today, "de": today}
    url = f"{base}?{urlencode(params)}"
    
    html = get_with_backoff(session, url)
    if not html: return []
    
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select("a.news_tit")
    links = []
    for a in anchors:
        href = a.get("href")
        if href and "news.naver.com" in href: links.append(href)
        if len(links) >= 10: break
    
    results = []
    for l in links:
        try:
            comp, tit, _, d = extract_news_content(l, session)
            results.append({"title": tit, "link": l, "source": comp, "date": d})
        except: continue
        if len(results) >= 6: break
        
    return results


# ------------------------------------------------------------------
# 5. 메인 화면 렌더링 (홈 / 분석)
# ------------------------------------------------------------------
def render_home():
    # 상단 여백
    st.markdown("<br>", unsafe_allow_html=True)

    # 중앙 정렬 (로고 및 검색창)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        # [수정됨] 로고 크기 조정 (width=300) 및 파일 확인
        logo_file = "image_3.png" 
        
        if os.path.exists(logo_file):
            # use_column_width=True 대신 width=300 사용 (화면 짤림 방지)
            st.image(logo_file, width=300) 
        else:
            st.markdown('<div class="main-logo-text">InvestWiki</div>', unsafe_allow_html=True)

        # 검색창
        st.markdown("<br>", unsafe_allow_html=True)
        search_val = st.text_input(
            "검색", placeholder="종목명 또는 티커 (예: 삼성전자, 005930)", 
            label_visibility="collapsed"
        )
        if search_val:
            st.session_state.selected_stock = search_val.split()[0]
            st.rerun()
            
        st.markdown(
            """<div style="text-align:center; color:#888; margin-top:10px; font-size:0.9rem;">
            🔍 인기 검색: 삼성전자, 테슬라, 비트코인, 엔비디아
            </div>""", unsafe_allow_html=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 하단 2단 레이아웃 (뉴스 | 인기종목)
    col_news, col_pop = st.columns([1.2, 1])
    
    with col_news:
        st.markdown("### 📰 실시간 증시 뉴스")
        # [수정됨] 실제 크롤링 함수 호출
        with st.spinner("최신 뉴스를 불러오는 중..."):
            news_data = get_popular_news()
            
        if not news_data:
            st.info("뉴스를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        else:
            for n in news_data:
                with st.expander(n['title']):
                    st.write(f"{n['source']} | {n['date']}")
                    st.markdown(f"[기사 원문 보기]({n['link']})")

    with col_pop:
        # [수정됨] 새로고침 버튼 및 인기종목 리스트
        pc1, pc2 = st.columns([3, 1])
        with pc1: st.markdown("### 🔥 인기 종목")
        with pc2: 
            if st.button("⟳", help="새로고침"):
                random.shuffle(st.session_state.popular_indices)
        
        # 상위 6개 표시
        for i in range(6):
            idx = st.session_state.popular_indices[i]
            name, code = ALL_POPULAR_STOCKS[idx]
            if st.button(f"📈 {name} ({code})", key=f"home_pop_{code}"):
                st.session_state.selected_stock = code
                st.rerun()

def render_analysis(page_id):
    # 현재 페이지 정보 찾기
    current_page = next((p for p in st.session_state.analysis_pages if p["id"] == page_id), None)
    
    if not current_page:
        st.error("페이지를 찾을 수 없습니다.")
        return

    # 종목 선택 (아직 선택 안 된 경우)
    if not current_page["ticker"]:
        st.title(f"📊 {current_page['title']}")
        ticker_input = st.text_input("분석할 종목 코드 입력 (예: 005930)", key=f"input_{page_id}")
        if st.button("분석 시작", key=f"btn_{page_id}"):
            stock_name = get_stock_name(ticker_input)
            current_page["ticker"] = ticker_input
            current_page["title"] = f"{stock_name}" # 제목 업데이트
            st.rerun()
        return

    # 분석 화면 렌더링
    ticker = current_page["ticker"]
    stock_name = get_stock_name(ticker)

    # --- [메인] 분석 결과 ---
    start_date = pd.to_datetime("2024-01-01")
    end_date = pd.to_datetime("2024-12-31")

    df = load_data(ticker, start_date, end_date)
    
    if df is None:
        st.error(f"'{ticker}'에 대한 데이터를 찾을 수 없습니다. 티커를 확인해주세요.")
        return

    st.title(f"{stock_name}")
    
    col_spacer, col_select = st.columns([4, 1])

    with col_select:
        # 1. 선택 가능한 메뉴 리스트 정의
        tab_options = ["📊 차트/시세", "🧠 AI 추세 분석", "📐 기술적 지표", "💰 수익률"]

        # 2. 오른쪽 작은 컬럼에 셀렉트박스를 배치합니다.
        selected_tab = st.selectbox(
            "분석 항목 선택",  # 라벨 (label_visibility로 숨길 예정이라 내용은 중요치 않음)
            tab_options, 
            index=0,
            label_visibility="collapsed" # 👈 'collapsed'로 설정하면 라벨(제목)이 숨겨져서 더 깔끔해집니다.
        )
    display_metrics(df)
    st.markdown("---") # 구분선 (선택사항)

    # 3. 선택된 값에 따라 다른 내용 렌더링
    if selected_tab == "📊 차트/시세":
        st.markdown("##### 일봉 캔들 차트")
        st.altair_chart(visualize_candlestick(df), use_container_width=True)
    elif selected_tab == "🧠 AI 추세 분석":
        with st.spinner("AI가 추세를 분석 중입니다..."):
            df_ai = detect_market_phases(df, 5, 3, 2, 2, 2, 9, 10)
        st.markdown("##### AI 추세 구간 탐지")
        st.altair_chart(visualize_phases_altair(df_ai), use_container_width=True)
        
        if "Phase" in df_ai.columns:
            c = df_ai["Phase"].value_counts()
            col1, col2, col3 = st.columns(3)
            col1.metric("상승 일수", f"{c.get('상승',0)}일")
            col2.metric("하락 일수", f"{c.get('하락',0)}일")
            col3.metric("박스권", f"{c.get('박스권',0)}일")

    elif selected_tab == "📐 기술적 지표":
        st.subheader("📐 기술적 지표 분석")
        
        # 1. 볼린저 밴드 설명 (정의 + 비유 + 툴팁)
        st.markdown("##### 1. 볼린저 밴드 (Bollinger Bands)", help="""
        **이동평균선을 기준으로 주가의 등락 범위를 표준편차로 계산해 표시한 지표입니다.**
        
        쉽게 말해, **주가가 평소에 다니는 '도로의 폭'**이라고 생각하면 됩니다.
        * **상단에 다다르면:** 주가가 단기적으로 너무 많이 올랐다는 신호입니다. (고평가/매도 고려)
        * **하단에 다다르면:** 주가가 단기적으로 너무 많이 떨어졌다는 신호입니다. (저평가/매수 고려)
        """)
        
        # 2. RSI 설명 (정의 + 비유 + 툴팁)
        st.markdown("##### 2. RSI (상대강도지수)", help="""
        **일정 기간 동안 주가가 전일 대비 얼마나 상승했는지를 백분율(%)로 나타낸 지표입니다.**
        
        쉽게 말해, **시장의 분위기가 얼마나 뜨거운지 보여주는 '온도계(0~100점)'**입니다.
        * **70점을 넘어서면:** 사는 사람이 너무 많아 '과열'된 상태입니다. (가격 하락 주의)
        * **30점 아래로 내려가면:** 파는 사람이 너무 많아 '침체'된 상태입니다. (반등 기회 가능)
        """)

        # 3. 차트 출력
        st.altair_chart(visualize_technical_indicators(df), use_container_width=True)

    elif selected_tab == "💰 수익률":
        st.markdown("##### 보유 기간 누적 수익률")
        st.altair_chart(visualize_return_analysis(df), use_container_width=True)

def render_sidebar():
    with st.sidebar: 
        # 1. 아이콘 URL 준비 (흰색)
        url_hamb = "https://img.icons8.com/ios-glyphs/60/ffffff/menu--v1.png"
        url_home = "https://img.icons8.com/ios-glyphs/60/ffffff/home.png"
        url_plus = "https://img.icons8.com/ios-glyphs/60/ffffff/plus-math.png"

        # 2. Base64 변환
        img_hamb = get_image_base64_from_url(url_hamb)
        img_home = get_image_base64_from_url(url_home)
        img_plus = get_image_base64_from_url(url_plus)
        
        images = [img for img in [img_hamb, img_home, img_plus] if img is not None]

        if images:
            # 3. 클릭 가능한 이미지 생성
            clicked = clickable_images(
                paths=images, 
                titles=["메뉴", "홈으로 가기", "새 분석 추가"],
                div_style={
                    "display": "flex", 
                    "flex-direction": "column", 
                    "align-items": "center", 
                    "justify-content": "start", 
                    "gap": "15px",
                    "background-color": "#2B2D3E", # 사이드바 배경색과 일치
                    "padding": "10px"
                }, 
                img_style={
                    "margin": "5px", 
                    "height": "30px", 
                    "cursor": "pointer"
                }, 
                key=str(st.session_state.menu_key) 
            )

            # 4. 클릭 이벤트 처리
            if clicked > -1:
                st.session_state.menu_key += 1 # 컴포넌트 리셋
                
                if clicked == 1: # 홈
                    st.session_state.current_page_id = "HOME"
                    st.rerun()
                    
                elif clicked == 2: # 추가
                    new_id = str(uuid.uuid4())
                    new_title = f"분석 리포트 {len(st.session_state.analysis_pages) + 1}"
                    
                    st.session_state.analysis_pages.append({
                        "id": new_id,
                        "title": new_title,
                        "ticker": None # 아직 종목 선택 안됨
                    })
                    
                    st.session_state.current_page_id = new_id
                    st.rerun()

        st.divider()

        # 5. 생성된 리포트 목록 표시
        st.caption("📑 생성된 리포트 목록")
        
        if not st.session_state.analysis_pages:
            st.info("생성된 분석 페이지가 없습니다.")
        
        for page in st.session_state.analysis_pages:
            # 현재 선택된 페이지 강조
            btn_type = "primary" if st.session_state.current_page_id == page["id"] else "secondary"
            
            # 버튼 클릭 시 해당 페이지로 이동
            if st.button(page["title"], key=page["id"], type=btn_type, use_container_width=True):
                st.session_state.current_page_id = page["id"]
                st.rerun()
        
        # 6. 초기화 버튼
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("모든 페이지 초기화", type="secondary", use_container_width=True):
            st.session_state.analysis_pages = []
            st.session_state.current_page_id = "HOME"
            st.session_state.menu_key += 1
            st.rerun()


# =========================
# 6. 메인 실행 루프
# =========================

# 세션 초기화
if "analysis_pages" not in st.session_state:
    st.session_state.analysis_pages = []
if "current_page_id" not in st.session_state:
    st.session_state.current_page_id = "HOME"
if "menu_key" not in st.session_state:
    st.session_state.menu_key = 0

# 1. 사이드바 렌더링 (항상 표시)
render_sidebar()

# 2. 메인 콘텐츠 라우팅
if st.session_state.current_page_id == "HOME":
    render_home()
else:
    render_analysis(st.session_state.current_page_id)

# 3. 챗봇 (항상 표시)
# (render_floating_chatbot 함수는 위 코드에 포함되어 있다고 가정)
# (app1.py 맨 마지막 줄에 추가)
render_floating_chatbot()