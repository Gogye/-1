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
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import textwrap

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

st.markdown("""
    <style>
    /* 1. Streamlit 기본 헤더 투명화 (버튼 보이게) */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 999999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* ==========================================================================
       1. 전체 페이지 레이아웃 & 테마
       ========================================================================== */
    .stApp { background-color: #eef6f6 !important; }
    
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
    .main-header-text {
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #004aad, #cb6ce6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        margin-top: 1rem;
    }
    .analysis-header-text {
        font-size: 2.5rem;
        font-weight: 800;
        width: 100%;
        display: block;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #004aad, #cb6ce6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
        margin-top: 0.8rem;
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

st.markdown("""
<style>
/* 카드 컨테이너 스타일 */
.dashboard-card {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    height: 100%;
    border: 1px solid #f0f0f0;
}

/* 카드 헤더 (제목 + 아이콘) */
.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 15px;
    background-color: #ffffff;
}

.card-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #2B3674;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* 아이콘 박스 (원형 배경) */
.icon-box {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}
.icon-news { background-color: #E6E6FA; color: #4318FF; } /* 연보라 */
.icon-fire { background-color: #FFF5E6; color: #FF8C00; } /* 연주황 */

/* 뉴스 리스트 스타일 */
.news-item {
    padding: 10px 0;
    border-bottom: 1px solid #f5f5f5;
}
.news-item:last-child { border-bottom: none; }
.news-title { font-weight: 600; color: #333; text-decoration: none; display: block; margin-bottom: 4px;}
.news-title:hover { color: #4318FF; text-decoration: underline; }
.news-meta { font-size: 0.8rem; color: #999; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 뉴스 컨테이너 (Key: pop_card_container1) 스타일 */
.st-key-pop_card_container1 {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    border: 1px solid #f0f0f0 !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
}

/* 내부 요소 투명화 (배경색 안 가리게) */
.st-key-pop_card_container1 > div {
    background-color: transparent !important;
}
            
/* 인기 종목 컨테이너 (Key: pop_card_container2) 스타일 */
.st-key-pop_card_container2 {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    border: 1px solid #f0f0f0 !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
}

/* 내부 요소 투명화 (배경색 안 가리게) */
.st-key-pop_card_container2 > div {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 뉴스 컨테이너 (Key: analysis_container1) 스타일 */
.st-key-analysis_container1 {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    border: 1px solid #f0f0f0 !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
}
.st-key-analysis_container1 > div {
    background-color: transparent !important;
}
            
/* 뉴스 컨테이너 (Key: analysis_container2) 스타일 */
.st-key-analysis_container2 {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    border: 1px solid #f0f0f0 !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
}
.st-key-analysis_container2 > div {
    background-color: transparent !important;
}
            
/* 뉴스 컨테이너 (Key: analysis_container3_1) 스타일 */
.st-key-analysis_container3_1 {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    border: 1px solid #f0f0f0 !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
}
.st-key-analysis_container3_1 > div {
    background-color: transparent !important;
}
            
/* 뉴스 컨테이너 (Key: analysis_container3_2) 스타일 */
.st-key-analysis_container3_2 {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    border: 1px solid #f0f0f0 !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
}
.st-key-analysis_container3_2 > div {
    background-color: transparent !important;
}
            
/* 뉴스 컨테이너 (Key: analysis_container4) 스타일 */
.st-key-analysis_container4 {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    border: 1px solid #f0f0f0 !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    padding: 20px !important;
}
.st-key-analysis_container4 > div {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

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

# =========================
# 2. 헬퍼 함수 (이미지 로드, 데이터 로드)
# =========================
@st.cache_data
def get_image_base64_from_url(url):
    # 브라우저인 척 위장하는 헤더 (차단 방지)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        # timeout을 설정하여 무한 대기 방지
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            encoded_string = base64.b64encode(response.content).decode()
            return f"data:image/png;base64,{encoded_string}"
    except:
        pass
    return None

@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        df = fdr.DataReader(ticker, start_date, end_date)
        df = df.dropna()
        if df.empty: return None
        return df.copy()
    except: return None

@st.cache_data
def load_news(url):
    try:
        df = pd.read_csv(url)
    except:
        df = pd.read_csv(url, encoding='cp949')
    return df

@st.cache_data
def news_work(df_ai, ticker):
    # data_ranges 만드는 작업
    df = df_ai.copy()
    df['phase_change'] = df['Phase'] != df['Phase'].shift(1)
    df['new_block_id'] = df['phase_change'].cumsum()
    df.index = pd.to_datetime(df.index)

    ranges_df = df.reset_index().groupby('new_block_id')['Date'].agg(['min', 'max']).sort_values('min')
    date_ranges = [
        (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')) 
        for start, end in zip(ranges_df['min'], ranges_df['max'])
    ]

    url = None
    if ticker == "005930":
        condensed = []
        for i in range(21):
            url = f"https://raw.githubusercontent.com/minjun069/DartB/main/Samsung_condensed_{i}.csv"
            cond_df = load_news(url)
            condensed.append(cond_df)
        return [condensed, date_ranges]
    elif ticker == "011200":
        url = "https://raw.githubusercontent.com/minjun069/DartB/main/HMM_all.csv"
    elif ticker == "068270":
        url = "https://raw.githubusercontent.com/minjun069/DartB/main/Celltrion_all.csv"
    
    if url:
        full_news = load_news(url)
        condensed = clustering_news(full_news, date_ranges) #리스트임
        return [condensed, date_ranges]
    else:
        return

@st.cache_data
def news_work2(condensed, news_idx):
    news = condensed[news_idx].copy()
    news['date'] = pd.to_datetime(news['date'], errors='coerce')
    news['date'] = news['date'].dt.date
    news = news[news['cluster']!=-1][['date','title','cluster_count','link']]
    news.rename(columns={'date':"날짜","title":"기사제목","cluster_count":"중복횟수","link":"링크"}, inplace=True)
    news.set_index('날짜', inplace=True)
    news = news.head(10).sort_values(by='중복횟수', ascending=False)
    return news

@st.cache_data
def total_news_work(ticker):
    url = None
    if ticker == "011200":
        url = "https://raw.githubusercontent.com/minjun069/DartB/main/HMM_total_news.csv"
    elif ticker == "068270":
        url = "https://raw.githubusercontent.com/minjun069/DartB/main/celltrion_total_news.csv"
    elif ticker == "005930":
        url = "https://raw.githubusercontent.com/minjun069/DartB/main/samsung_total_news.csv"

    if url:
        full_news = load_news(url)
        full_news = full_news[['날짜','기사 제목','중복횟수', '링크']]
        #full_news.set_index("날짜", inplace=True)
        return full_news
    else:
        return

@st.cache_data
def get_info(search_val):
    target = search_val.split()[0].upper().strip()
    # 기존 데이터 베이스 검색
    for name, code in ALL_POPULAR_STOCKS:
        if target == code or target == name:
            return code, name
    return "error", 'error'

def card_html(title, value, icon, color):
    # [핵심 변경] 
    # 1. 전체 컨테이너: flex-direction: column (위아래 배치)
    # 2. 상단 래퍼: display: flex (아이콘과 제목 가로 배치)
    
    return f"""
<div style="
    background-color: white; 
    border-radius: 15px; 
    padding: 20px 25px; 
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); 
    border: 1px solid #f0f0f0; 
    border-left: 7px solid {color}; 
    height: 100%; 
    margin-bottom: 10px;
    display: flex; 
    flex-direction: column; 
    justify-content: center;
">
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <div style="
            width: 35px; 
            height: 35px; 
            border-radius: 50%; 
            background-color: #F4F7FE; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-size: 1.1rem;
            margin-right: 10px; /* 제목과의 간격 */
        ">
            {icon}
        </div>
        <div style="color: #A3AED0; font-size: 0.9rem; font-weight: 600;">
            {title}
        </div>
    </div>
    <div style="color: #2B3674; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.5px;">
        {value}
    </div>
</div>
"""

def get_phase_bar_html(up, down, box):
    return f"""
    <div style="
        display: flex; 
        width: 100%; 
        height: 65px; 
        border-radius: 12px; 
        overflow: hidden; 
        font-family: 'Source Sans Pro', sans-serif; 
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    ">
        <div style="width: 33%; background-color: #C24E42; display: flex; flex-direction: column; align-items: center; justify-content: center; color: white;">
            <div style="font-weight: 700; font-size: 15px; margin-bottom: 2px;">↗ 상승</div>
            <div style="font-size: 12px; opacity: 0.9;">{up}일</div>
        </div>
        <div style="width: 33%; background-color: #5D7AE2; display: flex; flex-direction: column; align-items: center; justify-content: center; color: white;">
            <div style="font-weight: 700; font-size: 15px; margin-bottom: 2px;">↓ 하락</div>
            <div style="font-size: 12px; opacity: 0.9;">{down}일</div>
        </div>
        <div style="width: 33%; background-color: #999999; display: flex; flex-direction: column; align-items: center; justify-content: center; color: white;">
            <div style="font-weight: 700; font-size: 15px; margin-bottom: 2px;">⇄ 박스권</div>
            <div style="font-size: 12px; opacity: 0.9;">{box}일</div>
        </div>
    </div>
    """

def searching_func(search_val, page_id):
    ticker, stock_name = get_info(search_val)

    if ticker == 'error':
        st.toast(f"❌ '{search_val}'에 대한 검색 결과가 없습니다.", icon="⚠️")
        return

    # 이미 존재하는 분석 페이지인지 확인
    found_page_id = None
    for page in st.session_state.analysis_pages:
        if page.get("ticker") == ticker:
            found_page_id = page["id"]
            break
    
    # 이미 존재한다면
    if found_page_id:
        if page_id != "HOME":
            current_page_obj = next((p for p in st.session_state.analysis_pages if p["id"] == page_id), None)
            if current_page_obj in st.session_state.analysis_pages:
                st.session_state.analysis_pages.remove(current_page_obj)
        st.session_state.current_page_id = found_page_id

    # 새로운 종목이라면
    else:
        if page_id == 'HOME':
            new_id = str(uuid.uuid4())
            new_page = {
                "id": new_id,
                "title": f"{stock_name}",
                "ticker": ticker,
                "data": None,
                'stock_name':stock_name
            }
            st.session_state.analysis_pages.append(new_page)
            st.session_state.current_page_id = new_id

        else:
            current_page_obj = next((p for p in st.session_state.analysis_pages if p["id"] == page_id), None)
            if current_page_obj:
                current_page_obj["ticker"] = ticker
                current_page_obj["title"] = f"{stock_name}"
                current_page_obj['stock_name'] = stock_name
    st.rerun()

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

@st.cache_data
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
def visualize_candlestick(df):
    df_r = df.reset_index().rename(columns={"index":"Date"})
    base = alt.Chart(df_r).encode(x=alt.X("Date:T", title=None, axis=alt.Axis(format="%Y-%m-%d")))
    rule = base.mark_rule().encode(
        y=alt.Y("Low:Q", scale=alt.Scale(zero=False), title=None), y2="High:Q",
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff"))
    )
    bar = base.mark_bar().encode(
        y="Open:Q", y2="Close:Q",
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff")),
        tooltip=["Date:T", "Open", "Close", "High", "Low"]
    )
    return (rule + bar).properties(height=350).interactive()

def visualize_technical_indicators1(df):
    df = df.copy()
    if len(df) < 30: return alt.Chart(pd.DataFrame()).mark_text(text="데이터 부족")
    
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_h"] = bb.bollinger_hband(); df["bb_l"] = bb.bollinger_lband()
    
    df_r = df.reset_index().rename(columns={"index":"Date"})
    base = alt.Chart(df_r).encode(x=alt.X("Date:T", axis=alt.Axis(format="%Y-%m-%d"), title=None))
    
    bb_c = (base.mark_line(color="black").encode(y=alt.Y("Close:Q", scale=alt.Scale(zero=False), title=None)) + 
            base.mark_area(opacity=0.2).encode(y="bb_l:Q", y2="bb_h:Q")).properties(height=350)
    
    return alt.vconcat(bb_c).resolve_scale(x='shared').interactive()

def visualize_technical_indicators2(df):
    df = df.copy()
    if len(df) < 30: return alt.Chart(pd.DataFrame()).mark_text(text="데이터 부족")
    
    rsi = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
    df["rsi"] = rsi
    
    df_r = df.reset_index().rename(columns={"index":"Date"})
    base = alt.Chart(df_r).encode(x=alt.X("Date:T", axis=alt.Axis(format="%Y-%m-%d"), title=None))

    rsi_c = (base.mark_line(color="black").encode(y=alt.Y("rsi:Q", scale=alt.Scale(domain=[0,100]), title=None)) +
             alt.Chart(pd.DataFrame({'y':[70]})).mark_rule(color='red').encode(y='y') +
             alt.Chart(pd.DataFrame({'y':[30]})).mark_rule(color='blue').encode(y='y')).properties(height=350)
             
    return alt.vconcat(rsi_c).resolve_scale(x='shared').interactive()

def visualize_return_analysis(df):
    df = df.copy()
    df["Cum_Ret"] = (1 + df["Close"].pct_change()).cumprod() - 1
    df_r = df.dropna().reset_index().rename(columns={"index":"Date"})
    return alt.Chart(df_r).mark_area(
        line={'color':'green'},
        color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='green', offset=1)], x1=1, x2=1, y1=1, y2=0)
    ).encode(
        x=alt.X("Date:T", axis=alt.Axis(format="%Y-%m-%d"), title=None), y=alt.Y("Cum_Ret:Q", axis=alt.Axis(format="%"), title=None),
        tooltip=["Date:T", alt.Tooltip("Cum_Ret:Q", format=".2%")]
    ).properties(height=350).interactive()

@st.cache_data
def visualize_phases_altair_all_interactions(df, pinpoints_df):
    """
    Altair의 4가지 주요 상호작용을 모두 포함하는 차트를 생성합니다.
    1. 툴팁 (Tooltip)
    2. 하이라이트 (Highlight on Mouseover)
    3. 선택 (Selection on Click)
    4. 브러시 & 필터 (Interval Brush & Cross-filtering)
    """
    
    # --- 1. 데이터 준비 ---
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_text().properties(
            title="데이터가 없습니다."
        )
    df_reset = df.reset_index().rename(columns={'index': 'Date'})

    # ❗️ [추가] Y축 하위 5% 위치의 '가격' 값을 계산합니다.
    min_price = df_reset['Close'].min()
    max_price = df_reset['Close'].max()
    price_range = max_price - min_price
    
    # Y축 하위 5%에 해당하는 실제 가격 값
    target_y_value = min_price + (price_range * 0.001)
    
    # --- 2. (배경) Phase 블록 계산 (이전과 동일) ---
    background = alt.Chart(pd.DataFrame()).mark_text()
    phase_blocks_empty = True 

    if "Phase" in df_reset.columns and not df_reset['Phase'].isnull().all():
        df_phases = df_reset[['Date', 'Phase']].copy()
        df_phases['Phase'] = df_phases['Phase'].fillna('N/A')
        df_phases['New_Block'] = df_phases['Phase'] != df_phases['Phase'].shift(1)
        df_phases['Block_ID'] = df_phases['New_Block'].cumsum()
        
        phase_blocks = df_phases.groupby('Block_ID').agg(
            start_date=('Date', 'min'), end_date=('Date', 'max'), Phase=('Phase', 'first')
        ).reset_index()
        phase_blocks = phase_blocks[phase_blocks['Phase'] != 'N/A']
    
        if not phase_blocks.empty:
            # 1. 색상 매핑 정의 (이게 없어서 색이 맘대로 나옴)
            domain = ['하락', '상승', '박스권'] 
            range_ = ["#f77777", "#84b4fd", '#ffffff'] # 빨강, 파랑, 회색

            phase_blocks_empty = False
            background = alt.Chart(phase_blocks).mark_rect(opacity=0.15).encode(
                x=alt.X('start_date:T', title='날짜'), x2=alt.X2('end_date:T'),
                color=alt.Color(
                    'Phase:N', 
                    scale=alt.Scale(domain=domain, range=range_),
                    legend=alt.Legend(title='추세', orient='top')
                    ),
                tooltip=['start_date:T', 'end_date:T', 'Phase:N']
            )

    # --- 3. (전경) 선 그래프 (이전과 동일) ---
    line_chart = alt.Chart(df_reset).mark_line(color='gray').encode(
        x=alt.X('Date:T', title=None),
        y=alt.Y('Close:Q', title=None, scale=alt.Scale(zero=False)),
        tooltip=['Date:T', 'Close:Q']
    )
    # --- 4. (중요) 상호작용 셀렉터(Selector) 정의 ---
    
    # 핀포인트 위 '마우스 오버' 감지 (하이라이트용)
    hover_selection = alt.selection_point(
        on='mouseover', empty='all', fields=['Date']
    )

    # --- 5. (옵션) 핀포인트 레이어 생성 (모든 상호작용 적용) ---
    pinpoint_layer = alt.Chart(pd.DataFrame()).mark_text()

    if pinpoints_df is not None and not pinpoints_df.empty:
        # (데이터 병합 로직은 이전과 동일)
        
        pinpoints_df_copy = pinpoints_df.copy()
        pinpoints_df_copy['Date'] = pd.to_datetime(pinpoints_df_copy['날짜'])
        merged_pins = pd.merge(
            df_reset[['Date', 'Close']], pinpoints_df_copy, on='Date', how='inner'
        )

        if not merged_pins.empty:
            # 수직선
            rule = alt.Chart(merged_pins).mark_rule(
                color='black', strokeDash=[3, 3]
            ).encode(x='Date:T')

            # 핀포인트 (점) - 모든 상호작용이 여기에 적용됨
            points = alt.Chart(merged_pins).mark_point(
                filled=True,
                stroke='black',
                strokeWidth=0.5,
                color='yellow'  # 👈 [추가] 모든 점을 빨간색으로 고정
            ).transform_calculate(
                pin_y_position=f"{target_y_value}"  # 계산된 Y 위치 사용
            ).encode(
                x='Date:T',
                y=alt.Y('pin_y_position:Q', title='가격'),
                # 1. 툴팁 (Tooltip): 마우스 오버 시 정보 표시
                tooltip=[
                    alt.Tooltip('Date:T', title='날짜', format='%Y-%m-%d'),
                    alt.Tooltip('기사 제목:N', title='이벤트')
                    #,
                    #alt.Tooltip('Close:Q', title='종가', format=',.2f')
                ],
                # 2. 하이라이트 (Highlight): 마우스 오버 시 크기 변경
                size=alt.condition(hover_selection, 
                                 alt.value(200),alt.value(100)  # 마우스 올리면 200, 평상시 100
                )
            ).add_params(hover_selection)
            
            pinpoint_layer = rule + points

    # --- 6. [위] 메인 차트 조립 ---
    if phase_blocks_empty:
        base_chart = line_chart
    else:
        base_chart = background + line_chart
    target_y_df = pd.DataFrame({'target_y': [target_y_value]})
    base_line = alt.Chart(target_y_df).mark_rule(
        color='black', opacity=0
    ).encode(y='target_y:Q')
    main_chart = (base_chart + pinpoint_layer + base_line).properties(
        height=500
    )
    
    return main_chart

# =========================
# 5. 뉴스 관련
# =========================
def get_popular_news() -> List[Dict[str, str]]:
    url = "https://search.naver.com/search.naver?where=news&query=증시"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    titles = soup.select(".sds-comps-text-type-headline1")

    news_items = []
    for title_tag in titles:
        try:
            title_text = title_tag.get_text()
            link = title_tag.find_parent("a")["href"]

            desc_tag = title_tag.find_parent("div").select_one(".sds-comps-text-type-body1")
            desc_text = desc_tag.get_text() if desc_tag else "내용 없음"

            news_items.append({
                "title": title_text,
                "link": link,
                "desc": desc_text
            })
            
        except Exception as e:
            continue
    
    return news_items

def normalize_text(s: str) -> str:
    """보일러플레이트/괄호태그/특수기호/공백 정리"""
    if not s or s == "정보 없음":
        return ""
    s = re.sub(r"\[[^\]]*\]", " ", s)            # [단독], [속보]
    s = re.sub(r"\([^)]*\)", " ", s)             # (종합), (영상)
    s = re.sub(r"[가-힣\w\.-]+ 기자", " ", s)      # 기자명
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", s)  # 이메일
    special_chars = "▶■◆●◇★☆▲▼▷▶️□○※…·•�'\""
    pattern = "[" + re.escape(special_chars) + "]"
    s = re.sub(pattern, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def condense_news(df_news): 
    # 전역 벡터라이저 (문자 n-그램: 한국어에 강함)
    VEC = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2
    )
    # [핵심] 유사도(Similarity)를 거리(Distance)로 변환합니다. 
    # (유사도 1.0 = 거리 0.0) # (유사도 0.0 = 거리 1.0) 
    news = df_news.copy()
    news['compare_text'] = news.apply(lambda row: normalize_text(row['title']), axis=1) 
    texts = list(news['compare_text'])
    X = VEC.fit_transform(texts) 
    sims = cosine_similarity(X, X)

    distances = 1 - sims 
    distances[distances < 0] = 0 

    # 1. metric='precomputed': 우리가 이미 거리 행렬을 계산했음을 알려줍니다. 
    # # 2. eps: "같은 클러스터"로 인정할 최대 거리. 
    # # 3. min_samples: 하나의 클러스터(중복 묶음)를 이루는 최소 기사 수. (원본 + 중복 1 = 2개) 
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='precomputed') 
    # 클러스터링 실행 
    labels = clustering.fit_predict(distances) 
    df_news['cluster'] = labels 
    df_news.sort_values(by=['cluster'], inplace=True) 
    df_news['date'] = pd.to_datetime(df_news['date']) 
    day = df_news['date'].values[0]
    cluster_counts = df_news.groupby('cluster').size().rename('cluster_count') 
    df_condensed = ( df_news.sort_values('date') # 날짜 오름차순 정렬 
                     .groupby('cluster') # 클러스터별로 묶기 
                     .first() # 첫 번째(가장 오래된 뉴스) 
                     .reset_index() ) # 클러스터 개수 병합 
    df_condensed = df_condensed.merge(cluster_counts, on='cluster', how='left') 
    return df_condensed

def clustering_news(full_news, date_ranges):
    condensed = []
    full_news['date'] = pd.to_datetime(full_news['date'], format='mixed', errors='coerce')
    for start_date, end_date in date_ranges:
        mask = (full_news['date'] >= start_date) & (full_news['date'] <= end_date)
        period_news = full_news.loc[mask].copy() # .copy()로 경고 방지
        
        if not period_news.empty:
            df_condensed = condense_news(period_news)
            condensed.append(df_condensed)
    return condensed

# ------------------------------------------------------------------
# 5. 메인 화면 렌더링 (홈 / 분석)
# ------------------------------------------------------------------
def render_home():

    st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem !important; 
    }
    </style>
    """, unsafe_allow_html=True)

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
        search_val = st.text_input(
            "검색", placeholder="종목명 또는 티커 (예: 삼성전자, 005930)", 
            label_visibility="collapsed"
        )
        if search_val:
            searching_func(search_val, 'HOME')

        st.markdown(
            """<div style="text-align:center; color:#888; margin-top:5px; font-size:0.75rem;">
            🔍 인기 검색: 삼성전자, 테슬라, 비트코인, 엔비디아
            </div>""", unsafe_allow_html=True
        )
        st.markdown("<br><br>", unsafe_allow_html=True)

    # 하단 2단 레이아웃 (뉴스 | 인기종목)
    col_news, col_pop = st.columns([1.2, 1])

    with col_news:
        # [구조 변경] HTML 조립 대신 st.container 사용
        with st.container(border=True, key="pop_card_container1"):
            st.markdown("""
            <div class="card-title" style="margin-bottom:0;">
                <span class="icon-box icon-news">📰</span> 실시간 증시 뉴스
            </div>
            """, unsafe_allow_html=True)
            
            news_data = get_popular_news()
            
            # 내용물 출력
            if not news_data:
                st.markdown("""
                <div style="text-align: center; padding: 40px 0; color: #999;">
                    <div style="font-size: 2rem; margin-bottom: 10px;">📭</div>
                    <div>뉴스를 불러오지 못했습니다.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # 리스트 아이템 반복 출력
                # st.container 안에서는 st.markdown을 반복해서 써도 레이아웃이 안 깨집니다.
                for n in news_data[:3]:
                    st.markdown(f"""
                    <div class="news-item" style="padding:10px 0; border-bottom:1px solid #f9f9f9;">
                        <a href="{n['link']}" target="_blank" class="news-title" style="text-decoration:none; color:#333; font-weight:600; display:block; margin-bottom:4px;">
                            {n['title']}
                        </a>
                        <div class="news-meta" style="font-size:0.8rem; color:#999;">
                            {n['desc']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with col_pop:
        # 2. 컨테이너 시작
        with st.container(border=True, key="pop_card_container2"):
            
            # 헤더 영역 (제목 + 새로고침)
            h1, h2 = st.columns([4, 1])
            with h1:
                st.markdown("""
                <div class="card-title" style="margin-bottom:0;">
                    <span class="icon-box icon-fire">🔥</span> 인기 종목
                </div>
                """, unsafe_allow_html=True)
            with h2:
                if st.button("⟳", help="목록 새로고침", use_container_width=True):
                    random.shuffle(st.session_state.popular_indices)
                    st.rerun()

            # 종목 리스트 (2열 그리드) - 기존 로직 유지
            c1, c2 = st.columns(2)
            
            with c1:
                for i in range(3):
                    idx = st.session_state.popular_indices[i]
                    name, code = ALL_POPULAR_STOCKS[idx]
                    if st.button(f"{name}", key=f"pop_L_{code}", use_container_width=True):
                        searching_func(code, "HOME")
            with c2:
                for i in range(3, 6):
                    idx = st.session_state.popular_indices[i]
                    name, code = ALL_POPULAR_STOCKS[idx]
                    if st.button(f"{name}", key=f"pop_R_{code}", use_container_width=True):
                        searching_func(code, "HOME")

def render_analysis(page_id):
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    st.markdown("""
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #ffffff;">
    <div class="analysis-header-text">InvestWiki</div>
    </nav>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem !important; 
    }
    </style>
    """, unsafe_allow_html=True)

    col_text, col_empty, col_date1, col_date2, col_reload = st.columns([1, 3, 1, 1, 0.5])

    with col_text:
        st.text("HOME  >  DASHBOARD")

    with col_date1:
        # 시작일 선택 (기본값: 2024-01-01)
        # key를 unique하게 설정해야 다른 페이지와 충돌하지 않음
        start_date = st.date_input(
            "시작일", 
            value=pd.to_datetime("2024-01-01"),
            max_value=datetime.today(),
            key=f"start_date_{page_id}"
        )
        st.session_state.analysis_dates[0] = start_date
            
    with col_date2:
        # 종료일 선택 (기본값: 오늘)
        end_date = st.date_input(
            "종료일", 
            value=pd.to_datetime("2024-12-31"), # 또는 datetime.today()
            max_value=datetime.today(),
            min_value=start_date, # 시작일보다 앞설 수 없음
            key=f"end_date_{page_id}"
        )
        st.session_state.analysis_dates[1] = end_date

    with col_reload:
        # [디자인 팁] 옆의 날짜 입력창 라벨 높이만큼 빈 공간을 줘서 줄을 맞춥니다.
        st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        # 새로고침 버튼
        if st.button("조회", type="primary", use_container_width=True):
            # (선택사항) 만약 캐시된 데이터를 무시하고 새로 가져오고 싶다면:
            # load_data.clear() 
            st.rerun()
    
    #st.markdown("<br>", unsafe_allow_html=True)

    # 현재 페이지 정보 찾기
    current_page = next((p for p in st.session_state.analysis_pages if p["id"] == page_id), None)
    
    if not current_page:
        st.error("페이지를 찾을 수 없습니다.")
        return

    # 종목 선택 (아직 선택 안 된 경우)
    if not current_page["ticker"]:
        st.title(f"기업 분석")
        search_val = st.text_input("리포트 검색", 
                                     placeholder="분석할 기업이름 또는 종목코드 입력 (예: 삼성전자 또는 005930)", 
                                     key=f"input_{page_id}",
                                     label_visibility="collapsed")
        if search_val:
            searching_func(search_val, page_id)
        return

    # 분석 화면 렌더링
    ticker = current_page["ticker"]
    stock_name = current_page['stock_name']

    start_date = pd.to_datetime(st.session_state.analysis_dates[0])
    end_date = pd.to_datetime(st.session_state.analysis_dates[1])

    df = load_data(ticker, start_date, end_date)
    if df is None:
        st.error(f"'{ticker}'에 대한 데이터를 찾을 수 없습니다. 티커를 확인해주세요.")
        return

    st.title(f"{stock_name}")

    first = df.iloc[0]['Close']
    latest = df.iloc[-1]['Close']
    highest = df['Close'].max()
    lowest = df['Close'].min()
    return_rate = (latest-first)/first*100

    col_1, col_2, col_3, col_4 = st.columns([1, 1, 1, 1])

    with col_1: st.markdown(card_html("기말 주가", f"{latest:,.0f}원", "🏆", "yellow"), unsafe_allow_html=True)
    with col_2: st.markdown(card_html("수익률", f"{return_rate:+.2f}%", "💰", "green"), unsafe_allow_html=True)
    with col_3: st.markdown(card_html("최저가", f"{lowest:,.0f}원", "📉", "red"), unsafe_allow_html=True)
    with col_4: st.markdown(card_html("최고가", f"{highest:,.0f}원", "📈", "blue"), unsafe_allow_html=True)

    st.markdown("---") # 구분선 (선택사항)
    st.markdown("""
    <style>
    /* 1. 탭 컨테이너 (전체 틀) - 간격 넓게 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #e0e0e0;
    }

    /* 2. 개별 탭 버튼 (껍데기) */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        width: 100%; /* 너비 꽉 채우기 */
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        border: none;
        padding: 0 20px;
    }

    /* 3. [핵심] 탭 내부의 '글자'를 직접 타겟팅하여 폰트 변경 */
    .stTabs [data-baseweb="tab"] p {
        font-size: 18px !important;  /* 글자 크기 */
        font-weight: 700 !important; /* 글자 굵기 (Bold) */
        color: #6b7280 !important;   /* 기본 색상 (회색) */
        font-family: "Source Sans Pro", sans-serif !important; /* 폰트체 */
    }

    /* 4. 선택된 탭 스타일 (활성화 상태) */
    .stTabs [aria-selected="true"] {
        background-color: #F4F7FE !important;
        border-bottom: 3px solid #4318FF !important;
    }
    
    /* 5. [핵심] 선택된 탭의 '글자' 색상 변경 */
    .stTabs [aria-selected="true"] p {
        color: #4318FF !important; /* 선택된 글자색 (진한 파랑) */
    }

    /* 6. 기본 빨간 밑줄 제거 */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["캔들 차트", "추세 구간화", "기술적 지표", "누적 수익률"])
    # 3. 선택된 값에 따라 다른 내용 렌더링
    with tab1:
        with st.container(border=True, key="analysis_container1"):
            st.markdown("##### 일봉 캔들 차트")
            st.markdown("<br>", unsafe_allow_html=True)
            st.altair_chart(visualize_candlestick(df), use_container_width=True)

    with tab2:
        with st.container(border=True, key="analysis_container2"):
            with st.spinner("AI가 추세를 분석 중입니다..."):
                df_ai = detect_market_phases(df, 5, 3, 2, 2, 2, 9, 10)
            st.markdown("##### 추세 구간화 및 주요 뉴스")

            c = df_ai["Phase"].value_counts()
            up = c.get('상승',0)
            down = c.get('하락',0)
            box = c.get('박스권',0)
            bar_html = get_phase_bar_html(up, down, box)
            st.markdown(bar_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            total_news = total_news_work(ticker)
            st.altair_chart(visualize_phases_altair_all_interactions(df_ai, total_news), use_container_width=True)

            if total_news is not None and not total_news.empty:
                total_news.set_index("날짜", inplace=True)
                st.dataframe(total_news,                         
                             column_config={
                            "링크": st.column_config.LinkColumn(
                                "기사 보기",   # 컬럼 헤더 이름
                                display_text="원문 이동" # 셀 안에 표시될 텍스트 (URL 대신 이 글자가 뜸)
                            )
                        },
                        use_container_width=True)

            st.markdown("---")

            with st.expander("구간 별 뉴스"):
                news_work_li = news_work(df_ai, ticker)
                if news_work_li:
                    condensed = news_work_li[0]
                    date_ranges = news_work_li[1]
                    news_idx = st.selectbox(
                                "📅 뉴스 구간 선택",
                                options=range(len(condensed)), # [0, 1, 2...]
                                format_func=lambda i: f"{date_ranges[i][0]} ~ {date_ranges[i][1]}" # 화면엔 날짜로 표시
                            )
                    news = news_work2(condensed, news_idx)
                    st.dataframe(
                        news,
                        column_config={
                            "링크": st.column_config.LinkColumn(
                                "기사 보기",   # 컬럼 헤더 이름
                                display_text="원문 이동" # 셀 안에 표시될 텍스트 (URL 대신 이 글자가 뜸)
                            )
                        },
                        use_container_width=True
                    )

    with tab3:
        with st.container(border=True, key="analysis_container3_1"):
            # 1. 볼린저 밴드
            st.markdown("##### 1. 볼린저 밴드 (Bollinger Bands)")
            with st.expander("📖 볼린저 밴드가 뭔가요?"):
                st.info("""
                **이동평균선을 기준으로 주가의 등락 범위를 표준편차로 계산해 표시한 지표입니다.**
                
                쉽게 말해, 주가가 평소에 다니는 '도로의 폭'이라고 생각하면 됩니다.
                * **상단에 다다르면:** 주가가 단기적으로 너무 많이 올랐다는 신호입니다. (고평가 → 매도 고려)
                * **하단에 다다르면:** 주가가 단기적으로 너무 많이 떨어졌다는 신호입니다. (저평가 → 매수 고려)
                """)
            st.altair_chart(visualize_technical_indicators1(df), use_container_width=True)
        
        # 2. RSI
        with st.container(border=True, key="analysis_container3_2"):
            st.markdown("##### 2. RSI (상대강도지수)")
            with st.expander("📖 RSI가 뭔가요?"):
                st.info("""
                **일정 기간 동안 주가가 전일 대비 얼마나 상승했는지를 백분율(%)로 나타낸 지표입니다.**
                
                쉽게 말해, 시장의 분위기가 얼마나 뜨거운지 보여주는 '온도계(0~100점)'입니다.
                * **70점을 넘어서면:** 사는 사람이 너무 많아 '과열'된 상태입니다. (가격 하락 주의)
                * **30점 아래로 내려가면:** 파는 사람이 너무 많아 '침체'된 상태입니다. (반등 기회 가능)
                """)
            st.altair_chart(visualize_technical_indicators2(df), use_container_width=True)

    with tab4:
        with st.container(border=True, key="analysis_container4"):
            st.markdown("##### 보유 기간 누적 수익률")
            st.markdown("<br>", unsafe_allow_html=True)
            st.altair_chart(visualize_return_analysis(df), use_container_width=True)

def render_aipage():
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    st.markdown("""
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #ffffff;">
    <div class="analysis-header-text">InvestWiki</div>
    </nav>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem !important; 
    }
    </style>
    """, unsafe_allow_html=True)

    col_text, col_empty = st.columns([1, 5.5])
    with col_text:
        st.text("HOME  >  AI AGENT")

    # --- [여기부터 기존 채팅 로직 그대로 사용] ---
    st.markdown("### 투자 비서")
    st.caption("궁금한 점을 물어보세요.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 투자 도우미입니다."}]

    msgs = st.container(height=570)
    for m in st.session_state.messages:
        msgs.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("질문 입력..."):
        st.session_state.messages.append({"role":"user", "content":prompt})
        msgs.chat_message("user").write(prompt)
        
        # (답변 로직)
        ans = "AI 연결이 필요합니다."
        st.session_state.messages.append({"role":"assistant", "content":ans})
        msgs.chat_message("assistant").write(ans)

def render_sidebar():
    with st.sidebar: 
        # 1. 아이콘 URL 준비 (흰색)
        #url_hamb = "https://img.icons8.com/ios-glyphs/60/ffffff/menu--v1.png"
        url_home = "https://img.icons8.com/ios-glyphs/60/ffffff/home.png"
        url_plus = "https://img.icons8.com/ios-glyphs/60/ffffff/plus-math.png"
        url_ai = 'https://raw.githubusercontent.com/minjun069/DartB/refs/heads/main/gemini-color.png'

        # 2. Base64 변환
        #img_hamb = get_image_base64_from_url(url_hamb)
        img_home = get_image_base64_from_url(url_home)
        img_plus = get_image_base64_from_url(url_plus)
        img_ai = get_image_base64_from_url(url_ai)
        
        images = [img for img in [img_home, img_plus, img_ai] if img is not None]

        if images:
            # 3. 클릭 가능한 이미지 생성
            clicked = clickable_images(
                paths=images, 
                titles=["홈으로 가기", "새 분석 추가", 'AI'],
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
                    "margin": "10px", 
                    "height": "40px", 
                    "cursor": "pointer"
                }, 
                key=str(st.session_state.menu_key) 
            )

            # 4. 클릭 이벤트 처리
            if clicked > -1:
                st.session_state.menu_key += 1 # 컴포넌트 리셋
                
                if clicked == 0: # 홈
                    st.session_state.current_page_id = "HOME"
                    st.rerun()
                    
                elif clicked == 1: # 추가
                    new_id = str(uuid.uuid4())
                    new_title = f"분석 리포트 {len(st.session_state.analysis_pages) + 1}"
                    
                    st.session_state.analysis_pages.append({
                        "id": new_id,
                        "title": new_title,
                        "ticker": None, # 아직 종목 선택 안됨,
                        "stock_name": None
                    })
                    
                    st.session_state.current_page_id = new_id
                    st.rerun()

                elif clicked == 2:
                    st.session_state.current_page_id = "AI"
                    st.rerun()

        st.divider()

        # 5. 생성된 리포트 목록 표시
        st.caption("📑 생성된 리포트 목록")
        
        if not st.session_state.analysis_pages:
            st.info("생성된 분석 페이지가 없습니다.")
        
        else:
            for page in st.session_state.analysis_pages:
                # 현재 선택된 페이지 강조
                btn_type = "primary" if st.session_state.current_page_id == page["id"] else "secondary"
                
                col_nav, col_del = st.columns([0.8, 0.2])
                with col_nav:
                    # 페이지 이동 버튼
                    if st.button(page["title"], key=f"nav_{page['id']}", type=btn_type, use_container_width=True):
                        st.session_state.current_page_id = page["id"]
                        st.rerun()
                
                with col_del:
                    # 삭제 버튼 (X 또는 쓰레기통 아이콘)
                    # key는 유니크해야 하므로 page_id를 포함시킴
                    if st.button("✕", key=f"del_{page['id']}", help="이 리포트 삭제", use_container_width=True):
                        # 1. 리스트에서 해당 페이지 삭제
                        st.session_state.analysis_pages.remove(page)
                        
                        # 2. 만약 현재 보고 있던 페이지를 삭제했다면 홈으로 이동
                        if st.session_state.current_page_id == page["id"]:
                            st.session_state.current_page_id = "HOME"
                            
                        # 3. 변경사항 반영을 위해 새로고침
                        st.rerun()
            
            # 6. 초기화 버튼
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("모든 페이지 초기화", type="secondary", use_container_width=True):
                st.session_state.analysis_pages = []
                st.session_state.current_page_id = "HOME"
                st.session_state.menu_key += 1
                st.rerun()

def render_floating_chatbot():

    if "is_chat_open" not in st.session_state:
            st.session_state.is_chat_open = False

    image_url = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
    chatbot_img_base64 = get_image_base64_from_url(image_url)

    # 2. 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]

    st.markdown(f"""
    <div class="chatbot-visual"></div>
        <style>
        .chatbot-visual {{
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        width: 70px !important;
        height: 70px !important;
        z-index: 999998 !important; /* 버튼보다 한 단계 아래 */

        background-image: url('{chatbot_img_base64}') !important;
        background-size: 60% !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-color: #5D87FF !important;

        border-radius: 50% !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        pointer-events: none !important; /* 👈 핵심: 클릭 무시 */}}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        div[data-testid="stPopover"] {
            /* 위치 고정 (화면 우측 하단) */
            position: fixed !important;
            bottom: 30px !important;
            right: 30px !important;
            z-index: 999999 !important; /* 다른 요소보다 무조건 위에 */
                
            /* 크기 및 모양 */
            width: 70px !important;
            height: 70px !important;
            opacity: 0 !important;

            /* 폰트 크기 (이모지 크기) */
            font-size: 40px !important;
            
            /* 기타 */
            align-items: center !important;
            justify-content: center !important;
        }</style>""", unsafe_allow_html=True)

    # 5. 버튼 로직 실행
    with st.popover(""):
        # --- [여기부터 기존 채팅 로직 그대로 사용] ---
        st.markdown("### 투자 비서")
        st.caption("궁금한 점을 물어보세요.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 투자 도우미입니다."}]

        msgs = st.container(height=300)
        for m in st.session_state.messages:
            msgs.chat_message(m["role"]).write(m["content"])

        if prompt := st.chat_input("질문 입력..."):
            st.session_state.messages.append({"role":"user", "content":prompt})
            msgs.chat_message("user").write(prompt)
            
            # (답변 로직)
            ans = "AI 연결이 필요합니다."
            st.session_state.messages.append({"role":"assistant", "content":ans})
            msgs.chat_message("assistant").write(ans)

# =========================
# 6. 메인 실행 루프
# =========================

# 세션 초기화
if "analysis_pages" not in st.session_state:
    st.session_state.analysis_pages = []
if "analysis_dates" not in st.session_state:
    st.session_state.analysis_dates = [None, None]
if "current_page_id" not in st.session_state:
    st.session_state.current_page_id = "HOME"
if "menu_key" not in st.session_state:
    st.session_state.menu_key = 0
if "popular_indices" not in st.session_state:
    st.session_state.popular_indices = list(range(len(ALL_POPULAR_STOCKS)))

# 1. 사이드바 렌더링 (항상 표시)
render_sidebar()

# 2. 메인 콘텐츠 라우팅
if st.session_state.current_page_id == "HOME":
    render_home()
    render_floating_chatbot()
elif st.session_state.current_page_id == "AI":
    render_aipage()
else:
    render_analysis(st.session_state.current_page_id)
    render_floating_chatbot()
