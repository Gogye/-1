import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import koreanize_matplotlib  # 한글 깨짐 방지
from scipy.signal import savgol_filter
import math
import ta  # 노트북에서 import했으므로 포함 (현재 로직에서는 사용되지 않음)

# ----------------------------------------------------------------------
# 1. 데이터 로딩 함수 (캐싱 적용)
# ----------------------------------------------------------------------
@st.cache_data # 데이터 로딩 결과를 캐시합니다.
def load_data(ticker, start_date, end_date):
    """
    FinanceDataReader를 사용해 주가 데이터를 불러옵니다.
    """
    try:
        df = fdr.DataReader(ticker, start_date, end_date)
        df = df.dropna()
        if df.empty:
            st.error("해당 기간에 데이터가 없습니다.")
            return None
        return df.copy()
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return None

# ----------------------------------------------------------------------
# 2. 노트북의 알고리즘 함수들 (ipynb 파일 내용 그대로)
# ----------------------------------------------------------------------

# --- 2-1. 스무딩 & 초기 Phase (Cell 4) ---
def apply_smoothing_and_phase(df, window_length, polyorder):
    df = df.copy()
    # Savitzky-Golay 필터 적용
    if len(df) < window_length:
        st.warning("데이터가 스무딩 윈도우보다 적어 스무딩을 적용할 수 없습니다.")
        df["Smooth"] = df["Close"]
    else:
        df["Smooth"] = savgol_filter(df["Close"], window_length=window_length, polyorder=polyorder)
    
    df["Slope"] = np.gradient(df["Smooth"])
    classify = lambda s: "상승" if s > 0 else "하락"
    df["Phase"] = df["Slope"].apply(classify)
    return df

# --- 2-2. 박스권 탐지 (Cell 4) ---
def apply_box_range(df, min_hits, window):
    df = df.copy() # 원본 데이터 수정을 방지하기 위해 복사
    
    if df.empty:
        return df

    p_min, p_max = df['Close'].min(), df['Close'].max()
    limit = (p_max - p_min) / 25
    
    diffs = df['Close'].diff().abs()
    min_step = diffs[diffs > 0].min()
    
    if pd.isna(min_step): # 데이터가 너무 적거나 변동이 없는 경우
        min_step = 10 
        
    exponent = int(math.floor(math.log10(min_step)))
    step = 10 ** exponent if exponent >= 1 else 10
    
    # 로직 1: 가격 레벨 교차 기반
    for k in np.arange(p_min, p_max, step):
        crossings = [False] * len(df)
        for i in range(1, len(df)):
            y0, y1 = df['Close'].iloc[i-1], df['Close'].iloc[i]
            if (y0 - k) * (y1 - k) <= 0:
                crossings[i-1] = True
                crossings[i] = True
        
        if len(crossings) <= window:
            continue

        for i in range(1, len(crossings) - window):
            if sum(crossings[i:i+window]) >= min_hits:
                if abs(df["Close"].iloc[i+window] - df["Close"].iloc[i]) <= limit:
                    # .loc를 사용하여 안전하게 값 할당
                    df.loc[df.index[i:i+min_hits], "Phase"] = "박스권"
    
    if len(df) <= window:
        return df # 윈도우보다 데이터가 적으면 아래 로직 수행 불가

    # 로직 2: 윈도우 내 변동성 기반
    for i in range(len(df) - window):
        window_prices = df["Close"].iloc[i:i+window]
        window_mean = window_prices.mean()
        upper = window_mean + limit
        lower = window_mean - limit
        if window_prices.max() <= upper and window_prices.min() >= lower:
            df.loc[df.index[i:i+window], "Phase"] = "박스권"
            
    return df

# --- 2-3. 짧은 구간 병합 (Cell 5) ---
def merge_short_phases(df, min_days):
    df = df.copy()
    if "Phase" not in df.columns or df.empty:
        return df
        
    df["group_id"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    df["group_size"] = df.groupby("group_id")["Phase"].transform("size")
    
    unique_group_ids = df["group_id"].unique()
    if len(unique_group_ids) < 2: # 그룹이 1개 이하면 병합할 대상이 없음
        return df

    min_group_id = df["group_id"].min()
    max_group_id = df["group_id"].max()

    for group_id in unique_group_ids:
        mask = df["group_id"] == group_id
        size = df.loc[mask, "group_size"].iloc[0]
        
        if size <= min_days and group_id > min_group_id:
            if group_id == max_group_id:
                continue # 마지막 그룹은 다음 그룹이 없으므로 병합 안 함
                
            g_min, g_max = df.loc[mask, 'Close'].min(), df.loc[mask, 'Close'].max()
            if g_max - g_min >= (df['Close'].max() - df['Close'].min()) / 5:
                continue
                
            prev_phase = df.loc[df["group_id"] == group_id - 1, "Phase"].iloc[0]
            next_phase = df.loc[df["group_id"] == group_id + 1, "Phase"].iloc[0]
            
            if prev_phase != '박스권':
                df.loc[mask, "Phase"] = prev_phase
            elif next_phase != '박스권':
                df.loc[mask, "Phase"] = next_phase
    return df

# --- 2-4. 전환점 보정 (Cell 5) ---
def adjust_change_points(df, adjust_window):
    df = df.copy()
    if "Phase" not in df.columns or df.empty or len(df) < adjust_window:
        return df
        
    df["group_id"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    change_points = df.index[df["Phase"] != df["Phase"].shift()]
    
    if len(change_points) < 2: # 전환점이 1개 이하면 보정할 필요 없음
        return df

    for cp in change_points:
        cp_idx = df.index.get_loc(cp) # 구간 첫 시작점
        if cp_idx == 0: continue # 첫 번째 데이터는 보정 대상 아님

        current_phase = df.loc[cp, "Phase"]
        prev_phase = df.loc[df.index[cp_idx - 1], "Phase"]
        
        start_win = max(0, cp_idx - adjust_window)
        end_win = min(len(df), cp_idx + adjust_window + 1)
        window_data = df.iloc[start_win:end_win]

        if window_data.empty:
            continue

        if current_phase == "상승":
            local_min_idx = window_data["Close"].idxmin()
            local_min_pos = df.index.get_loc(local_min_idx)
            diff = abs(local_min_pos - cp_idx)
            
            if local_min_pos > cp_idx:
                df.loc[df.index[cp_idx:local_min_pos], "Phase"] = prev_phase
            elif local_min_pos < cp_idx:
                df.loc[df.index[local_min_pos:cp_idx], "Phase"] = "상승"
                
        elif current_phase == "하락":
            local_max_idx = window_data["Close"].idxmax()
            local_max_pos = df.index.get_loc(local_max_idx)
            diff = abs(local_max_pos - cp_idx)

            if local_max_pos > cp_idx:
                df.loc[df.index[cp_idx:local_max_pos], "Phase"] = prev_phase
            elif local_max_pos < cp_idx:
                df.loc[df.index[local_max_pos:cp_idx], "Phase"] = "하락"
    return df

# ----------------------------------------------------------------------
# 3. 알고리즘 실행 메인 함수 (Cell 6 수정)
# ----------------------------------------------------------------------
def detect_market_phases(df, window_length, polyorder, min_days1, min_days2, adjust_window, min_hits, box_window):
    """
    노트북의 알고리즘을 순서대로 실행합니다.
    (Cell 10의 실행 순서와 Cell 6의 함수 정의를 참고하여 재구성)
    """
    df_result = df.copy()
    
    # 1. 스무딩 & 초기 Phase
    df_result = apply_smoothing_and_phase(df_result, window_length, polyorder)
    
    # 2. 박스권 탐지
    df_result = apply_box_range(df_result, min_hits, box_window)
    
    # 3. 짧은 구간 병합 (1차)
    df_result = merge_short_phases(df_result, min_days1)
    
    # 4. 전환점 보정
    df_result = adjust_change_points(df_result, adjust_window)
    
    # 5. 짧은 구간 병합 (2차)
    df_result = merge_short_phases(df_result, min_days2)
    
    return df_result

# ----------------------------------------------------------------------
# 4. 시각화 함수 (Cell 3 수정)
# ----------------------------------------------------------------------
def visualize_phases_streamlit(df):
    """
    Streamlit에 Matplotlib 차트를 그리기 위한 함수
    plt.show() 대신 fig 객체를 반환합니다.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["Close"], color="gray", linewidth=2, label="실제 종가")
    
    # 스무딩 곡선이 있으면 함께 표시
    if "Smooth" in df.columns:
        ax.plot(df.index, df["Smooth"], color="black", alpha=0.5, linestyle='--', label="스무딩 곡선")

    colors = {"상승": "green", "하락": "red", '박스권': 'blue'}
    
    if "Phase" not in df.columns or df.empty:
        ax.set_title("데이터가 부족하거나 Phase가 계산되지 않았습니다.")
        return fig

    current_phase = df["Phase"].iloc[0]
    start_idx = 0

    for i in range(1, len(df)):
        if df["Phase"].iloc[i] != current_phase:
            ax.axvspan(df.index[start_idx], df.index[i],
                       color=colors.get(current_phase, 'grey'), alpha=0.15) # get으로 안전하게 색상 가져오기
            start_idx = i
            current_phase = df["Phase"].iloc[i]

    # 마지막 구간 색칠
    ax.axvspan(df.index[start_idx], df.index[-1],
               color=colors.get(current_phase, 'grey'), alpha=0.15)

    ax.set_title("알고리즘 기반 주가 추세 구간 시각화")
    ax.legend()
    return fig

# ----------------------------------------------------------------------
# 5. Streamlit 앱 메인 로직
# ----------------------------------------------------------------------
st.set_page_config(layout="wide") # 페이지를 넓게 사용
st.title("주가 추세 구간화 알고리즘 (구간화 알고리즘_최종1차)")

# --- 사이드바: 사용자 입력 ---
st.sidebar.header("📈 조회 설정")
ticker = st.sidebar.text_input("종목 코드 (예: 005930)", "005930")
start_date = st.sidebar.date_input("시작일", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("종료일", pd.to_datetime("2024-12-31"))

st.sidebar.header("⚙️ 알고리즘 파라미터")
# 노트북 Cell 7의 파라미터들
window_length = st.sidebar.slider("스무딩 윈도우 (홀수)", 3, 21, 5, step=2)
polyorder = st.sidebar.slider("스무딩 다항식 차수", 1, 5, 3)
min_days1 = st.sidebar.slider("초기 짧은 구간 병합 일수", 1, 10, 2)
min_days2 = st.sidebar.slider("최종 짧은 구간 병합 일수", 1, 10, 2)
adjust_window = st.sidebar.slider("전환점 보정 윈도우", 1, 10, 2)
min_hits = st.sidebar.slider("박스권 최소 교차 횟수", 1, 20, 9)
box_window = st.sidebar.slider("박스권 판정 윈도우", 1, 20, 10)


# --- 메인 패널: 결과 출력 ---
if ticker:
    # 1. 데이터 로드
    df_raw = load_data(ticker, start_date, end_date)
    
    if df_raw is not None and not df_raw.empty:
        st.subheader(f"'{ticker}' 원본 데이터 (최근 5일)")
        st.dataframe(df_raw.tail(), use_container_width=True)

        # 2. 알고리즘 실행
        if len(df_raw) < window_length:
            st.warning(f"데이터가 스무딩 윈도우({window_length}일)보다 적습니다. 더 긴 기간을 선택하세요.")
        else:
            with st.spinner("구간화 알고리즘을 실행 중입니다..."):
                df_processed = detect_market_phases(
                    df_raw,
                    window_length=window_length,
                    polyorder=polyorder,
                    min_days1=min_days1,
                    min_days2=min_days2,
                    adjust_window=adjust_window,
                    min_hits=min_hits,
                    box_window=box_window
                )
            
            # 3. 시각화
            st.subheader("구간화 분석 결과 차트")
            fig = visualize_phases_streamlit(df_processed)
            st.pyplot(fig, use_container_width=True)
            
            # 4. 데이터 표시
            st.subheader("구간화 상세 데이터")
            st.dataframe(df_processed, use_container_width=True)
            
            # 5. 다운로드 버튼
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=True, encoding='utf-8-sig').encode('utf-8-sig')

            csv_data = convert_df_to_csv(df_processed)
            st.download_button(
                label="📈 결과 데이터 다운로드 (CSV)",
                data=csv_data,
                file_name=f"{ticker}_phase_analysis.csv",
                mime="text/csv",
            )
else:
    st.info("좌측 사이드바에서 종목 코드를 입력하고 기간을 설정해주세요.")