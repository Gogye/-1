import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
import altair as alt
import ta

pinpoints_df = pd.DataFrame({
    'Date': ['2024-06-05', '2024-10-10'],
    'Event': ['Vision Pro 발표', '신제품 출시'],
    'Content': ['Apple이 Vision Pro를 발표했습니다.', 'Apple이 새로운 제품을 출시했습니다.'],
    'Link': ['https://www.apple.com/newsroom/2024/06/apple-unveils-vision-pro-revolutionary-spatial-computing-platform/',
             'https://www.apple.com/newsroom/2024/10/apple-announces-new-products/']
})

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


def display_metrics(df):
    if len(df) < 2: return
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    close_price = latest['Close']
    price_diff = close_price - prev['Close']
    pct_change = (price_diff / prev['Close']) * 100
    volume = latest['Volume']
    rsi = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().iloc[-1]
    high_52w = df['Close'][-250:].max() if len(df) > 250 else df['Close'].max()
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric(label="현재 주가", value=f"{close_price:,.0f} 원", delta=f"{price_diff:,.0f} 원 ({pct_change:+.2f}%)")
    with m2: st.metric(label="거래량", value=f"{volume:,.0f} 주")
    with m3: st.metric(label="RSI (14일)", value=f"{rsi:.2f}")
    with m4: st.metric(label="52주 최고가", value=f"{high_52w:,.0f} 원")
    st.divider()

# --- [새로 추가] 캔들스틱 차트 함수 ---
def visualize_candlestick(df):
    df_reset = df.reset_index().rename(columns={'index': 'Date'})
    base = alt.Chart(df_reset).encode(x=alt.X('Date:T', axis=alt.Axis(format='%Y-%m-%d', title='날짜')))
    rule = base.mark_rule().encode(
        y=alt.Y('Low:Q', scale=alt.Scale(zero=False), title='주가'), y2='High:Q',
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff"))
    )
    bar = base.mark_bar(width=5).encode(
        y='Open:Q', y2='Close:Q',
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff")),
        tooltip=['Date:T', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    return (rule + bar).properties(height=300, title="일봉 캔들 차트")

# ----------------------------------------------------------------------
# 4. 시각화 함수 (Cell 3 수정)
# ----------------------------------------------------------------------
def visualize_phases_altair_all_interactions(df, pinpoints_df=None):
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
            phase_blocks_empty = False
            # -------------------------------------------------------
            # [수정됨] 한국 주식 스타일 색상 적용 (상승=빨강, 하락=파랑)
            # -------------------------------------------------------
            
            # 1. 어떤 구간인지 정의 (순서 중요!)
            domain = ['상승', '하락', '박스권']
            
            # 2. 각 구간별 색상 지정 (은은한 파스텔톤)
            # 상승(빨강) / 하락(파랑) / 박스권(회색)
            range_ = ['#ff9999', '#aaccff', '#d9d9d9'] 

            background = alt.Chart(phase_blocks).mark_rect(opacity=0.5).encode(
                x=alt.X('start_date:T', title='날짜'), 
                x2=alt.X2('end_date:T'),
                color=alt.Color('Phase:N', 
                                scale=alt.Scale(domain=domain, range=range_),  # <-- 이 부분이 새로 추가된 핵심입니다!
                                legend=alt.Legend(title='추세 구간')),
                tooltip=['start_date:T', 'end_date:T', 'Phase:N']
            )

    # --- 3. (전경) 선 그래프 (이전과 동일) ---
    line_chart = alt.Chart(df_reset).mark_line(color='gray').encode(
        x=alt.X('Date:T', title='날짜'),
        y=alt.Y('Close:Q', title='가격', scale=alt.Scale(zero=False)),
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
        pinpoints_df_copy['Date'] = pd.to_datetime(pinpoints_df_copy['Date'])
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
                strokeWidth=0.5
            ).transform_calculate(
                pin_y_position=f"{target_y_value}"  # 계산된 Y 위치 사용
            ).encode(
                x='Date:T',
                y=alt.Y('pin_y_position:Q', title='가격'),
                
                # 1. 툴팁 (Tooltip): 마우스 오버 시 정보 표시
                tooltip=[
                    alt.Tooltip('Date:T', title='날짜', format='%Y-%m-%d'),
                    alt.Tooltip('Event:N', title='이벤트')
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
        height=400
    )
    
    return main_chart


# ----------------------------------------------------------------------
# 5. Streamlit 앱 메인 로직
# ----------------------------------------------------------------------
st.set_page_config(layout="wide") # 페이지를 넓게 사용
st.title("주가 추세 구간화 알고리즘 (구간화 알고리즘_최종1차)")

cols = st.columns([1, 3])

left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)


STOCKS = [
    "005930",
    "000270",
    "005932",
]
DEFAULT_STOCKS = ["005930"]

def stocks_to_str(stocks):
    return ",".join(stocks)

if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = st.query_params.get(
        "stocks", stocks_to_str(DEFAULT_STOCKS)
    ).split(",")

all_options = sorted(set(STOCKS) | set(st.session_state.tickers_input))
default_ticker = "005930"
if st.session_state.tickers_input:
    default_ticker = st.session_state.tickers_input[0] # 리스트의 첫 번째 값
try:
    default_index = all_options.index(default_ticker)
except ValueError:
    default_index = 0 # 기본값이 옵션에 없으면 0번째(첫 번째) 항목 선택
    
with left_cell:
    st.markdown("### 주가 구간화 알고리즘")
    # --- 사이드바: 사용자 입력 ---
    ticker = st.selectbox(
        "종목 선택",
        options=all_options,
        index=all_options.index(st.session_state.tickers_input[0]),
        placeholder="종목 코드를 입력하세요 (예: 005930)"
    )
    with st.expander("### 📈 기간 설정"):
        start_date = st.date_input("시작일", pd.to_datetime("2024-01-01"))
        end_date = st.date_input("종료일", pd.to_datetime("2024-12-31"))

    with st.expander("### ⚙️ 구간화 파라미터"):
        # 노트북 Cell 7의 파라미터들
        window_length = st.number_input("스무딩 윈도우 (홀수)",min_value=3,max_value=21,value=5,step=2)
        polyorder = st.slider("스무딩 다항식 차수", 1, 5, 3)
        min_days1 = st.slider("초기 짧은 구간 병합 일수", 1, 10, 2)
        min_days2 = st.slider("최종 짧은 구간 병합 일수", 1, 10, 2)
        adjust_window = st.slider("전환점 보정 윈도우", 1, 10, 2)
        min_hits = st.slider("박스권 최소 교차 횟수", 1, 20, 9)
        box_window = st.slider("박스권 판정 윈도우", 1, 20, 10)


right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)


with right_cell:  
    # --- 메인 패널: 결과 출력 ---
    if ticker:
        # 1. 데이터 로드
        df_raw = load_data(ticker, start_date, end_date)
        
        if df_raw is not None and not df_raw.empty:
            # [NEW] 상단 주요 지표 대시보드 표시
            display_metrics(df_raw)
            
            # [NEW] 탭 구성 (기본 차트 vs 알고리즘 분석)
            tab1, tab2 = st.tabs(["📈 기본 시세", "🧠 AI 추세 분석"])
            
            # 탭 1: 캔들스틱 차트 (새로 추가된 기능)
            with tab1:
                candle_chart = visualize_candlestick(df_raw)
                st.altair_chart(candle_chart, use_container_width=True)
                st.subheader("일별 시세 데이터")
                st.dataframe(df_raw.sort_index(ascending=False).head(5), use_container_width=True)

            # 탭 2: 기존 알고리즘 분석 (기존 기능 이동)
            with tab2:
                if len(df_raw) < window_length:
                    st.warning(f"데이터 부족: 최소 {window_length}일 이상 필요합니다.")
                else:
                    with st.spinner("구간화 알고리즘을 실행 중입니다..."):
                        df_processed = detect_market_phases(
                            df_raw, window_length, polyorder, min_days1, min_days2, adjust_window, min_hits, box_window
                        )
                    
                    st.subheader("구간화 분석 결과")
                    fig = visualize_phases_altair_all_interactions(df_processed, pinpoints_df=pinpoints_df)
                    st.altair_chart(fig, use_container_width=True)
                    
                    # 통계 요약 추가
                    if "Phase" in df_processed.columns:
                        counts = df_processed['Phase'].value_counts()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("상승 구간", f"{counts.get('상승', 0)}일")
                        c2.metric("하락 구간", f"{counts.get('하락', 0)}일")
                        c3.metric("박스권", f"{counts.get('박스권', 0)}일")
                    
                    st.subheader("관련 뉴스 이벤트")
                    st.dataframe(pinpoints_df, use_container_width=True, hide_index=True)
    else:
        st.info("좌측 사이드바에서 종목 코드를 입력하고 기간을 설정해주세요.")