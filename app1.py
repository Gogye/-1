import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
import altair as alt
import ta

# ----------------------------------------------------------------------
# 0. 디자인 설정 및 CSS 커스터마이징 (추가/수정된 핵심 부분)
# ----------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="주가 추세 분석기 (Final)",
    # icon="📈"
)

# 깔끔한 디자인을 위한 CSS 적용
st.markdown("""
<style>
/* 폰트 및 기본 설정: 깔끔한 sans-serif 폰트 */
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;700&display=swap');
html, body, [class*="st-emotion-cache"] {
    font-family: 'Pretendard', sans-serif;
}

/* 주요 제목 (H1) 디자인: 굵게, 넓은 공간 */
h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a1a1a;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e0e0e0;
}

/* 모든 컨테이너 (st.container) 및 위젯에 부드러운 모서리, 은은한 그림자 적용 */
.st-emotion-cache-1kyxreq { /* Container/Block selector for main content */
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); /* 은은한 그림자 */
    transition: all 0.3s ease;
}

/* st.metric 배경과 폰트 */
[data-testid="stMetric"] > div {
    background-color: #f7f9fc; /* 연한 배경색 */
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #4a90e2; /* 포인트 색상 */
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
}
[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    color: #555555 !important;
}

/* Expander (접기) 디자인 */
[data-testid="stExpander"] > div > div:first-child {
    background-color: #f0f4f8;
    border-radius: 8px;
    padding: 10px 15px;
    margin-bottom: 5px;
    font-weight: 600;
}

/* Info 메시지 (st.info) */
.st-emotion-cache-12fmwpl {
    border-radius: 8px;
    background-color: #e6f7ff; /* 라이트 블루 */
    border-left: 5px solid #1890ff; /* 진한 파랑 */
}

/* 탭 디자인 개선 */
[data-testid="stTab"] {
    border-radius: 8px 8px 0 0 !important;
    margin-right: 5px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


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
    rsi = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().iloc[-1] if len(df) >= 14 else np.nan
    high_52w = df['Close'][-250:].max() if len(df) > 250 else df['Close'].max()
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric(label="현재 주가", value=f"{close_price:,.0f} 원", delta=f"{price_diff:,.0f} 원 ({pct_change:+.2f}%)")
    with m2: st.metric(label="거래량", value=f"{volume:,.0f} 주")
    with m3: st.metric(label="RSI (14일)", value=f"{rsi:.2f}" if not np.isnan(rsi) else "N/A")
    with m4: st.metric(label="52주 최고가", value=f"{high_52w:,.0f} 원")
    st.divider()

def visualize_candlestick(df):
    df_reset = df.reset_index().rename(columns={'index': 'Date'})
    
    # [핵심 해결책] 캔들 너비를 '픽셀'이 아닌 '시간 간격'으로 정의합니다.
    df_reset['Date_start'] = df_reset['Date'] - pd.Timedelta(hours=9)
    df_reset['Date_end']   = df_reset['Date'] + pd.Timedelta(hours=9)

    # 1. 캔들 꼬리 (High-Low) 그리기 (얇은 선)
    rule = alt.Chart(df_reset).mark_rule().encode(
        x=alt.X('Date:T', axis=alt.Axis(format='%Y-%m-%d', title='날짜')),
        y=alt.Y('Low:Q', scale=alt.Scale(zero=False), title='주가'),
        y2='High:Q',
        # 한국 주식 색상 (빨강=상승, 파랑=하락)
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff"))
    )

    # 2. 캔들 몸통 (Open-Close) 그리기 (사각형 영역)
    body = alt.Chart(df_reset).mark_rect().encode(
        x='Date_start:T',
        x2='Date_end:T',
        y='Open:Q',
        y2='Close:Q',
        # 한국 주식 색상 (빨강=상승, 파랑=하락)
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff")),
        tooltip=['Date:T', 'Open', 'High', 'Low', 'Close', 'Volume']
    )

    # 차트 합치기 + 인터랙티브 기능
    chart = (rule + body).properties(
        height=300,
        title="일봉 캔들 차트"
    ).interactive()
    
    return chart

# --- [복구됨] 3. 기술적 지표 시각화 (NaN 처리 + 한국식 색상) ---
def visualize_technical_indicators(df):
    df = df.copy()
    
    # 데이터 길이 체크 (최소 30일)
    if len(df) < 30:
        return alt.Chart(pd.DataFrame({'text': ['데이터가 부족하여 지표를 계산할 수 없습니다. (최소 30일 이상 필요)']})).mark_text(size=20).encode(text='text')

    # 1. 지표 계산
    # 볼린저 밴드
    indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_h'] = indicator_bb.bollinger_hband()
    df['bb_l'] = indicator_bb.bollinger_lband()
    
    # MACD
    indicator_macd = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = indicator_macd.macd()
    df['macd_signal'] = indicator_macd.macd_signal()
    df['macd_diff'] = indicator_macd.macd_diff()

    # RSI
    indicator_rsi = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df['rsi'] = indicator_rsi.rsi()

    # [중요] 빈 값(NaN) 제거
    df_reset = df.dropna().reset_index().rename(columns={'index': 'Date'})
    
    if df_reset.empty:
          return alt.Chart(pd.DataFrame({'text': ['유효한 데이터가 없습니다.']})).mark_text().encode(text='text')

    # 2. 차트 그리기
    base = alt.Chart(df_reset).encode(x=alt.X('Date:T', axis=alt.Axis(title=None, format='%Y-%m-%d')))

    # (1) 볼린저 밴드
    bb_line = base.mark_line(color='black', strokeWidth=1).encode(y=alt.Y('Close:Q', scale=alt.Scale(zero=False), title='주가'))
    bb_band = base.mark_area(opacity=0.2, color='#aaccff').encode(y='bb_l:Q', y2='bb_h:Q') # 밴드 색상 변경
    chart_bb = (bb_line + bb_band).properties(height=250, title="볼린저 밴드 (가격 변동폭)")

    # (2) MACD (상승=빨강, 하락=파랑)
    macd_line = base.mark_line(color='grey').encode(y='macd:Q')
    sig_line = base.mark_line(color='#ff9999').encode(y='macd_signal:Q') # 시그널 색상 변경
    hist_bar = base.mark_bar().encode(
        y=alt.Y('macd_diff:Q', title='MACD Diff'),
        color=alt.condition(alt.datum.macd_diff > 0, alt.value("#ff0000"), alt.value("#0000ff")) # 막대 색상 명확화
    )
    chart_macd = (hist_bar + macd_line + sig_line).properties(height=150, title="MACD (추세 강도)")

    # (3) RSI
    rsi_line = base.mark_line(color='#4a90e2').encode(y=alt.Y('rsi:Q', scale=alt.Scale(domain=[0, 100]), title='RSI')) # 선 색상 변경
    rsi_rule_high = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(color='#ff0000', strokeDash=[3,3]).encode(y='y')
    rsi_rule_low = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(color='#0000ff', strokeDash=[3,3]).encode(y='y')
    chart_rsi = (rsi_line + rsi_rule_high + rsi_rule_low).properties(height=150, title="RSI (과열/침체)")

    return alt.vconcat(chart_bb, chart_macd, chart_rsi).resolve_scale(x='shared').interactive()

# --- [복구됨] 4. 수익률 분석 시각화 ---
def visualize_return_analysis(df):
    df = df.copy()
    # 수익률 계산
    df['Daily_Ret'] = df['Close'].pct_change()
    df['Cum_Ret'] = (1 + df['Daily_Ret']).cumprod() - 1
    df_reset = df.dropna().reset_index().rename(columns={'index': 'Date'})

    # (1) 누적 수익률 곡선
    cum_chart = alt.Chart(df_reset).mark_area(
        line={'color':'#4a90e2'}, # 선 색상 변경
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0),
                   alt.GradientStop(color='#aaccff', offset=1)], # 채우기 색상 변경
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('Date:T', title='날짜'),
        y=alt.Y('Cum_Ret:Q', title='누적 수익률', axis=alt.Axis(format='%')),
        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Cum_Ret:Q', format='.2%')]
    ).properties(height=300, title="누적 수익률 추이 (Cumulative Return)").interactive()

    # (2) 수익률 분포 히스토그램
    hist_chart = alt.Chart(df_reset).mark_bar().encode(
        x=alt.X('Daily_Ret:Q', bin=alt.Bin(maxbins=50), title='일별 등락률'),
        y=alt.Y('count()', title='빈도수'),
        color=alt.value('#4a90e2') # 막대 색상 변경
    ).properties(height=200, title="일별 등락률 분포 (Histogram)")

    return alt.vconcat(cum_chart, hist_chart)
# ----------------------------------------------------------------------
# 4. 시각화 함수 (Cell 3 수정)
# ----------------------------------------------------------------------
def visualize_phases_altair_all_interactions(df, pinpoints_df=None):
    """
    Altair의 4가지 주요 상호작용을 모두 포함하는 차트를 생성합니다.
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
            
            # 2. 각 구간별 색상 지정 (은은한 파스텔톤 유지)
            # 상승(빨강) / 하락(파랑) / 박스권(회색)
            range_ = ['#ff9999', '#aaccff', '#d9d9d9'] 

            background = alt.Chart(phase_blocks).mark_rect(opacity=0.5).encode(
                x=alt.X('start_date:T', title='날짜'), 
                x2=alt.X2('end_date:T'),
                color=alt.Color('Phase:N', 
                                 scale=alt.Scale(domain=domain, range=range_), 
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
                strokeWidth=0.5,
                color='gold', # 핀포인트 색상 강조
                size=100
            ).transform_calculate(
                pin_y_position=f"{target_y_value}"  # 계산된 Y 위치 사용
            ).encode(
                x='Date:T',
                y=alt.Y('pin_y_position:Q', title='가격'),
                
                # 1. 툴팁 (Tooltip): 마우스 오버 시 정보 표시
                tooltip=[
                    alt.Tooltip('Date:T', title='날짜', format='%Y-%m-%d'),
                    alt.Tooltip('Event:N', title='이벤트')
                ],
                
                # 2. 하이라이트 (Highlight): 마우스 오버 시 크기 변경
                size=alt.condition(hover_selection, 
                                   alt.value(250),alt.value(100)  # 마우스 올리면 250, 평상시 100
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
# 5. Streamlit 앱 메인 로직 (레이아웃 및 제목 수정)
# ----------------------------------------------------------------------
st.title("주가 추세 구간화 분석기") # 제목 간결화

cols = st.columns([1, 3])

# 좌측 컨테이너 (파라미터 설정)
left_cell = cols[0].container()

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
    st.markdown("### 🛠️ 분석 파라미터")
    # --- 종목 선택 ---
    ticker = st.selectbox(
        "종목 선택",
        options=all_options,
        index=all_options.index(st.session_state.tickers_input[0]),
        placeholder="종목 코드를 입력하세요 (예: 005930)"
    )
    
    st.markdown("---") # 구분선 추가

    with st.expander("### 📅 기간 설정", expanded=True):
        start_date = st.date_input("시작일", pd.to_datetime("2024-01-01"))
        end_date = st.date_input("종료일", pd.to_datetime("2024-12-31"))

    with st.expander("### ⚙️ 알고리즘 파라미터"):
        # 노트북 Cell 7의 파라미터들
        window_length = st.number_input("스무딩 윈도우 (홀수)",min_value=3,max_value=21,value=5,step=2)
        polyorder = st.slider("스무딩 다항식 차수", 1, 5, 3)
        min_days1 = st.slider("초기 짧은 구간 병합 일수", 1, 10, 2)
        min_days2 = st.slider("최종 짧은 구간 병합 일수", 1, 10, 2)
        adjust_window = st.slider("전환점 보정 윈도우", 1, 10, 2)
        min_hits = st.slider("박스권 최소 교차 횟수", 1, 20, 9)
        box_window = st.slider("박스권 판정 윈도우", 1, 20, 10)


# 우측 컨테이너 (결과 출력)
right_cell = cols[1].container()


with right_cell:
    # --- 메인 패널: 결과 출력 ---
    if ticker:
        # 1. 데이터 로드
        df_raw = load_data(ticker, start_date, end_date)
        
        if df_raw is not None and not df_raw.empty:
            
            # 상단 주요 지표
            st.markdown("## 📊 종목 개요 및 주요 지표")
            display_metrics(df_raw)
            
            # 탭 4개 구성
            tab1, tab2, tab3, tab4 = st.tabs(["📈 캔들 시세", "🧠 AI 추세", "⚙️ 기술 지표", "📈 수익률"])
            
            # [Tab 1] 캔들스틱 
            with tab1:
                candle_chart = visualize_candlestick(df_raw)
                st.altair_chart(candle_chart, use_container_width=True)
                st.subheader("📝 일별 데이터")
                st.dataframe(df_raw.sort_index(ascending=False).head(10), use_container_width=True)

            # [Tab 2] 알고리즘 분석 
            with tab2:
                if len(df_raw) < window_length:
                    st.warning(f"데이터 부족: 최소 {window_length}일 이상 필요합니다.")
                else:
                    with st.spinner("추세 패턴 분석 중..."):
                        df_processed = detect_market_phases(
                            df_raw, window_length, polyorder, min_days1, min_days2, adjust_window, min_hits, box_window
                        )
                    
                    st.subheader("🤖 AI 추세 구간 분석 결과")
                    fig = visualize_phases_altair_all_interactions(df_processed, pinpoints_df=pinpoints_df)
                    st.altair_chart(fig, use_container_width=True)
                    
                    if "Phase" in df_processed.columns:
                        counts = df_processed['Phase'].value_counts()
                        st.markdown("#### 추세 분포 요약")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("상승 구간", f"{counts.get('상승', 0)}일")
                        c2.metric("하락 구간", f"{counts.get('하락', 0)}일")
                        c3.metric("박스권", f"{counts.get('박스권', 0)}일")
                    
                    st.subheader("📰 뉴스 이벤트 매칭")
                    st.dataframe(pinpoints_df, use_container_width=True, hide_index=True)

            # [Tab 3] 기술적 지표
            with tab3:
                st.subheader("📐 기술적 지표 분석")
                
                # 1. 초보자용 요약 (유지)
                st.info("""
                **💡 초보자를 위한 1분 요약**
                * **볼린저 밴드:** 주가가 회색 띠를 벗어나면 다시 돌아오려는 성질이 있어요. (밴드 상단=비쌈, 하단=쌈)
                * **MACD:** 빨간 막대가 커지면 '상승세', 파란 막대가 커지면 '하락세'입니다.
                * **RSI:** 70을 넘으면 '과열(비쌈)', 30 밑이면 '침체(쌈)' 신호입니다.
                """)

                # 2. 차트
                tech_chart = visualize_technical_indicators(df_raw)
                st.altair_chart(tech_chart, use_container_width=True)
                
                # 3. 상세 설명 (Expander) (유지)
                with st.expander("📚 지표 상세 해석 가이드 (눌러서 보기)"):
                    st.markdown("""
                    ### 1. 볼린저 밴드 (Bollinger Bands)
                    - **무엇인가요?** 주가가 다니는 '길'이라고 생각하세요. 
                    - **해석법:** 주가는 보통 밴드 안에서 움직입니다. 
                         - 캔들이 **위쪽 선**을 치면? 단기 고점일 수 있습니다. (매도 고려)
                         - 캔들이 **아래쪽 선**을 치면? 단기 저점일 수 있습니다. (매수 고려)
                    
                    ### 2. MACD (추세)
                    - **무엇인가요?** 주가의 '방향'과 '에너지'를 보여줍니다.
                    - **해석법:** - **빨간 막대**가 점점 길어지면 상승 힘이 강해지는 것입니다.
                         - **파란 막대**가 줄어들면서 빨간색으로 바뀌려는 순간이 '매수 타이밍'으로 불립니다.
                    
                    ### 3. RSI (상대강도지수)
                    - **무엇인가요?** 시장의 '과열' 여부를 0~100 점수로 매긴 것입니다.
                    - **해석법:**
                         - **70 이상 (점선 위):** "너무 뜨겁다!" 사람들이 너무 많이 사서 비싼 상태일 수 있습니다. (조심!)
                         - **30 이하 (점선 아래):** "너무 차갑다!" 사람들이 너무 많이 팔아서 싼 상태일 수 있습니다. (기회?)
                    """)

            # [Tab 4] 수익률 분석
            with tab4:
                st.subheader("📊 수익률 퍼포먼스")
                st.caption("이 기간 동안 보유했을 때의 누적 수익률과 변동성입니다.")
                return_chart = visualize_return_analysis(df_raw)
                st.altair_chart(return_chart, use_container_width=True)

        else:
            st.info("좌측 사이드바에서 분석할 종목을 선택하고 기간을 설정해주세요.")
    else:
        st.info("좌측 사이드바에서 분석할 종목을 선택해주세요.")
