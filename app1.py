import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
import altair as alt
import ta
import google.generativeai as genai

# ----------------------------------------------------------------------
# 0. 페이지 설정 & 전역 스타일
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="주가 추세 구간화 대시보드",
    page_icon="📈",
    layout="wide",
    menu_items={
        "Get Help": "mailto:youremail@example.com",
        "Report a bug": "mailto:youremail@example.com",
        "About": "주가 추세 구간화 알고리즘 데모 대시보드입니다."
    }
)

st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f8;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .st-emotion-cache-1r6slb0, .st-emotion-cache-ocqkz7 {
        background-color: white !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important;
        padding: 1.5rem !important;
    }
    .app-header {
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #e5e5ef;
        margin-bottom: 0.5rem;
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #777;
        margin: 0;
    }
    .search-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .search-subtitle {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.8rem;
    }
    .app-footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e5ef;
        font-size: 0.8rem;
        color: #999;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 1. 핀포인트(이벤트) 데이터
# ----------------------------------------------------------------------
pinpoints_df = pd.DataFrame({
    'Date': ['2024-06-05', '2024-10-10'],
    'Event': ['Vision Pro 발표', '신제품 출시'],
    'Content': ['Apple이 Vision Pro를 발표했습니다.', 'Apple이 새로운 제품을 출시했습니다.'],
    'Link': ['https://www.apple.com/newsroom/2024/06/apple-unveils-vision-pro-revolutionary-spatial-computing-platform/',
             'https://www.apple.com/newsroom/2024/10/apple-announces-new-products/']
})

# ----------------------------------------------------------------------
# 2. 데이터 로딩 함수 (캐싱 적용)
# ----------------------------------------------------------------------
@st.cache_data
def load_data(ticker, start_date, end_date):
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
# 3. 알고리즘 함수들
# ----------------------------------------------------------------------
def apply_smoothing_and_phase(df, window_length, polyorder):
    df = df.copy()
    if len(df) < window_length:
        st.warning("데이터가 스무딩 윈도우보다 적어 스무딩을 적용할 수 없습니다.")
        df["Smooth"] = df["Close"]
    else:
        df["Smooth"] = savgol_filter(df["Close"], window_length=window_length, polyorder=polyorder)
    df["Slope"] = np.gradient(df["Smooth"])
    classify = lambda s: "상승" if s > 0 else "하락"
    df["Phase"] = df["Slope"].apply(classify)
    return df

def apply_box_range(df, min_hits, window):
    df = df.copy()
    if df.empty:
        return df

    p_min, p_max = df['Close'].min(), df['Close'].max()
    limit = (p_max - p_min) / 25
    
    diffs = df['Close'].diff().abs()
    min_step = diffs[diffs > 0].min()
    if pd.isna(min_step):
        min_step = 10 
    
    exponent = int(math.floor(math.log10(min_step)))
    step = 10 ** exponent if exponent >= 1 else 10
    
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
                    df.loc[df.index[i:i+min_hits], "Phase"] = "박스권"
    
    if len(df) <= window:
        return df

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
    if "Phase" not in df.columns or df.empty:
        return df
        
    df["group_id"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    df["group_size"] = df.groupby("group_id")["Phase"].transform("size")
    
    unique_group_ids = df["group_id"].unique()
    if len(unique_group_ids) < 2:
        return df

    min_group_id = df["group_id"].min()
    max_group_id = df["group_id"].max()

    for group_id in unique_group_ids:
        mask = df["group_id"] == group_id
        size = df.loc[mask, "group_size"].iloc[0]
        
        if size <= min_days and group_id > min_group_id:
            if group_id == max_group_id:
                continue
                
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

def adjust_change_points(df, adjust_window):
    df = df.copy()
    if "Phase" not in df.columns or df.empty or len(df) < adjust_window:
        return df
        
    df["group_id"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    change_points = df.index[df["Phase"] != df["Phase"].shift()]
    
    if len(change_points) < 2:
        return df

    for cp in change_points:
        cp_idx = df.index.get_loc(cp)
        if cp_idx == 0:
            continue

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
            
            if local_min_pos > cp_idx:
                df.loc[df.index[cp_idx:local_min_pos], "Phase"] = prev_phase
            elif local_min_pos < cp_idx:
                df.loc[df.index[local_min_pos:cp_idx], "Phase"] = "상승"
                
        elif current_phase == "하락":
            local_max_idx = window_data["Close"].idxmax()
            local_max_pos = df.index.get_loc(local_max_idx)

            if local_max_pos > cp_idx:
                df.loc[df.index[cp_idx:local_max_pos], "Phase"] = prev_phase
            elif local_max_pos < cp_idx:
                df.loc[df.index[local_max_pos:cp_idx], "Phase"] = "하락"
    return df

def detect_market_phases(df, window_length, polyorder, min_days1, min_days2, adjust_window, min_hits, box_window):
    df_result = df.copy()
    df_result = apply_smoothing_and_phase(df_result, window_length, polyorder)
    df_result = apply_box_range(df_result, min_hits, box_window)
    df_result = merge_short_phases(df_result, min_days1)
    df_result = adjust_change_points(df_result, adjust_window)
    df_result = merge_short_phases(df_result, min_days2)
    return df_result

# ----------------------------------------------------------------------
# 5. 지표 / 시각화 함수들
# ----------------------------------------------------------------------
def display_metrics(df):
    if len(df) < 2:
        return
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    close_price = latest['Close']
    price_diff = close_price - prev['Close']
    pct_change = (price_diff / prev['Close']) * 100
    volume = latest['Volume']
    rsi = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().iloc[-1]
    high_52w = df['Close'][-250:].max() if len(df) > 250 else df['Close'].max()
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("현재 주가", f"{close_price:,.0f} 원", f"{price_diff:,.0f} 원 ({pct_change:+.2f}%)")
    with m2:
        st.metric("거래량", f"{volume:,.0f} 주")
    with m3:
        st.metric("RSI (14일)", f"{rsi:.2f}")
    with m4:
        st.metric("52주 최고가", f"{high_52w:,.0f} 원")
    st.divider()

def visualize_candlestick(df):
    df_reset = df.reset_index().rename(columns={'index': 'Date'})
    df_reset['Date_start'] = df_reset['Date'] - pd.Timedelta(hours=9)
    df_reset['Date_end']   = df_reset['Date'] + pd.Timedelta(hours=9)

    rule = alt.Chart(df_reset).mark_rule().encode(
        x=alt.X('Date:T', axis=alt.Axis(format='%Y-%m-%d', title='날짜')),
        y=alt.Y('Low:Q', scale=alt.Scale(zero=False), title='주가'),
        y2='High:Q',
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff"))
    )

    body = alt.Chart(df_reset).mark_rect().encode(
        x='Date_start:T',
        x2='Date_end:T',
        y='Open:Q',
        y2='Close:Q',
        color=alt.condition("datum.Open <= datum.Close", alt.value("#ff0000"), alt.value("#0000ff")),
        tooltip=['Date:T', 'Open', 'High', 'Low', 'Close', 'Volume']
    )

    chart = (rule + body).properties(
        height=300,
        title="일봉 캔들 차트"
    ).interactive()
    return chart

def visualize_technical_indicators(df):
    df = df.copy()
    if len(df) < 30:
        return alt.Chart(pd.DataFrame({'text': ['데이터가 부족하여 지표를 계산할 수 없습니다. (최소 30일 이상 필요)']})).mark_text(size=20).encode(text='text')

    indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_h'] = indicator_bb.bollinger_hband()
    df['bb_l'] = indicator_bb.bollinger_lband()
    
    indicator_macd = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = indicator_macd.macd()
    df['macd_signal'] = indicator_macd.macd_signal()
    df['macd_diff'] = indicator_macd.macd_diff()

    indicator_rsi = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df['rsi'] = indicator_rsi.rsi()

    df_reset = df.dropna().reset_index().rename(columns={'index': 'Date'})
    if df_reset.empty:
        return alt.Chart(pd.DataFrame({'text': ['유효한 데이터가 없습니다.']})).mark_text().encode(text='text')

    base = alt.Chart(df_reset).encode(x=alt.X('Date:T', axis=alt.Axis(title=None, format='%Y-%m-%d')))

    bb_line = base.mark_line(color='black', strokeWidth=1).encode(
        y=alt.Y('Close:Q', scale=alt.Scale(zero=False), title='주가')
    )
    bb_band = base.mark_area(opacity=0.2, color='gray').encode(
        y='bb_l:Q',
        y2='bb_h:Q'
    )
    chart_bb = (bb_line + bb_band).properties(height=250, title="볼린저 밴드 (가격 변동폭)")

    macd_line = base.mark_line(color='grey').encode(y='macd:Q')
    sig_line = base.mark_line(color='orange').encode(y='macd_signal:Q')
    hist_bar = base.mark_bar().encode(
        y=alt.Y('macd_diff:Q', title='MACD Diff'),
        color=alt.condition(alt.datum.macd_diff > 0, alt.value("#ff9999"), alt.value("#aaccff"))
    )
    chart_macd = (hist_bar + macd_line + sig_line).properties(height=150, title="MACD (추세 강도)")

    rsi_line = base.mark_line(color='purple').encode(
        y=alt.Y('rsi:Q', scale=alt.Scale(domain=[0, 100]), title='RSI')
    )
    rsi_rule_high = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(color='red', strokeDash=[3, 3]).encode(y='y')
    rsi_rule_low = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(color='blue', strokeDash=[3, 3]).encode(y='y')
    chart_rsi = (rsi_line + rsi_rule_high + rsi_rule_low).properties(height=150, title="RSI (과열/침체)")

    return alt.vconcat(chart_bb, chart_macd, chart_rsi).resolve_scale(x='shared').interactive()

def visualize_return_analysis(df):
    df = df.copy()
    df['Daily_Ret'] = df['Close'].pct_change()
    df['Cum_Ret'] = (1 + df['Daily_Ret']).cumprod() - 1
    df_reset = df.dropna().reset_index().rename(columns={'index': 'Date'})

    cum_chart = alt.Chart(df_reset).mark_area(
        line={'color': 'darkgreen'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0),
                   alt.GradientStop(color='darkgreen', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('Date:T', title='날짜'),
        y=alt.Y('Cum_Ret:Q', title='누적 수익률', axis=alt.Axis(format='%')),
        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'),
                 alt.Tooltip('Cum_Ret:Q', format='.2%')]
    ).properties(height=300, title="누적 수익률 추이 (Cumulative Return)").interactive()

    hist_chart = alt.Chart(df_reset).mark_bar().encode(
        x=alt.X('Daily_Ret:Q', bin=alt.Bin(maxbins=50), title='일별 등락률'),
        y=alt.Y('count()', title='빈도수'),
        color=alt.value('purple')
    ).properties(height=200, title="일별 등락률 분포 (Histogram)")

    return alt.vconcat(cum_chart, hist_chart)

def visualize_phases_altair_all_interactions(df, pinpoints_df=None):
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_text().properties(title="데이터가 없습니다.")
    df_reset = df.reset_index().rename(columns={'index': 'Date'})

    min_price = df_reset['Close'].min()
    max_price = df_reset['Close'].max()
    price_range = max_price - min_price
    target_y_value = min_price + (price_range * 0.001)

    background = alt.Chart(pd.DataFrame()).mark_text()
    phase_blocks_empty = True

    if "Phase" in df_reset.columns and not df_reset['Phase'].isnull().all():
        df_phases = df_reset[['Date', 'Phase']].copy()
        df_phases['Phase'] = df_phases['Phase'].fillna('N/A')
        df_phases['New_Block'] = df_phases['Phase'] != df_phases['Phase'].shift(1)
        df_phases['Block_ID'] = df_phases['New_Block'].cumsum()
        
        phase_blocks = df_phases.groupby('Block_ID').agg(
            start_date=('Date', 'min'),
            end_date=('Date', 'max'),
            Phase=('Phase', 'first')
        ).reset_index()
        phase_blocks = phase_blocks[phase_blocks['Phase'] != 'N/A']
        
        if not phase_blocks.empty:
            phase_blocks_empty = False
            domain = ['상승', '하락', '박스권']
            range_ = ['#ff9999', '#aaccff', '#d9d9d9']
            background = alt.Chart(phase_blocks).mark_rect(opacity=0.5).encode(
                x=alt.X('start_date:T', title='날짜'),
                x2=alt.X2('end_date:T'),
                color=alt.Color('Phase:N',
                                scale=alt.Scale(domain=domain, range=range_),
                                legend=alt.Legend(title='추세 구간')),
                tooltip=['start_date:T', 'end_date:T', 'Phase:N']
            )

    line_chart = alt.Chart(df_reset).mark_line(color='gray').encode(
        x=alt.X('Date:T', title='날짜'),
        y=alt.Y('Close:Q', title='가격', scale=alt.Scale(zero=False)),
        tooltip=['Date:T', 'Close:Q']
    )

    hover_selection = alt.selection_point(on='mouseover', empty='all', fields=['Date'])
    pinpoint_layer = alt.Chart(pd.DataFrame()).mark_text()

    if pinpoints_df is not None and not pinpoints_df.empty:
        pinpoints_df_copy = pinpoints_df.copy()
        pinpoints_df_copy['Date'] = pd.to_datetime(pinpoints_df_copy['Date'])
        merged_pins = pd.merge(df_reset[['Date', 'Close']], pinpoints_df_copy, on='Date', how='inner')

        if not merged_pins.empty:
            rule = alt.Chart(merged_pins).mark_rule(
                color='black', strokeDash=[3, 3]
            ).encode(x='Date:T')

            points = alt.Chart(merged_pins).mark_point(
                filled=True,
                stroke='black',
                strokeWidth=0.5
            ).transform_calculate(
                pin_y_position=f"{target_y_value}"
            ).encode(
                x='Date:T',
                y=alt.Y('pin_y_position:Q', title='가격'),
                tooltip=[
                    alt.Tooltip('Date:T', title='날짜', format='%Y-%m-%d'),
                    alt.Tooltip('Event:N', title='이벤트')
                ],
                size=alt.condition(hover_selection, alt.value(200), alt.value(100))
            ).add_params(hover_selection)
            pinpoint_layer = rule + points

    if phase_blocks_empty:
        base_chart = line_chart
    else:
        base_chart = background + line_chart

    target_y_df = pd.DataFrame({'target_y': [target_y_value]})
    base_line = alt.Chart(target_y_df).mark_rule(color='black', opacity=0).encode(y='target_y:Q')

    main_chart = (base_chart + pinpoint_layer + base_line).properties(height=400)
    return main_chart

# ----------------------------------------------------------------------
# 6. 상단 헤더
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">따라가기 힘든 금융 정보,</div>
        <div class="app-title">이 대시보드로 한 발 앞서가세요!</div>
        <p class="app-subtitle">
            종목을 입력하면 기본 시세 · AI 추세 분석 · 기술적 지표 · 수익률 분석을 한 번에 확인할 수 있습니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 7. 세션 상태 초기값
# ----------------------------------------------------------------------
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "005930"  # 기본 삼성전자
if "focus_mode" not in st.session_state:
    # True = 분석 화면만 보기 (인기 종목/뉴스 숨김)
    st.session_state.focus_mode = False
if "selected_news_idx" not in st.session_state:
    st.session_state.selected_news_idx = None

# ----------------------------------------------------------------------
# 8. 중간 검색 영역 (찾으시는 종목)
# ----------------------------------------------------------------------
center_cols = st.columns([1, 2, 1])
with center_cols[1]:
    st.markdown('<div class="search-title">찾으시는 종목을 검색해 보세요</div>', unsafe_allow_html=True)
    st.markdown('<div class="search-subtitle">종목 코드를 입력하면 아래에서 분석 기능을 선택할 수 있습니다.</div>', unsafe_allow_html=True)
    
    ticker_input = st.text_input(
        "종목 코드 입력",
        value=st.session_state.selected_ticker,
        label_visibility="collapsed",
        placeholder="예: 005930",
    )
    search_col1, search_col2 = st.columns([3, 1])
    with search_col2:
        if st.button("조회", use_container_width=True):
            st.session_state.selected_ticker = ticker_input.strip()
            st.session_state.focus_mode = True      # 조회하면 분석 집중 모드 ON
            st.rerun()

    with st.expander("📈 분석 기간 및 알고리즘 설정", expanded=False):
        start_date = st.date_input("시작일", pd.to_datetime("2024-01-01"))
        end_date = st.date_input("종료일", pd.to_datetime("2024-12-31"))
        window_length = st.number_input("스무딩 윈도우 (홀수)", 3, 21, 5, 2)
        polyorder = st.slider("스무딩 다항식 차수", 1, 5, 3)
        min_days1 = st.slider("초기 짧은 구간 병합 일수", 1, 10, 2)
        min_days2 = st.slider("최종 짧은 구간 병합 일수", 1, 10, 2)
        adjust_window = st.slider("전환점 보정 윈도우", 1, 10, 2)
        min_hits = st.slider("박스권 최소 교차 횟수", 1, 20, 9)
        box_window = st.slider("박스권 판정 윈도우", 1, 20, 10)

ticker = st.session_state.selected_ticker

# ----------------------------------------------------------------------
# 9. 인기 종목 / 분석 영역 레이아웃
# ----------------------------------------------------------------------
POPULAR_STOCKS = [
    ("삼성전자", "005930"),
    ("셀트리온", "068270"),
    ("HMM", "011200"),
]

main_container = st.container(border=True)

with main_container:
    # focus_mode 이면 분석 화면만 전체 폭으로 보여줌
    if st.session_state.focus_mode:
        # 상단에 돌아가기 버튼
        back_col, _ = st.columns([1, 3])
        with back_col:
            if st.button("← 인기 종목/뉴스 다시 보기"):
                st.session_state.focus_mode = False
                st.rerun()

        # 전체 폭 분석 영역
        if ticker:
            df_raw = load_data(ticker, start_date, end_date)
            if df_raw is not None and not df_raw.empty:
                st.markdown(f"#### 선택 종목 : `{ticker}`")
                display_metrics(df_raw)

                feature = st.radio(
                    "분석 기능을 선택하세요",
                    ["📈 기본 시세", "🧠 AI 추세 분석", "📐 기술적 지표", "📊 수익률 분석"],
                    horizontal=True,
                )

                if feature == "📈 기본 시세":
                    candle_chart = visualize_candlestick(df_raw)
                    st.altair_chart(candle_chart, use_container_width=True)
                    st.subheader("일별 시세 데이터")
                    st.dataframe(df_raw.sort_index(ascending=False).head(10),
                                 use_container_width=True)

                elif feature == "🧠 AI 추세 분석":
                    if len(df_raw) < window_length:
                        st.warning(f"데이터 부족: 최소 {window_length}일 이상 필요합니다.")
                    else:
                        with st.spinner("추세 패턴 분석 중..."):
                            df_processed = detect_market_phases(
                                df_raw, window_length, polyorder,
                                min_days1, min_days2,
                                adjust_window, min_hits, box_window
                            )
                        fig = visualize_phases_altair_all_interactions(
                            df_processed, pinpoints_df=pinpoints_df
                        )
                        st.altair_chart(fig, use_container_width=True)

                        if "Phase" in df_processed.columns:
                            counts = df_processed['Phase'].value_counts()
                            st.markdown("#### 추세 분포 요약")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("상승 구간", f"{counts.get('상승', 0)}일")
                            c2.metric("하락 구간", f"{counts.get('하락', 0)}일")
                            c3.metric("박스권", f"{counts.get('박스권', 0)}일")

                        st.subheader("뉴스 이벤트 매칭")
                        st.dataframe(pinpoints_df, use_container_width=True, hide_index=True)

                elif feature == "📐 기술적 지표":
                    st.subheader("📐 기술적 지표 분석")
                    st.info("""
                    **💡 초보자용 요약**
                    * **볼린저 밴드:** 회색 띠 바깥으로 나가면 다시 안으로 들어오는 성향
                    * **MACD:** 빨간 막대↑ = 상승세 강화, 파란 막대↑ = 하락세 강화
                    * **RSI:** 70 이상 과열, 30 이하 침체
                    """)
                    tech_chart = visualize_technical_indicators(df_raw)
                    st.altair_chart(tech_chart, use_container_width=True)

                elif feature == "📊 수익률 분석":
                    st.subheader("📊 수익률 퍼포먼스")
                    st.caption("이 기간 동안 보유했을 때의 누적 수익률과 일별 등락률 분포입니다.")
                    return_chart = visualize_return_analysis(df_raw)
                    st.altair_chart(return_chart, use_container_width=True)
        else:
            st.info("위에서 종목 코드를 입력 후 조회를 눌러주세요.")

    else:
        # ------- 탐색 모드: 분석 + 오른쪽 인기 종목 -------
        left_col, right_col = st.columns([3, 1])

        # 왼쪽: 분석 (구조 동일)
        with left_col:
            if ticker:
                df_raw = load_data(ticker, start_date, end_date)
                if df_raw is not None and not df_raw.empty:
                    st.markdown(f"#### 선택 종목 : `{ticker}`")
                    display_metrics(df_raw)

                    feature = st.radio(
                        "분석 기능을 선택하세요",
                        ["📈 기본 시세", "🧠 AI 추세 분석", "📐 기술적 지표", "📊 수익률 분석"],
                        horizontal=True,
                    )

                    if feature == "📈 기본 시세":
                        candle_chart = visualize_candlestick(df_raw)
                        st.altair_chart(candle_chart, use_container_width=True)
                        st.subheader("일별 시세 데이터")
                        st.dataframe(df_raw.sort_index(ascending=False).head(10),
                                     use_container_width=True)

                    elif feature == "🧠 AI 추세 분석":
                        if len(df_raw) < window_length:
                            st.warning(f"데이터 부족: 최소 {window_length}일 이상 필요합니다.")
                        else:
                            with st.spinner("추세 패턴 분석 중..."):
                                df_processed = detect_market_phases(
                                    df_raw, window_length, polyorder,
                                    min_days1, min_days2,
                                    adjust_window, min_hits, box_window
                                )
                            fig = visualize_phases_altair_all_interactions(
                                df_processed, pinpoints_df=pinpoints_df
                            )
                            st.altair_chart(fig, use_container_width=True)

                            if "Phase" in df_processed.columns:
                                counts = df_processed['Phase'].value_counts()
                                st.markdown("#### 추세 분포 요약")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("상승 구간", f"{counts.get('상승', 0)}일")
                                c2.metric("하락 구간", f"{counts.get('하락', 0)}일")
                                c3.metric("박스권", f"{counts.get('박스권', 0)}일")

                            st.subheader("뉴스 이벤트 매칭")
                            st.dataframe(pinpoints_df, use_container_width=True, hide_index=True)

                    elif feature == "📐 기술적 지표":
                        st.subheader("📐 기술적 지표 분석")
                        st.info("""
                        **💡 초보자용 요약**
                        * **볼린저 밴드:** 회색 띠 바깥으로 나가면 다시 안으로 들어오는 성향
                        * **MACD:** 빨간 막대↑ = 상승세 강화, 파란 막대↑ = 하락세 강화
                        * **RSI:** 70 이상 과열, 30 이하 침체
                        """)
                        tech_chart = visualize_technical_indicators(df_raw)
                        st.altair_chart(tech_chart, use_container_width=True)

                    elif feature == "📊 수익률 분석":
                        st.subheader("📊 수익률 퍼포먼스")
                        st.caption("이 기간 동안 보유했을 때의 누적 수익률과 일별 등락률 분포입니다.")
                        return_chart = visualize_return_analysis(df_raw)
                        st.altair_chart(return_chart, use_container_width=True)
            else:
                st.info("위에서 종목 코드를 입력 후 조회를 눌러주세요.")

        # 오른쪽: 인기 종목
        with right_col:
            st.markdown("####  인기 종목")
            st.caption("클릭하면 해당 종목으로 이동합니다.")
            for i, (name, code) in enumerate(POPULAR_STOCKS, start=1):
                if st.button(f"{i}. {name} ({code})", key=f"popular-{code}", use_container_width=True):
                    st.session_state.selected_ticker = code
                    st.session_state.focus_mode = True   # 인기 종목 눌러도 분석 집중 모드로
                    st.rerun()

# ----------------------------------------------------------------------
# 10. 많이 본 뉴스 영역 (focus_mode 일 때는 숨김)
# ----------------------------------------------------------------------
POPULAR_NEWS = [
    {
        "title": "예시) 삼성전자 실적 발표, 시장 기대 상회",
        "content": "여기에 뉴스 전문 또는 요약이 들어갑니다. 나중에 실제 데이터로 교체하세요."
    },
    {
        "title": "예시) 셀트리온, 글로벌 임상 승인 소식",
        "content": "두 번째 뉴스 예시입니다. 실제 뉴스 데이터 연동 예정."
    },
    {
        "title": "예시) HMM, 운임 지수 상승에 따른 수혜 전망",
        "content": "세 번째 뉴스 예시입니다."
    },
]

if not st.session_state.focus_mode:
    st.markdown("### 많이 본 뉴스")
    news_left, news_right = st.columns([2, 3])
    with news_left:
        st.caption("최근 일주일 기준 (예시)")
        for idx, item in enumerate(POPULAR_NEWS):
            if st.button(f"{idx+1}. {item['title']}", key=f"news-{idx}", use_container_width=True):
                st.session_state.selected_news_idx = idx

    with news_right:
        if st.session_state.selected_news_idx is not None:
            item = POPULAR_NEWS[st.session_state.selected_news_idx]
            st.subheader(item["title"])
            st.write(item["content"])
        else:
            st.info("왼쪽에서 뉴스를 클릭하면 여기 내용이 표시됩니다. (추후 실제 뉴스 데이터로 교체)")

# ----------------------------------------------------------------------
# [NEW] 12. AI 주식 상담 챗봇 (Google Gemini - 자동 키 감지)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.header("🤖 Gemini 주식 비서 (Free)")

    # [수정됨] Secrets에서 키를 먼저 찾고, 없으면 입력창 띄우기
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API 키가 연동되었습니다! ✅")
    else:
        api_key = st.text_input("Google API Key를 입력하세요", type="password")
        if not api_key:
            st.info("API 키를 입력하거나, Secrets에 설정하면 자동으로 연동됩니다.")
            st.markdown("[👉 키 발급받으러 가기](https://aistudio.google.com/app/apikey)")
    
    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 저는 구글 Gemini입니다. 주식에 대해 물어보세요! 🌕"}
        ]

    # 채팅 메시지 출력
    # (키가 있을 때만 채팅창 활성화)
    if api_key:
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant", avatar="🤖").write(msg["content"])

        # 사용자 입력 처리
        if prompt := st.chat_input("질문을 입력하세요... (예: RSI가 뭐야?)"):
            # 1. 설정 (매번 호출 시 설정)
            genai.configure(api_key=api_key)
            
            # 2. 사용자 메시지 저장
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            try:
                with st.spinner("Gemini가 분석 중입니다..."):
                    # 모델 설정 (에러 방지를 위해 안전한 모델명 사용 권장)
                    # 만약 1.5-flash가 계속 안 되면 'gemini-pro'로 바꿔보세요.
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    response = model.generate_content(prompt)
                    ai_msg = response.text
                    
                    # AI 응답 저장
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
                    st.chat_message("assistant", avatar="🤖").write(ai_msg)
                    
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                
# ----------------------------------------------------------------------
# 11. 푸터
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        본 서비스는 학습·연구용 데모이며, 실제 투자 의사결정에 사용하기 전 반드시 별도의 검증이 필요합니다.
    </div>
    """,
    unsafe_allow_html=True,
)
