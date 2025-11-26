import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
import altair as alt
import ta
import random
from datetime import datetime
import google.generativeai as genai
import uuid # For generating unique chat IDs

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

    .app-header {
        padding: 0.6rem 0 1.0rem 0;
        border-bottom: 1px solid #e5e5ef;
        margin-bottom: 0.8rem;
    }

    .app-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        line-height: 1.3;
    }

    .app-subtitle {
        font-size: 0.85rem;
        color: #777;
        margin: 0.3rem 0 0 0;
        line-height: 1.4;
    }

    .app-footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e5ef;
        font-size: 0.8rem;
        color: #999;
        text-align: center;
    }
    .stChatInputContainer {
        border-top: 1px solid #ccc;
    }
    /* Chat history button style */
    .chat-btn {
        width: 100%;
        text-align: left;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.25rem;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .chat-btn:hover {
        background-color: #f0f0f0;
    }
    .chat-btn-active {
        background-color: #e6f7ff; /* light blue */
        border: 1px solid #91d5ff;
        font-weight: 600;
    }
    .chat-btn-title {
        font-size: 0.85rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .chat-btn-category {
        font-size: 0.7rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 1. 핀포인트(이벤트) 데이터 (임시)
# ----------------------------------------------------------------------
pinpoints_df = pd.DataFrame({
    'Date': ['2024-06-05', '2024-10-10'],
    'Event': ['Vision Pro 발표', '신제품 출시'],
    'Content': ['Apple이 Vision Pro를 발표했습니다.', 'Apple이 새로운 제품을 출시했습니다.'],
    'Link': ['https://www.apple.com/newsroom/2024/06/apple-unveils-vision-pro-revolutionary-spatial-computing-platform/',
             'https://www.apple.com/newsroom/2024/10/apple-announces-new-products/']
})

# ----------------------------------------------------------------------
# 2. 인기 종목 풀 리스트 (총 20개)
# ----------------------------------------------------------------------
POPULAR_STOCKS_ALL = [
    # 기존 3개
    {"code": "005930", "name": "삼성전자"},
    {"code": "068270", "name": "셀트리온"},
    {"code": "011200", "name": "HMM"},

    # 미국 나스닥 상위 (예시)
    {"code": "NVDA", "name": "NVIDIA"},
    {"code": "AAPL", "name": "애플"},
    {"code": "MSFT", "name": "마이크로소프트"},
    {"code": "AMZN", "name": "아마존"},
    {"code": "GOOGL", "name": "알파벳 A"},
    {"code": "GOOG", "name": "알파벳 C"},
    {"code": "AVGO", "name": "브로드컴"},
    {"code": "META", "name": "메타 플랫폼스"},
    {"code": "TSLA", "name": "테슬라"},
    {"code": "NFLX", "name": "넷플릭스"},

    # 국내 시총 상위 (삼성전자 제외)
    {"code": "000660", "name": "SK하이닉스"},
    {"code": "373220", "name": "LG에너지솔루션"},
    {"code": "207940", "name": "삼성바이오로직스"},
    {"code": "005380", "name": "현대자동차"},
    {"code": "329180", "name": "HD현대중공업"},
    {"code": "034020", "name": "두산에너빌리티"},
    {"code": "012450", "name": "한화에어로스페이스"},
]

CHAT_CATEGORIES = ["기술적 분석", "기본적 분석", "시장 뉴스", "투자 심리", "기타"]

# ----------------------------------------------------------------------
# 3. 세션 상태 초기화 (CHAT HISTORY 추가)
# ----------------------------------------------------------------------
if "page_mode" not in st.session_state:
    st.session_state.page_mode = "HOME"  # HOME 또는 DETAIL

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = ""

if "popular_sample" not in st.session_state:
    st.session_state.popular_sample = random.sample(POPULAR_STOCKS_ALL, 5)

if "popular_refresh_time" not in st.session_state:
    st.session_state.popular_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# [추가] 챗봇 세션 관리 상태 초기화
if "chat_sessions" not in st.session_state:
    # Key: UUID (Session ID)
    # Value: {'title': str, 'category': str, 'messages': list, 'created_at': datetime}
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
# [기존 메시지 대신 세션 사용]
if "messages" in st.session_state:
    del st.session_state.messages 


# ----------------------------------------------------------------------
# 4. 데이터 로딩 함수 (캐싱 적용)
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
# 5. 알고리즘 함수들 (기존 코드 유지)
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

            prev_group_mask = df["group_id"] == group_id - 1
            if not prev_group_mask.empty:
                prev_phase = df.loc[prev_group_mask, "Phase"].iloc[0]
            else:
                prev_phase = None
            
            next_group_mask = df["group_id"] == group_id + 1
            if not next_group_mask.empty:
                next_phase = df.loc[next_group_mask, "Phase"].iloc[0]
            else:
                next_phase = None

            if prev_phase and prev_phase != '박스권':
                df.loc[mask, "Phase"] = prev_phase
            elif next_phase and next_phase != '박스권':
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
# 6. 시각화 / 지표 함수들 (기존 코드 유지)
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
    df_reset['Date_end'] = df_reset['Date'] + pd.Timedelta(hours=9)

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
    rsi_rule_high = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(
        color='red', strokeDash=[3, 3]
    ).encode(y='y')
    rsi_rule_low = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(
        color='blue', strokeDash=[3, 3]
    ).encode(y='y')
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
            stops=[
                alt.GradientStop(color='white', offset=0),
                alt.GradientStop(color='darkgreen', offset=1)
            ],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('Date:T', title='날짜'),
        y=alt.Y('Cum_Ret:Q', title='누적 수익률', axis=alt.Axis(format='%')),
        tooltip=[
            alt.Tooltip('Date:T', format='%Y-%m-%d'),
            alt.Tooltip('Cum_Ret:Q', format='.2%')
        ]
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
                color=alt.Color(
                    'Phase:N',
                    scale=alt.Scale(domain=domain, range=range_),
                    legend=alt.Legend(title='추세 구간')
                ),
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
        merged_pins = pd.merge(
            df_reset[['Date', 'Close']], pinpoints_df_copy, on='Date', how='inner'
        )

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
                size=alt.condition(
                    hover_selection,
                    alt.value(200),
                    alt.value(100)
                )
            ).add_params(hover_selection)
            pinpoint_layer = rule + points

    if phase_blocks_empty:
        base_chart = line_chart
    else:
        base_chart = background + line_chart

    target_y_df = pd.DataFrame({'target_y': [target_y_value]})
    base_line = alt.Chart(target_y_df).mark_rule(
        color='black', opacity=0
    ).encode(y='target_y:Q')

    main_chart = (base_chart + pinpoint_layer + base_line).properties(height=400)
    return main_chart

# ----------------------------------------------------------------------
# 7. 상단 헤더 (문구 2줄)
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div style="display:flex; flex-direction:column; gap:0.1rem;">
            <div class="app-title">따라가기 힘든 금융 정보,</div>
            <div class="app-title">📈 투자위키로 한 발 앞서가세요!</div>
        
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 8. 홈 화면 렌더 함수 (기존 코드 유지)
# ----------------------------------------------------------------------
def render_home():
    # 왼쪽: 찾는 종목 / 가운데 여백 / 오른쪽: 인기종목
    left_col, spacer_col, mid_col = st.columns([2.4, 0.5, 1.6])

    # ----- 왼쪽: 찾는 종목 -----
    with left_col:
        st.subheader("🔍 찾는 종목")
        search_input = st.text_input(
            "종목 코드 / 티커를 입력하세요",
            value=st.session_state.selected_ticker,
            placeholder="예: 005930 (삼성전자), AAPL (Apple)",
            key="search_input_home",
        )
        search_btn = st.button("이 종목 분석하기", type="primary")

        if search_btn and search_input.strip():
            st.session_state.selected_ticker = search_input.strip()
            st.session_state.page_mode = "DETAIL"

    # spacer_col 은 비워둬서 공백만 생성
    with spacer_col:
        st.write("")

    # ----- 오른쪽: 인기종목 (조금 더 오른쪽으로 이동된 느낌) -----
    with mid_col:
        header_col, btn_col, time_col = st.columns([1.4, 0.4, 1.2])

        with header_col:
            # 줄바꿈 방지 + 한 줄로 보이게
            st.markdown(
                "<h4 style='margin-bottom:0.2rem; white-space:nowrap;'>🔥 인기종목</h4>",
                unsafe_allow_html=True,
            )

        with btn_col:
            if st.button("⟳", help="인기종목 리스트 새로고침"):
                st.session_state.popular_sample = random.sample(POPULAR_STOCKS_ALL, 5)
                st.session_state.popular_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with time_col:
            st.markdown(
                f"""
                <p style="font-size:0.70rem; color:#999; margin-top:1.1rem; text-align:right;">
                    마지막 새로고침: {st.session_state.popular_refresh_time}
                </p>
                """,
                unsafe_allow_html=True,
            )

        # 인기종목 리스트: 종목명 (코드)
        for stock in st.session_state.popular_sample:
            code = stock["code"]
            name = stock["name"]
            if st.button(f"{name} ({code})", key=f"popular_btn_{code}", use_container_width=True):
                st.session_state.selected_ticker = code
                st.session_state.page_mode = "DETAIL"

    st.markdown("---")

    # ----- 아래: 많이 본 뉴스 (예시) -----
    st.subheader("📰 많이 본 뉴스")
    st.caption("※ 현재는 예시입니다. 나중에 실제 리포트/뉴스 데이터를 연결하면 됩니다.")

    example_news = [
        {"title": "[예시] 삼성전자 AI 반도체 수요 급증 리포트", "source": "뉴스1", "date": "2025-11-20"},
        {"title": "[예시] 미국 나스닥 기술주 조정, 향후 전망은?", "source": "연합뉴스", "date": "2025-11-18"},
        {"title": "[예시] 방산·조선주 강세, 한화에어로스페이스·HD현대중공업 급등", "source": "매일경제", "date": "2025-11-15"},
    ]

    for i, news in enumerate(example_news, start=1):
        with st.expander(f"{i}. {news['title']}"):
            st.write(f"출처: {news['source']}")
            st.write(f"날짜: {news['date']}")
            st.info("👉 이 영역에 실제 뉴스 본문 또는 링크를 나중에 넣으면 됩니다.")

# ----------------------------------------------------------------------
# 9. 상세 분석 화면 렌더 함수 (기존 코드 유지)
# ----------------------------------------------------------------------
def render_detail():
    ticker = st.session_state.selected_ticker

    top_cols = st.columns([1, 3])
    with top_cols[0]:
        if st.button("← 홈으로 돌아가기"):
            st.session_state.page_mode = "HOME"
            st.rerun() # Ensure navigation works instantly
    with top_cols[1]:
        st.markdown(f"### 📊 {ticker} 상세 분석")

    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.markdown("#### ⚙️ 분석 설정")

        start_date = st.date_input("시작일", pd.to_datetime("2024-01-01"))
        end_date = st.date_input("종료일", pd.to_datetime("2024-12-31"))

        st.markdown("##### 구간화 파라미터")
        window_length = st.number_input(
            "스무딩 윈도우 (홀수)", min_value=3, max_value=21, value=5, step=2
        )
        polyorder = st.slider("스무딩 다항식 차수", 1, 5, 3)
        min_days1 = st.slider("초기 짧은 구간 병합 일수", 1, 10, 2)
        min_days2 = st.slider("최종 짧은 구간 병합 일수", 1, 10, 2)
        adjust_window = st.slider("전환점 보정 윈도우", 1, 10, 2)
        min_hits = st.slider("박스권 최소 교차 횟수", 1, 20, 9)
        box_window = st.slider("박스권 판정 윈도우", 1, 20, 10)

    with right_col:
        df_raw = load_data(ticker, start_date, end_date)
        if df_raw is None or df_raw.empty:
            st.warning("데이터를 불러올 수 없습니다. 종목 코드/티커와 기간을 다시 확인해 주세요.")
            return

        display_metrics(df_raw)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 기본 시세", "🧠 AI 추세 분석", "📐 기술적 지표", "📊 수익률 분석"]
        )

        with tab1:
            candle_chart = visualize_candlestick(df_raw)
            st.altair_chart(candle_chart, use_container_width=True)
            st.subheader("일별 시세 데이터")
            st.dataframe(
                df_raw.sort_index(ascending=False).head(10),
                use_container_width=True
            )

        with tab2:
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

        with tab3:
            st.subheader("📐 기술적 지표 분석")
            st.info("""
            **💡 초보자를 위한 1분 요약**
            * **볼린저 밴드:** 주가가 회색 띠를 벗어나면 다시 돌아오려는 성질이 있어요. (밴드 상단=비쌈, 하단=쌈)
            * **MACD:** 빨간 막대가 커지면 '상승세', 파란 막대가 커지면 '하락세'입니다.
            * **RSI:** 70을 넘으면 '과열(비쌈)', 30 밑이면 '침체(쌈)' 신호입니다.
            """)
            tech_chart = visualize_technical_indicators(df_raw)
            st.altair_chart(tech_chart, use_container_width=True)

            with st.expander("📚 지표 상세 해석 가이드 (눌러서 보기)"):
                st.markdown("""
                ### 1. 볼린저 밴드 (Bollinger Bands)
                - **무엇인가요?** 주가가 다니는 '길'이라고 생각하세요. 
                - **해석법:** 주가는 보통 밴드 안에서 움직입니다. 
                    - 캔들이 **위쪽 선**을 치면? 단기 고점일 수 있습니다. (매도 고려)
                    - 캔들이 **아래쪽 선**을 치면? 단기 저점일 수 있습니다. (매수 고려)
                    
                ### 2. MACD (추세)
                - **무엇인가요?** 주가의 '방향'과 '에너지'를 보여줍니다.
                - **해석법:**
                    - **빨간 막대**가 점점 길어지면 상승 힘이 강해지는 것입니다.
                    - **파란 막대**가 줄어들면서 빨간색으로 바뀌려는 순간이 '매수 타이밍'으로 불립니다.
                    
                ### 3. RSI (상대강도지수)
                - **무엇인가요?** 시장의 '과열' 여부를 0~100 점수로 매긴 것입니다.
                - **해석법:**
                    - **70 이상 (점선 위):** "너무 뜨겁다!" 사람들이 너무 많이 사서 비싼 상태일 수 있습니다. (조심!)
                    - **30 이하 (점선 아래):** "너무 차갑다!" 사람들이 너무 많이 팔아서 싼 상태일 수 있습니다. (기회?)
                """)

        with tab4:
            st.subheader("📊 수익률 퍼포먼스")
            st.caption("이 기간 동안 보유했을 때의 누적 수익률과 변동성입니다.")
            return_chart = visualize_return_analysis(df_raw)
            st.altair_chart(return_chart, use_container_width=True)

# ----------------------------------------------------------------------
# 10. 라우팅 (HOME / DETAIL)
# ----------------------------------------------------------------------
if st.session_state.page_mode == "DETAIL" and st.session_state.selected_ticker:
    render_detail()
else:
    st.session_state.page_mode = "HOME"
    render_home()


# ----------------------------------------------------------------------
# 12. AI 주식 상담 챗봇 (Google Gemini - History/Category Logic Added)
# ----------------------------------------------------------------------

# 헬퍼 함수
def _create_new_chat(title, category):
    """새로운 채팅 세션을 생성하고 현재 세션으로 설정합니다."""
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {
        'title': title,
        'category': category,
        # 초기 메시지는 환영 메시지로 설정
        'messages': [{"role": "assistant", "content": "안녕하세요! 저는 구글 Gemini입니다. 주식에 대해 물어보세요! 🌕"}],
        'created_at': datetime.now()
    }
    st.session_state.current_session_id = new_id
    st.session_state.new_chat_title = "" # clear input
    st.rerun()

def _load_chat(session_id):
    """선택된 채팅 세션을 현재 세션으로 로드합니다."""
    st.session_state.current_session_id = session_id
    st.rerun()

# --- 사이드바 시작 ---
with st.sidebar:
    st.markdown("---")
    st.header("🤖 Gemini 주식 비서")

    # [수정됨] API 키 연동
    api_key = ""
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API 키가 연동되었습니다! ✅")
    else:
        # Canvas 환경에서 st.text_input을 사용해 키를 받도록 처리
        key_input = st.text_input("Google API Key를 입력하세요", type="password", key="sidebar_api_key_input")
        if key_input:
            api_key = key_input
            st.session_state['api_key_set'] = True # For rerunning only when key is set
        if not api_key:
            st.info("API 키를 입력하거나, Secrets에 설정하면 자동으로 연동됩니다.")
            st.markdown("[👉 키 발급받으러 가기](https://aistudio.google.com/app/apikey)")

    if not api_key:
        st.warning("API 키가 설정되지 않아 챗봇 기능을 사용할 수 없습니다.")
    else:
        # --- 챗봇 히스토리 관리 UI ---
        st.markdown("#### 📁 대화 기록 관리")
        
        # 1. 새 대화 만들기 폼
        with st.expander("➕ 새 대화 시작"):
            # 새 대화 제목과 카테고리 입력
            new_title = st.text_input(
                "대화 제목", 
                value=st.session_state.get('new_chat_title', ''),
                key="new_chat_title_input", 
                placeholder="예: 삼성전자 기술적 분석"
            )
            new_category = st.selectbox(
                "카테고리", 
                options=CHAT_CATEGORIES, 
                key="new_chat_category_select"
            )
            if st.button("새 대화 시작하기", use_container_width=True, type="primary"):
                if new_title.strip():
                    _create_new_chat(new_title.strip(), new_category)
                else:
                    st.error("제목을 입력해 주세요.")
        
        st.markdown("##### 저장된 대화")
        
        # 2. 대화 목록 표시
        if st.session_state.chat_sessions:
            # 최신순으로 정렬
            sorted_sessions = sorted(
                st.session_state.chat_sessions.items(), 
                key=lambda item: item[1]['created_at'], 
                reverse=True
            )
            
            for session_id, session_data in sorted_sessions:
                is_active = session_id == st.session_state.current_session_id
                
                # HTML을 사용하여 클릭 가능한 버튼처럼 만듦
                btn_class = "chat-btn chat-btn-active" if is_active else "chat-btn"
                btn_style = "background-color: #e6f7ff;" if is_active else ""
                
                # Streamlit 버튼을 사용하여 세션 로드 (HTML 버튼은 시각적인 역할)
                if st.button(
                    f"🏷️ {session_data['title']} \n\n <span style='font-size: 0.7rem; color: #666;'>{session_data['category']} | {session_data['created_at'].strftime('%m-%d %H:%M')}</span>",
                    key=f"chat_load_{session_id}", 
                    on_click=_load_chat, 
                    args=(session_id,), 
                    help=f"대화 불러오기: {session_data['title']}", 
                    use_container_width=True
                ):
                    pass # on_click 핸들러가 rerunning을 유발하여 세션을 로드

        else:
            st.info("아직 저장된 대화가 없습니다. 새 대화를 시작해 보세요!")


        # --- 현재 채팅창 및 입력 ---
        st.markdown("---")
        
        # 현재 세션 메시지 로드
        if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.chat_sessions:
            current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
            st.subheader(f"대화: {current_session['title']}")
            current_messages = current_session['messages']
        else:
            # 현재 세션이 없거나 초기 상태인 경우, 새 임시 세션을 보여줍니다.
            st.subheader("대화: 새 대화")
            current_messages = [{"role": "assistant", "content": "새 대화를 시작하거나 기존 대화를 불러오세요. 👆"}]


        # 채팅 메시지 출력
        chat_container = st.container()
        with chat_container:
            for msg in current_messages:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant", avatar="🤖").write(msg["content"])
        
        # 새 대화에서 질문이 들어오면 세션 시작
        if not st.session_state.current_session_id:
            # 현재 선택된 세션이 없을 경우, 질문을 입력하면 새 세션으로 자동 시작
            if prompt := st.chat_input("질문을 입력하세요... (자동으로 새 대화 시작)"):
                _create_new_chat("무제 대화", "기타")
                # 새 세션이 생성되었으니 prompt 처리를 위해 rerun
                st.session_state.initial_prompt = prompt
                st.rerun()
        
        # 사용자 입력 처리 (세션이 활성화된 경우)
        if st.session_state.current_session_id and (prompt := st.chat_input("질문을 입력하세요... (예: RSI가 뭐야?)", key="chat_input_active")):
            
            # 초기 프롬프트 처리 (이전 단계에서 자동 생성된 경우)
            if 'initial_prompt' in st.session_state:
                 prompt = st.session_state.initial_prompt
                 del st.session_state.initial_prompt

            current_session_id = st.session_state.current_session_id
            
            # 1. 설정
            genai.configure(api_key=api_key)
            
            # 2. 사용자 메시지 저장 (현재 세션에 추가)
            st.session_state.chat_sessions[current_session_id]['messages'].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            try:
                with st.spinner("Gemini가 분석 중입니다..."):
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # --- [수정된 핵심 부분]: 대화 맥락을 포함하도록 generate_content 호출 변경 ---
                    # 대화 기록을 모델에 전달할 형식으로 변환
                    history_for_api = [
                        {
                            # Gemini API는 'model' role을 사용합니다.
                            "role": m['role'].replace('assistant', 'model'), 
                            "parts": [{"text": m['content']}]
                        }
                        for m in st.session_state.chat_sessions[current_session_id]['messages']
                    ]
                    
                    # system_instruction은 config 딕셔너리가 아닌, generate_content의 키워드 인수로 직접 전달하는 방식이
                    # 가장 최신 및 표준 SDK에서 안정적입니다.
                    system_instruction_text = (
                        "당신은 금융 및 주식 시장 분석에 특화된 유능한 Gemini AI 어시스턴트입니다. "
                        "친절하고 정확하게 답변하며, 질문에 대한 구체적인 근거와 설명을 제공합니다. "
                        "한국어로 대화하며, 전문 용어는 쉽게 풀어서 설명해주고, 투자 권유가 아닌 정보 제공임을 명시합니다."
                    )
                    
                    response = model.generate_content(
                        contents=history_for_api, # 전체 대화 기록을 전달하여 맥락 유지
                        system_instruction=system_instruction_text # config 대신 직접 인수로 전달
                    )
                    # -----------------------------------------------------------------------

                    ai_msg = response.text
                    
                    # AI 응답 저장 (현재 세션에 추가)
                    st.session_state.chat_sessions[current_session_id]['messages'].append({"role": "assistant", "content": ai_msg})
                    st.session_state.chat_sessions[current_session_id]['created_at'] = datetime.now() # Update timestamp
                    
                    st.chat_message("assistant", avatar="🤖").write(ai_msg)
                    
            except Exception as e:
                # 에러 메시지 출력 시, 오류 원인을 더 명확히 알 수 있도록 예외 처리를 유지합니다.
                st.error(f"오류가 발생했습니다: {e}")
                # 에러 발생 시 사용자 메시지만 남기고 AI 메시지는 추가하지 않음
                st.session_state.chat_sessions[current_session_id]['messages'].pop() 
                
# ----------------------------------------------------------------------
# 11. 푸터 (기존 코드 유지)
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        본 서비스는 학습·연구용 데모이며, 실제 투자 의사결정에 사용하기 전 반드시 별도의 검증이 필요합니다.
    </div>
    """,
    unsafe_allow_html=True,
)
