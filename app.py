import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ==================================================
# 페이지 설정 및 타이틀
# ==================================================
st.set_page_config(
    page_title="Project Forecast Stock Manager",
    layout="wide"
)

st.title("📊 Project Forecast Stock Manager")
st.caption("성지라미텍 특판 모양지 통합 재고 관리 시스템")

# ==================================================
# [핵심] 에러 방지용 안전 장치 함수
# ==================================================
def clean_numeric(val):
    """NaN, Inf, None 등을 모두 0.0으로 안전하게 변환"""
    try:
        if pd.isna(val) or np.isinf(val):
            return 0.0
        return float(val)
    except:
        return 0.0

def to_num_series(series):
    """문자열 숫자를 안전하게 숫자로 변환 (쉼표 제거 포함)"""
    if series is None: return pd.Series(0.0)
    s = series.astype(str).str.replace(",", "").str.replace(" ", "").str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def safe_read(file):
    """다양한 인코딩과 헤더 위치를 자동으로 찾아 읽기"""
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
            df_temp = pd.read_csv(file, encoding=enc, header=None, nrows=10)
            header_idx = 0
            for i, row in df_temp.iterrows():
                row_str = " ".join(row.astype(str))
                # 핵심 키워드로 헤더 줄 찾기
                if any(k in row_str for k in ["상품코드", "품번", "재고", "수주", "PO"]):
                    header_idx = i
                    break
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, skiprows=header_idx)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, ~df.columns.str.contains("Unnamed")]
            return df
        except:
            continue
    return None

# ==================================================
# 파일 업로드 및 식별
# ==================================================
files = st.sidebar.file_uploader("CSV 파일들을 한꺼번에 선택하여 업로드하세요", type="csv", accept_multiple_files=True)

if not files:
    st.info("👈 왼쪽 사이드바에서 관련 CSV 파일들을 드래그하여 업로드해 주세요.")
    st.stop()

data_map = {}
for f in files:
    df = safe_read(f)
    if df is not None:
        cols = " ".join(df.columns)
        if '재고수량' in cols: data_map["stock"] = df
        elif 'PO 수량' in cols or 'PO잔량' in cols: data_map["po"] = df
        elif '수주잔량' in cols: data_map["order"] = df
        elif '4개월판매량' in cols: data_map["market"] = df
        elif 'B/P무게' in cols or 'B/P weight' in cols: data_map["item_info"] = df

if "stock" not in data_map or "order" not in data_map:
    st.warning("⚠️ 필수 파일('수주예정등록'과 '현재고')이 인식되지 않았습니다. 파일 안의 컬럼명을 확인해 주세요.")
    st.stop()

# ==================================================
# 분석 기준 설정
# ==================================================
st.sidebar.divider()
base_date = st.sidebar.date_input("조회 기준일 (Today)", value=datetime.today())
period_type = st.sidebar.selectbox("예측 단위", ["월별", "분기별"])
period_count = st.sidebar.slider("예측 기간", 4, 12, 6)

# ==================================================
# 메인 계산 로직 (에러 방어 강화)
# ==================================================
order = data_map["order"]
item_col = '상품코드' if '상품코드' in order.columns else '품번'
order['수주잔량_n'] = to_num_series(order['수주잔량'])
order['납기일'] = pd.to_datetime(order['납품예정일'].astype(str), errors='coerce')

# 수주잔량이 있는 품번만 정렬하여 추출
target_items = sorted(order[order['수주잔량_n'] > 0][item_col].unique())

stock_df = data_map["stock"]
po_df = data_map.get("po")
info_df = data_map.get("item_info")
market_df = data_map.get("market")

# 기간 리스트 생성
periods = []
for i in range(period_count):
    if period_type == "월별":
        d = base_date + relativedelta(months=i)
        periods.append(d.strftime("%Y-%m"))
    else:
        d = base_date + relativedelta(months=i*3)
        periods.append(f"{d.year} Q{(d.month-1)//3 + 1}")

matrix_rows = []

for code in target_items:
    # 1. 품명 및 기본 정보
    item_rows = order[order[item_col] == code]
    if item_rows.empty: continue
    
    raw_name = str(item_rows['상품명'].iloc[0])
    # 시판 공용 여부 체크
    m_list = market_df['품번' if '품번' in market_df.columns else '상품코드'].values if market_df is not None else []
    display_name = raw_name + " (🏷️시판공용)" if code in m_list else raw_name

    # 2. 평량 확보 (PO 환산용)
    bw = 70.0
    if info_df is not None:
        i_col = '상품코드' if '상품코드' in info_df.columns else '품번'
        w_col = 'B/P무게' if 'B/P무게' in info_df.columns else 'B/P weight'
        bw_match = info_df[info_df[i_col] == code]
        if not bw_match.empty:
            bw = clean_numeric(bw_match[w_col].iloc[0])
    if bw <= 0: bw = 70.0 # 평량이 0이면 70으로 고정 (나눗셈 에러 방지)

    # 3. 기초 가용 재고 (현재고 + PO)
    curr_inv = clean_numeric(to_num_series(stock_df[stock_df['품번'] == code]['재고수량']).sum())
    po_m = 0
    if po_df is not None:
        p_item_col = '품번' if '품번' in po_df.columns else '상품코드'
        p_qty_col = 'PO 수량' if 'PO 수량' in po_df.columns else 'PO잔량'
        po_kg = clean_numeric(to_num_series(po_df[po_df[p_item_col] == code][p_qty_col]).sum())
        po_m = (po_kg * 1000) / (bw * 1.26)

    # 4. 행 데이터 구성
    row_dem = {"품번": code, "상품명": display_name, "구분": "소요량(m)"}
    row_inv = {"품번": "", "상품명": "", "구분": "예상재고(m)"}
    
    current_running_balance = curr_inv + po_m
    
    for p in periods:
        if period_type == "월별":
            p_start = datetime.strptime(p, "%Y-%m")
            p_end = p_start + relativedelta(months=1)
        else:
            y, q = int(p.split(' ')[0]), int(p.split('Q')[1])
            p_start = datetime(y, (q-1)*3 + 1, 1)
            p_end = p_start + relativedelta(months=3)
        
        # 순수 특판 수요만 계산
        demand = clean_numeric(order[(order[item_col] == code) & (order['납기일'] >= p_start) & (order['납기일'] < p_end)]['수주잔량_n'].sum())
        current_running_balance -= demand
        
        row_dem[p] = int(round(demand))
        # [에러 해결 지점] clean_numeric으로 한 번 더 감싸서 안전하게 변환
        row_inv[p] = int(round(clean_numeric(current_running_balance)))
        
    matrix_rows.append(row_dem)
    matrix_rows.append(row_inv)

# ==================================================
# 결과 출력 및 대시보드
# ==================================================
if matrix_rows:
    final_df = pd.DataFrame(matrix_rows)
    
    def style_inventory(v):
        if isinstance(v, (int, float)) and v < 0: 
            return 'background-color: #ffcccc; color: #900; font-weight: bold;'
        if isinstance(v, (int, float)) and v > 0: 
            return 'background-color: #f0fff4; color: #060;'
        return ''

    st.subheader("🗓️ 품번별 통합 수지 분석 매트릭스")
    # 최신 Streamlit 문법에 맞춰 스타일 적용
    st.dataframe(final_df.style.applymap(style_inventory, subset=periods), use_container_width=True, height=500)
    
    # 상세 현장 조회
    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        sel_item = st.selectbox("🔍 상세 내역을 볼 품번을 선택하세요", target_items)
    
    if sel_item:
        detail = order[order[item_col] == sel_item][['현장명', '건설사', '수주잔량_n', '납품예정일', '비고']]
        st.table(detail.dropna(subset=['현장명']).sort_values('납품예정일'))
        st.caption(f"※ 위 분석 결과는 순수 특판 납기 일정 기반입니다. 시판 공용 품번은 별도로 주의해 주세요.")

    # 결과 다운로드
    csv = final_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 결과 다운로드 (CSV)", csv, f"Forecast_Report_{datetime.now().strftime('%m%d')}.csv", "text/csv")
