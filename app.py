import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ==================================================
# 페이지 설정
# ==================================================
st.set_page_config(
    page_title="Project Forecast Stock Manager",
    layout="wide"
)

st.title("📊 Project Forecast Stock Manager")
st.caption("성지라미텍 특판 모양지 오더 검토 시스템 (수주잔량 · 재고 · PO 통합)")

# ==================================================
# 공통 유틸 (더 똑똑하게 개조)
# ==================================================
def to_num(s):
    """문자열 숫자를 안전하게 숫자로 변환"""
    if s is None: return pd.Series(0.0)
    return pd.to_numeric(
        s.astype(str).str.replace(",", "").str.replace(" ", "").str.strip(),
        errors="coerce"
    ).fillna(0.0)

def safe_read(file):
    """다양한 인코딩과 헤더 위치를 자동으로 찾아 읽기"""
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
            # 일단 읽어보고 첫 5줄에서 진짜 헤더(컬럼명)가 어디인지 찾음
            df_temp = pd.read_csv(file, encoding=enc, header=None, nrows=10)
            header_idx = 0
            for i, row in df_temp.iterrows():
                row_str = " ".join(row.astype(str))
                if any(k in row_str for k in ["상품코드", "품번", "재고", "수주", "PO"]):
                    header_idx = i
                    break
            
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, skiprows=header_idx)
            df.columns = [str(c).strip() for c in df.columns]
            # Unnamed 컬럼 제거
            df = df.loc[:, ~df.columns.str.contains("Unnamed")]
            return df
        except:
            continue
    return None

# ==================================================
# 파일 업로드 및 식별
# ==================================================
st.subheader("① 소스 파일 통합 업로드")
files = st.file_uploader(
    "필수: 현재고, PO, 수주예정등록 / 선택: 시판스펙관리, 품목정보",
    type="csv",
    accept_multiple_files=True
)

if not files:
    st.info("👈 왼쪽에서 파일들을 드래그해서 올려주세요.")
    st.stop()

data_map = {}
for f in files:
    df = safe_read(f)
    if df is not None:
        cols = " ".join(df.columns)
        # 키워드 기반 자동 분류 (더 넉넉하게 잡음)
        if '재고수량' in cols: data_map["stock"] = df
        elif 'PO 수량' in cols or 'PO잔량' in cols: data_map["po"] = df
        elif '수주잔량' in cols: data_map["order"] = df
        elif '4개월판매량' in cols: data_map["market"] = df
        elif 'B/P무게' in cols or 'B/P weight' in cols: data_map["item_info"] = df

# 필수 파일 체크
required_keys = ["stock", "order"] # PO는 없어도 돌아가게 수정
missing = [k for k in required_keys if k not in data_map]

if missing:
    st.error(f"❌ 필수 파일이 부족합니다: {missing} (수주예정등록과 현재고는 반드시 필요합니다)")
    st.stop()

# ==================================================
# 기준 설정
# ==================================================
st.sidebar.header("⚙️ 분석 기준 설정")
base_date = st.sidebar.date_input("조회 기준일 (오늘)", value=datetime.today())
period_type = st.sidebar.selectbox("예측 단위", ["월별", "분기별"])
period_count = st.sidebar.slider("예측 기간", 4, 12, 6)

# ==================================================
# 데이터 정제 및 계산
# ==================================================
# 1. 수주 데이터 기반 품번 리스트업
order = data_map["order"]
item_col = '상품코드' if '상품코드' in order.columns else '품번'
order['수주잔량_n'] = to_num(order['수주잔량'])
order['납기일'] = pd.to_datetime(order['납품예정일'].astype(str), errors='coerce')

# 잔량이 있는 품번만 대상
target_items = order[order['수주잔량_n'] > 0][item_col].unique()

# 2. 결과 테이블 뼈대 만들기
results = []
stock_df = data_map["stock"]
po_df = data_map.get("po")
info_df = data_map.get("item_info")
market_df = data_map.get("market")

# 기간 생성
periods = []
for i in range(period_count):
    if period_type == "월별":
        d = base_date + relativedelta(months=i)
        periods.append(d.strftime("%Y-%m"))
    else:
        d = base_date + relativedelta(months=i*3)
        periods.append(f"{d.year} Q{(d.month-1)//3 + 1}")

for code in target_items:
    # 품명 찾기
    item_rows = order[order[item_col] == code]
    name = str(item_rows['상품명'].iloc[0]) if not item_rows.empty else "알수없음"
    
    # 시판 공용 여부
    is_market = False
    if market_df is not None:
        m_col = '품번' if '품번' in market_df.columns else '상품코드'
        is_market = code in market_df[m_col].values
    display_name = name + " (🏷️시판공용)" if is_market else name

    # 평량 확인 (PO 환산용)
    bw = 70.0
    if info_df is not None:
        i_col = '상품코드' if '상품코드' in info_df.columns else '품번'
        w_col = 'B/P무게' if 'B/P무게' in info_df.columns else 'B/P weight'
        bw_match = info_df[info_df[i_col] == code]
        if not bw_match.empty:
            try: bw = float(bw_match[w_col].iloc[0]) or 70.0
            except: bw = 70.0

    # 초기 재고 계산
    curr_inv = to_num(stock_df[stock_df['품번'] == code]['재고수량']).sum()
    po_m = 0
    if po_df is not None:
        p_item_col = '품번' if '품번' in po_df.columns else '상품코드'
        p_qty_col = 'PO 수량' if 'PO 수량' in po_df.columns else 'PO잔량'
        po_kg = to_num(po_df[po_df[p_item_col] == code][p_qty_col]).sum()
        po_m = (po_kg * 1000) / (bw * 1.26)

    # 행 생성
    row_dem = {"품번": code, "상품명": display_name, "구분": "소요량"}
    row_inv = {"품번": "", "상품명": "", "구분": "예상재고"}
    
    balance = curr_inv + po_m
    
    for p in periods:
        if period_type == "월별":
            p_start = datetime.strptime(p, "%Y-%m")
            p_end = p_start + relativedelta(months=1)
        else:
            y, q = int(p.split(' ')[0]), int(p.split('Q')[1])
            p_start = datetime(y, (q-1)*3 + 1, 1)
            p_end = p_start + relativedelta(months=3)
        
        # 해당 기간 내 특판 소요량
        demand = order[(order[item_col] == code) & (order['납기일'] >= p_start) & (order['납기일'] < p_end)]['수주잔량_n'].sum()
        
        balance -= demand
        row_dem[p] = int(demand)
        row_inv[p] = int(balance)
        
    results.append(row_dem)
    results.append(row_inv)

# ==================================================
# 결과 출력
# ==================================================
if results:
    final_df = pd.DataFrame(results)
    
    def style_fn(v):
        if isinstance(v, (int, float)) and v < 0: return 'background-color: #ffcccc; color: #900;'
        if isinstance(v, (int, float)) and v > 0: return 'background-color: #f0fff4; color: #060;'
        return ''

    st.subheader("③ 오더 검토 매트릭스")
    st.dataframe(final_df.style.applymap(style_fn, subset=periods), use_container_width=True)
    
    # 상세 내역 조회
    st.divider()
    sel = st.selectbox("🔍 상세 현장 내역 조회", target_items)
    if sel:
        detail = order[order[item_col] == sel][['현장명', '건설사', '수주잔량_n', '납품예정일', '비고']]
        st.table(detail.sort_values('납품예정일'))

    st.download_button("📥 결과 다운로드 (CSV)", final_df.to_csv(index=False).encode('utf-8-sig'), "forecast_result.csv", "text/csv")
