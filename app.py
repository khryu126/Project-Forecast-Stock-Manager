import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ==================================================
# 페이지 설정
# ==================================================
st.set_page_config(page_title="Project Forecast Stock Manager", layout="wide")
st.title("📊 Project Forecast Stock Manager")
st.caption("성지라미텍 특판 모양지 통합 재고 관리 시스템 (V7)")

# ==================================================
# [안전 장치] 유틸리티 함수
# ==================================================
def to_num_series(series):
    if series is None: return pd.Series(0.0)
    s = series.astype(str).str.replace(",", "").str.replace(" ", "").str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def safe_int_cast(val):
    try:
        if pd.isna(val) or np.isinf(val): return 0
        return int(round(float(val)))
    except: return 0

def safe_read(file):
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
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
            df = df.loc[:, ~df.columns.str.contains("Unnamed")]
            return df
        except: continue
    return None

# ==================================================
# 파일 업로드 및 데이터 매핑
# ==================================================
files = st.sidebar.file_uploader("CSV 파일 통합 업로드", type="csv", accept_multiple_files=True)

if not files:
    st.info("👈 왼쪽 사이드바에 파일들을 한꺼번에 드래그해서 올려주세요.")
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
    st.warning("⚠️ '수주예정등록'과 '현재고' 파일이 반드시 필요합니다.")
    st.stop()

# --- 기준 설정 ---
st.sidebar.divider()
base_date = st.sidebar.date_input("조회 기준일", value=datetime.today())
period_type = st.sidebar.selectbox("예측 단위", ["월별", "분기별"])
period_count = st.sidebar.slider("예측 기간", 4, 12, 6)

# ==================================================
# 전처리 및 계산 (소요량 0 에러 해결)
# ==================================================
order = data_map["order"]
item_col = '상품코드' if '상품코드' in order.columns else '품번'
order['수주잔량_n'] = to_num_series(order['수주잔량'])

# 날짜 변환 (yyyyMMdd 혹은 다른 형식 모두 대응)
order['납기일'] = pd.to_datetime(order['납품예정일'].astype(str), format='%Y%m%d', errors='coerce')
if order['납기일'].isna().all():
    order['납기일'] = pd.to_datetime(order['납품예정일'].astype(str), errors='coerce')

raw_targets = order[order['수주잔량_n'] > 0][item_col].dropna().unique()
target_items = sorted([str(x).strip() for x in raw_targets])

stock_df = data_map["stock"]
po_df = data_map.get("po")
info_df = data_map.get("item_info")
market_df = data_map.get("market")

# 기간 헤더
periods = []
for i in range(period_count):
    if period_type == "월별":
        d = base_date + relativedelta(months=i)
        periods.append(d.strftime("%Y-%m"))
    else:
        d = base_date + relativedelta(months=i*3)
        periods.append(f"{d.year} Q{(d.month-1)//3 + 1}")

matrix_rows = []
row_no = 1

for code in target_items:
    item_rows = order[order[item_col].astype(str).str.strip() == code]
    if item_rows.empty: continue
    
    raw_name = str(item_rows['상품명'].iloc[0])
    m_list = [str(x).strip() for x in market_df['품번' if '품번' in market_df.columns else '상품코드'].values] if market_df is not None else []
    display_name = raw_name + " (🏷️시판공용)" if code in m_list else raw_name

    # 평량 확보
    bw = 70.0
    if info_df is not None:
        i_col = '상품코드' if '상품코드' in info_df.columns else '품번'
        w_col = 'B/P무게' if 'B/P무게' in info_df.columns else 'B/P weight'
        bw_match = info_df[info_df[i_col].astype(str).str.strip() == code]
        if not bw_match.empty:
            try: bw = float(bw_match[w_col].iloc[0])
            except: bw = 70.0
    if bw <= 0: bw = 70.0

    # 가용 재고 계산
    hq_stock = to_num_series(stock_df[stock_df['품번'].astype(str).str.strip() == code]['재고수량']).sum()
    po_stock_m = 0
    if po_df is not None:
        p_item_col = '품번' if '품번' in po_df.columns else '상품코드'
        p_qty_col = next((c for c in po_df.columns if "PO 수량" in c or "PO잔량" in c), None)
        if p_qty_col:
            po_kg = to_num_series(po_df[po_df[p_item_col].astype(str).str.strip() == code][p_qty_col]).sum()
            po_stock_m = (po_kg * 1000) / (bw * 1.26)

    # 행 생성 (유 대리님 요청: 행넘버 통합 및 재고 열 추가)
    row_dem = {"No.": row_no, "품번": code, "상품명": display_name, "본사재고": int(hq_stock), "PO재고": int(po_stock_m), "구분": "소요량(m)"}
    row_inv = {"No.": row_no, "품번": "", "상품명": "", "본사재고": "", "PO재고": "", "구분": "예상재고(m)"}
    
    current_running_balance = hq_stock + po_stock_m
    
    for p in periods:
        if period_type == "월별":
            p_start = datetime.strptime(p, "%Y-%m")
            p_end = p_start + relativedelta(months=1)
        else:
            y, q = int(p.split(' ')[0]), int(p.split('Q')[1])
            p_start = datetime(y, (q-1)*3 + 1, 1); p_end = p_start + relativedelta(months=3)
        
        demand = order[(order[item_col].astype(str).str.strip() == code) & (order['납기일'] >= p_start) & (order['납기일'] < p_end)]['수주잔량_n'].sum()
        current_running_balance -= demand
        
        row_dem[p] = safe_int_cast(demand)
        row_inv[p] = safe_int_cast(current_running_balance)
        
    matrix_rows.append(row_dem)
    matrix_rows.append(row_inv)
    row_no += 1

# ==================================================
# 결과 출력 및 스타일링
# ==================================================
if matrix_rows:
    final_df = pd.DataFrame(matrix_rows)
    
    def style_inventory(v):
        if isinstance(v, (int, float)) and v < 0: return 'background-color: #ffcccc; color: #900;'
        if isinstance(v, (int, float)) and v > 0: return 'background-color: #f0fff4; color: #060;'
        return ''

    st.subheader("🗓️ 오더 검토 및 수지 분석 매트릭스")
    st.dataframe(final_df.style.applymap(style_inventory, subset=periods), use_container_width=True, height=600)
    
    st.divider()
    
    # --- 상세 내역 조회 (에러 방어 버전) ---
    st.subheader("🔍 품번별 수주 상세 내역")
    sel_item = st.selectbox("조회할 품번을 선택하세요", target_items)
    
    if sel_item:
        detail_view = order[order[item_col].astype(str).str.strip() == sel_item].copy()
        # 존재하지 않는 컬럼이 있을 경우를 대비해 안전하게 필터링
        available_cols = [c for c in ['현장명', '건설사', '수주잔량_n', '납품예정일', '비고'] if c in detail_view.columns]
        st.table(detail_view[available_cols].dropna(subset=[available_cols[0]]).sort_values('납품예정일'))

    csv = final_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 전체 결과 다운로드 (CSV)", csv, f"Inventory_Report_{datetime.now().strftime('%m%d')}.csv", "text/csv")
