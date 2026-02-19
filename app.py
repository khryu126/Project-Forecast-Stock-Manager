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
# 공통 유틸
# ==================================================
def safe_float(val):
    """NaN이나 무한대를 0으로 변환하는 안전 함수"""
    try:
        res = float(val)
        if np.isnan(res) or np.isinf(res):
            return 0.0
        return res
    except:
        return 0.0

def to_num(series):
    if series is None: return pd.Series(0.0)
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.replace(" ", "").str.strip(),
        errors="coerce"
    ).fillna(0.0)

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
        except:
            continue
    return None

# ==================================================
# 파일 업로드 및 식별
# ==================================================
files = st.sidebar.file_uploader("CSV 파일 통합 업로드", type="csv", accept_multiple_files=True)

if not files:
    st.info("👈 왼쪽에서 파일들을 드래그해서 올려주세요.")
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
    st.error("❌ 필수 파일 부족: '수주예정등록'과 '현재고' 파일은 반드시 필요합니다.")
    st.stop()

# ==================================================
# 기준 설정
# ==================================================
base_date = st.sidebar.date_input("조회 기준일", value=datetime.today())
period_type = st.sidebar.selectbox("예측 단위", ["월별", "분기별"])
period_count = st.sidebar.slider("예측 기간", 4, 12, 6)

# ==================================================
# 메인 계산 로직
# ==================================================
order = data_map["order"]
item_col = '상품코드' if '상품코드' in order.columns else '품번'
order['수주잔량_n'] = to_num(order['수주잔량'])
order['납기일'] = pd.to_datetime(order['납품예정일'].astype(str), errors='coerce')

target_items = sorted(order[order['수주잔량_n'] > 0][item_col].unique())

stock_df = data_map["stock"]
po_df = data_map.get("po")
info_df = data_map.get("item_info")
market_df = data_map.get("market")

periods = []
for i in range(period_count):
    if period_type == "월별":
        d = base_date + relativedelta(months=i)
        periods.append(d.strftime("%Y-%m"))
    else:
        d = base_date + relativedelta(months=i*3)
        periods.append(f"{d.year} Q{(d.month-1)//3 + 1}")

results = []
for code in target_items:
    item_rows = order[order[item_col] == code]
    if item_rows.empty: continue
    
    name = str(item_rows['상품명'].iloc[0])
    is_market = (code in market_df['품번' if '품번' in market_df.columns else '상품코드'].values) if market_df is not None else False
    display_name = name + " (🏷️시판공용)" if is_market else name

    # 평량 체크 (나누기 오류 방지)
    bw = 70.0
    if info_df is not None:
        i_col = '상품코드' if '상품코드' in info_df.columns else '품번'
        w_col = 'B/P무게' if 'B/P무게' in info_df.columns else 'B/P weight'
        bw_match = info_df[info_df[i_col] == code]
        if not bw_match.empty:
            bw = safe_float(bw_match[w_col].iloc[0])
    if bw <= 0: bw = 70.0

    # 초기 재고
    curr_inv = safe_float(to_num(stock_df[stock_df['품번'] == code]['재고수량']).sum())
    po_m = 0
    if po_df is not None:
        p_item_col = '품번' if '품번' in po_df.columns else '상품코드'
        p_qty_col = 'PO 수량' if 'PO 수량' in po_df.columns else 'PO잔량'
        po_kg = safe_float(to_num(po_df[po_df[p_item_col] == code][p_qty_col]).sum())
        po_m = (po_kg * 1000) / (bw * 1.26)

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
        
        demand = safe_float(order[(order[item_col] == code) & (order['납기일'] >= p_start) & (order['납기일'] < p_end)]['수주잔량_n'].sum())
        balance -= demand
        
        row_dem[p] = int(round(demand))
        # 핵심 수정: NaN/Inf 체크 후 정수 변환
        row_inv[p] = int(round(safe_float(balance)))
        
    results.append(row_dem)
    results.append(row_inv)

# ==================================================
# 결과 출력
# ==================================================
if results:
    final_df = pd.DataFrame(results)
    
    # 최신 스타일링 방식 (applymap -> map 권장이나 구버전 호환용으로 유지)
    def style_fn(v):
        if isinstance(v, (int, float)) and v < 0: return 'background-color: #ffcccc; color: #900;'
        if isinstance(v, (int, float)) and v > 0: return 'background-color: #f0fff4; color: #060;'
        return ''

    st.subheader("③ 오더 검토 매트릭스")
    st.dataframe(final_df.style.applymap(style_fn, subset=periods), use_container_width=True)
    
    st.divider()
    sel = st.selectbox("🔍 상세 현장 내역 조회", target_items)
    if sel:
        detail = order[order[item_col] == sel][['현장명', '건설사', '수주잔량_n', '납품예정일', '비고']]
        st.table(detail.dropna(subset=['현장명']).sort_values('납품예정일'))

    csv = final_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 결과 다운로드 (CSV)", csv, "forecast_result.csv", "text/csv")
