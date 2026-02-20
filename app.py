import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- [1. 기본 설정 및 마스터 데이터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_MASTER = {'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8}

# --- [2. 유틸리티 함수] ---
def clean_numeric(series):
    """숫자 정제: 콤마 제거 및 NaN 0 처리"""
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def smart_load_csv(file):
    """인코딩 및 빈 줄 대응 지능형 로더"""
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            if df.columns.str.contains('Unnamed').sum() > len(df.columns) * 0.4:
                for i in range(1, 5):
                    file.seek(0)
                    df = pd.read_csv(file, skiprows=i, encoding=enc)
                    if not df.columns.str.contains('Unnamed').all(): break
            return df
        except: continue
    return None

# --- [3. 상세 팝업 (Drill-down)] ---
@st.dialog("현장별 상세 수주 내역")
def show_detail_popup(group_ids, df_bl):
    st.write(f"🔎 분석 품번 그룹: {', '.join(group_ids)}")
    # 수주 데이터에서 해당 그룹 추출
    code_col = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
    detail = df_bl[df_bl[code_col].astype(str).isin(group_ids)].copy()
    
    if detail.empty:
        st.info("수주 데이터가 없습니다.")
        return

    st.dataframe(detail.sort_values(detail.columns[24] if len(detail.columns)>24 else detail.columns[-1]), 
                 use_container_width=True, hide_index=True)

# --- [4. 메인 UI] ---
st.title("🚀 P·Forecast Stock Manager v3.0")

# 설정 영역
with st.sidebar:
    st.header("⚙️ 분석 설정")
    start_date = st.date_input("검토 시작일", datetime.now())
    freq_opt = st.selectbox("집계 단위", ["주별", "월별", "분기별", "년도별"], index=1)
    freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
    
    st.markdown("---")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)

# 파일 매핑
data = {}
RECOGNITION = {
    "backlog": ["수주잔량", "총예상수량"], "po": ["PO잔량", "미선적"],
    "stock": ["재고수량", "현재고액"], "item": ["최종생산지명", "이전상품코드"],
    "retail": ["출시예정", "4개월판매량"]
}

if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            cols = "|".join(df.columns)
            for k, v in RECOGNITION.items():
                if any(key in cols for key in v): data[k] = df; break

# 사이드바 상태 표시
with st.sidebar:
    for k, v in RECOGNITION.items():
        if k in data: st.success(f"✅ {k}")
        else: st.error(f"❌ {k}")

# --- [5. 메인 분석 로직] ---
if len(data) >= 5:
    with st.spinner('매트릭스를 생성 중입니다...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
        
        # 1. 컬럼 매핑 및 정제
        bl_code = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
        bl_date = '납품예정일' if '납품예정일' in df_bl.columns else df_bl.columns[24]
        df_bl['수주잔량'] = clean_numeric(df_bl['수주잔량'])
        df_bl['dt'] = pd.to_datetime(df_bl[bl_date], errors='coerce')

        po_code = '품번' if '품번' in df_po.columns else df_po.columns[12]
        po_date = '입고요청일' if '입고요청일' in df_po.columns else 'PO일자'
        df_po['PO잔량(미선적)'] = clean_numeric(df_po['PO잔량(미선적)'])
        df_po['dt'] = pd.to_datetime(df_po[po_date], errors='coerce')

        st_code = '품번' if '품번' in df_st.columns else df_st.columns[7]
        st_qty = '재고수량' if '재고수량' in df_st.columns else df_st.columns[17]
        df_st[st_qty] = clean_numeric(df_st[st_qty])

        # 2. 기간 축 생성
        base_dt = datetime(start_date.year, start_date.month, start_date.day)
        date_range = pd.date_range(start=base_dt, periods=12, freq=freq_map[freq_opt])
        time_cols = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range]

        # 3. 품번 그룹별 루프
        target_ids = df_bl[df_bl['수주잔량'] > 0][bl_code].unique()
        matrix_rows = []
        idx_counter = 1

        for pid in target_ids:
            pid_s = str(pid)
            # 이전/변경 품번 정보 가져오기
            item_info = df_item[df_item['상품코드'].astype(str) == pid_s]
            prev_id = str(item_info['이전상품코드'].iloc[0]) if not item_info.empty and pd.notnull(item_info['이전상품코드'].iloc[0]) else "-"
            next_id = str(item_info['변경상품코드'].iloc[0]) if not item_info.empty and pd.notnull(item_info['변경상품코드'].iloc[0]) else "-"
            group = list(set([pid_s, prev_id, next_id]))
            group = [g for g in group if g != "-"]
            
            # 생산지 및 LT
            site = str(item_info['최종생산지명'].iloc[0]) if not item_info.empty else "ETC"
            lt = LT_MASTER.get(site[:2].upper(), 0)

            # [재고 열 데이터 산출]
            main_stock = df_st[df_st[st_code].astype(str).isin(group)][st_qty].sum()
            po_kg = df_po[df_po[po_code].astype(str).isin(group)]['PO잔량(미선적)'].sum()
            po_m = (po_kg * 1000) / (70 * 1.26) # PO 잔량 m 환산

            # [수지 전개]
            overdue_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt'] < base_dt)]['수주잔량'].sum()
            running_inv = main_stock - overdue_dem
            
            row1_vals = {"납기경과": overdue_dem}
            row2_vals = {"납기경과": running_inv}

            for i, d in enumerate(date_range):
                col_name = time_cols[i]
                next_d = date_range[i+1] if i+1 < len(date_range) else d + pd.DateOffset(years=1)
                
                # 해당 기간 소요량
                m_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt'] >= d) & (df_bl['dt'] < next_d)]['수주잔량'].sum()
                
                # 해당 기간 PO 입고량
                m_po_df = df_po[(df_po[po_code].astype(str).isin(group)) & (df_po['dt'] >= d) & (df_po['dt'] < next_d)]
                m_sup = sum([(r['PO잔량(미선적)'] * 1000) / (70 * 1.26) for _, r in m_po_df.iterrows()])
                
                running_inv = (running_inv + m_sup) - m_dem
                row1_vals[col_name] = round(m_dem, 0)
                row2_vals[col_name] = round(running_inv, 0)

            # 공통 데이터
            common = {"No": idx_counter, "수주품번": pid_s, "본사재고(m)": round(main_stock, 0), "PO잔량(m)": round(po_m, 0), "생산지": f"{site}({lt}M)", "group": group}
            
            # 1행: 소요량 줄
            matrix_rows.append({**common, "구분": "소요량", "연계품번": f"이전:{prev_id}", **row1_vals})
            # 2행: 예상재고 줄
            matrix_rows.append({**common, "구분": "예상재고", "연계품번": f"변경:{next_id}", **row2_vals})
            
            idx_counter += 1

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        
        def style_matrix(row):
            styles = [''] * len(row)
            if row['구분'] == "예상재고":
                for i, col in enumerate(row.index):
                    if (col == "납기경과" or col in time_cols) and isinstance(row[col], (int, float)) and row[col] < 0:
                        styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        st.subheader(f"📊 수급 분석 매트릭스 ({freq_opt} 기준)")
        
        # [수정] selection_mode="single-row" (하이픈 사용)
        try:
            sel = st.dataframe(
                res_df.style.apply(style_matrix, axis=1),
                use_container_width=True, hide_index=True,
                column_order=["No", "수주품번", "구분", "연계품번", "본사재고(m)", "PO잔량(m)", "생산지", "납기경과"] + time_cols,
                on_select="rerun", selection_mode="single-row"
            )

            if sel.selection.rows:
                sel_idx = sel.selection.rows[0]
                if st.button(f"🔍 {res_df.iloc[sel_idx]['수주품번']} 상세 내역 팝업"):
                    show_detail_popup(res_df.iloc[sel_idx]['group'], df_bl)
        except Exception as e:
            st.dataframe(res_df.style.apply(style_matrix, axis=1), use_container_width=True, hide_index=True)
            st.error(f"UI 오류: {e}")
else:
    st.info("사이드바에서 5종 파일을 업로드하면 분석이 시작됩니다.")
