import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- [1. 기본 설정 및 마스터 데이터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_MASTER = {'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8}

# --- [2. 데이터 정제 유틸리티] ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date_smart(series):
    s = series.astype(str).str.replace('.0', '', regex=False).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

def smart_load_csv(file):
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            if df.columns.str.contains('Unnamed').sum() > len(df.columns) * 0.4:
                for i in range(1, 6):
                    file.seek(0)
                    df = pd.read_csv(file, skiprows=i, encoding=enc)
                    if not df.columns.str.contains('Unnamed').all(): break
            return df
        except: continue
    return None

# --- [3. 상세 팝업창 (크기 확대 및 필터)] ---
@st.dialog("현장별 상세 수주 내역", width="large") # 팝업 크기 대폭 확대
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.write(f"🔎 분석 대상 품번 그룹: {', '.join(group_ids)}")
    code_col = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
    
    # 필터링 및 정렬
    detail = df_bl[df_bl[code_col].astype(str).isin(group_ids)].copy()
    detail = detail[(detail['clean_qty'] > 0) & (detail['dt_clean'] >= cutoff_date)]
    
    if detail.empty:
        st.info("조건에 맞는 수주 데이터가 없습니다.")
        return

    st.dataframe(detail.sort_values('dt_clean', ascending=True), use_container_width=True, hide_index=True)

# --- [4. 메인 UI] ---
st.title("🚀 P·Forecast Stock Manager v4.0")

# 파일 인식용 사전 (전역 관리)
RECOGNITION = {
    "backlog": {"name": "수주예정(Demand)", "keys": ["수주잔량", "총예상수량"], "found": False},
    "po": {"name": "구매발주(PO)", "keys": ["PO잔량", "미선적"], "found": False},
    "stock": {"name": "현재고(Stock)", "keys": ["재고수량", "현재고액"], "found": False},
    "item": {"name": "품목정보(Master)", "keys": ["최종생산지명", "이전상품코드"], "found": False},
    "retail": {"name": "시판스펙(Retail)", "keys": ["출시예정", "4개월판매량"], "found": False}
}

with st.sidebar:
    st.header("⚙️ 분석 설정")
    start_date = st.date_input("검토 시점(조회 시작일)", datetime.now())
    freq_opt = st.selectbox("집계 단위", ["주별", "월별", "분기별", "년도별"], index=1)
    exclude_months = st.slider("과거 수주 제외 (N개월 경과)", 1, 36, 12)
    cutoff_date = pd.Timestamp(start_date) - relativedelta(months=exclude_months)
    
    st.markdown("---")
    st.subheader("📁 파일 로드 상태")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)

# 데이터 로드 및 상태 업데이트
data = {}
if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            cols = "|".join(df.columns)
            for k, v in RECOGNITION.items():
                if any(key in cols for key in v["keys"]):
                    data[k] = df
                    RECOGNITION[k]["found"] = True
                    break

# 사이드바 상태 표시 (업로드 대기중 -> 완료)
with st.sidebar:
    for k, v in RECOGNITION.items():
        if v["found"]:
            st.success(f"✅ {v['name']} (업로드 완료)")
        else:
            st.warning(f"⏳ {v['name']} (업로드 대기중)")

# --- [5. 메인 분석 로직] ---
if len(data) >= 5:
    with st.spinner('매트릭스를 생성 중입니다...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
        
        # 데이터 정제
        bl_code = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
        bl_date = '납품예정일' if '납품예정일' in df_bl.columns else df_bl.columns[24]
        df_bl['clean_qty'] = clean_numeric(df_bl['수주잔량'])
        df_bl['dt_clean'] = parse_date_smart(df_bl[bl_date])
        df_bl = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

        po_code = '품번' if '품번' in df_po.columns else df_po.columns[12]
        df_po['clean_qty'] = clean_numeric(df_po['PO잔량(미선적)'])
        df_po['dt_clean'] = parse_date_smart(df_po['입고요청일'] if '입고요청일' in df_po.columns else 'PO일자')

        st_code = '품번' if '품번' in df_st.columns else df_st.columns[7]
        st_qty = '재고수량' if '재고수량' in df_st.columns else df_st.columns[17]
        df_st['clean_qty'] = clean_numeric(df_st[st_qty])

        # 기간 축 생성
        base_dt = pd.Timestamp(start_date)
        freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
        date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
        time_labels = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range[:12]]

        target_ids = df_bl[df_bl['clean_qty'] > 0][bl_code].unique()
        matrix_rows = []
        idx = 1

        for pid in target_ids:
            pid_s = str(pid)
            item_match = df_item[df_item['상품코드'].astype(str) == pid_s]
            
            # 연계 품번 및 생산지 정보 추출
            prev_raw = str(item_match['이전상품코드'].iloc[0]) if not item_match.empty else "nan"
            chng_raw = str(item_match['변경상품코드'].iloc[0]) if not item_match.empty else "nan"
            
            prev_id = prev_raw if prev_raw not in ["nan", "0", "-"] else ""
            chng_id = chng_raw if chng_raw not in ["nan", "0", "-"] else ""
            
            # 연계 품번 생산지 조회
            def get_site_info(code):
                if not code: return ""
                m = df_item[df_item['상품코드'].astype(str) == code]
                if not m.empty:
                    s = str(m['최종생산지명'].iloc[0])
                    l = LT_MASTER.get(s[:2].upper(), 0)
                    return f"({s[:2]}/{l}M)"
                return ""

            prev_site = get_site_info(prev_id)
            chng_site = get_site_info(chng_id)

            group = [g for g in [pid_s, prev_id, chng_id] if g]
            site = str(item_match['최종생산지명'].iloc[0]) if not item_match.empty else "ETC"
            lt = LT_MASTER.get(site[:2].upper(), 0)

            # 시판공용 체크
            is_retail = " 🏷️" if any(str(g) in df_retail.iloc[:, 8].astype(str).values for g in group) else ""

            # 재고 계산
            main_stk = df_st[df_st[st_code].astype(str).isin(group)]['clean_qty'].sum()
            po_kg = df_po[df_po[po_code].astype(str).isin(group)]['clean_qty'].sum()
            po_m = (po_kg * 1000) / (70 * 1.26)

            overdue_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt_clean'] < base_dt)]['clean_qty'].sum()
            running_inv = main_stk - overdue_dem
            d_vals, s_vals = {"납기경과": overdue_dem}, {"납기경과": running_inv}

            for i in range(12):
                start, end = date_range[i], date_range[i+1]
                m_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt_clean'] >= start) & (df_bl['dt_clean'] < end)]['clean_qty'].sum()
                m_po_df = df_po[(df_po[po_code].astype(str).isin(group)) & (df_po['dt_clean'] >= start) & (df_po['dt_clean'] < end)]
                m_sup = sum([(r['clean_qty'] * 1000) / (70 * 1.26) for _, r in m_po_df.iterrows()])
                running_inv = (running_inv + m_sup) - m_dem
                d_vals[time_labels[i]], s_vals[time_labels[i]] = round(m_dem, 0), round(running_inv, 0)

            common = {"No": idx, "수주품번": pid_s + is_retail, "본사재고": main_stk, "PO잔량(m)": po_m, "생산지": f"{site}({lt}M)", "group": group}
            matrix_rows.append({**common, "구분": "소요량", "연계정보": f"이전:{prev_id} {prev_site}" if prev_id else "", **d_vals})
            matrix_rows.append({"No": idx, "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "group": group, "구분": "예상재고", "연계정보": f"변경:{chng_id} {chng_site}" if chng_id else "", **s_vals})
            idx += 1

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        
        def style_fn(row):
            group_idx = (row.name // 2)
            base_bg = '#f5f5f5' if group_idx % 2 == 0 else '#ffffff'
            styles = [f'background-color: {base_bg}'] * len(row)
            for i, col in enumerate(row.index):
                if col == "구분": styles[i] = 'background-color: #e1f5fe; font-weight: bold'
                elif row['구분'] == "예상재고" and (col == "납기경과" or col in time_labels):
                    if row[col] < 0: styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        st.subheader(f"📊 수급 분석 매트릭스 ({freq_opt} 집계)")
        
        sel = st.dataframe(
            res_df.style.apply(style_fn, axis=1).format({"본사재고": "{:,.0f}", "PO잔량(m)": "{:,.0f}"}, na_rep=""),
            use_container_width=True, hide_index=True,
            column_order=["No", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels,
            on_select="rerun", selection_mode="single-row"
        )

        if sel.selection.rows:
            sel_idx = sel.selection.rows[0]
            target_row = res_df.iloc[sel_idx if res_df.iloc[sel_idx]['수주품번'] != '' else sel_idx-1]
            if st.button(f"🔍 {target_row['수주품번'].replace('🏷️','')} 상세 내역 보기"):
                show_detail_popup(target_row['group'], df_bl, cutoff_date)
else:
    st.info("사이드바에 5종 파일을 모두 업로드해주세요.")
