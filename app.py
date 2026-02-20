import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- [1. 설정 및 리드타임 마스터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_CONFIG = {
    'SE': {'total': 6, 'ship_days': 90},
    'SRL': {'total': 8, 'ship_days': 90},
    'SP': {'total': 8, 'ship_days': 90},
    'SH': {'total': 1, 'ship_days': 30}, # 상해 1개월
    'KD': {'total': 2, 'ship_days': 30},
    'QZ': {'total': 2, 'ship_days': 30}
}

# --- [2. 유틸리티 함수] ---
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

# --- [3. 상세 팝업창] ---
@st.dialog("현장별 상세 수주 내역", width="large")
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.write(f"🔎 분석 대상 품번 그룹: {', '.join(group_ids)}")
    # 수주예정등록 G열(index 5) 또는 이름 기반
    code_col = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
    detail = df_bl[df_bl[code_col].astype(str).isin(group_ids)].copy()
    detail = detail[(detail['clean_qty'] > 0) & (detail['dt_clean'] >= cutoff_date)]
    if detail.empty:
        st.info("조건에 맞는 수주 데이터가 없습니다.")
        return
    st.dataframe(detail.sort_values('dt_clean', ascending=True), use_container_width=True, hide_index=True)

# --- [4. 메인 UI] ---
st.title("🚀 P·Forecast Stock Manager v5.5")

with st.sidebar:
    st.header("⚙️ 분석 설정")
    start_date_val = st.date_input("검토 시점(조회 시작일)", datetime.now())
    freq_opt = st.selectbox("집계 단위", ["주별", "월별", "분기별", "년도별"], index=1)
    exclude_months = st.slider("과거 수주 제외 (N개월 경과)", 1, 36, 12)
    cutoff_date = pd.Timestamp(start_date_val) - relativedelta(months=exclude_months)
    st.markdown("---")
    search_query = st.text_input("🔍 품명/품번 키워드 검색", "")
    st.markdown("---")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)

# 데이터 인식
data = {}
RECOGNITION = {
    "backlog": ["수주잔량", "총예상수량"], "po": ["PO잔량", "미선적"],
    "stock": ["재고수량", "현재고액"], "item": ["최종생산지", "이전상품코드"],
    "retail": ["출시예정", "4개월판매량"]
}
if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            cols = "|".join(df.columns)
            for k, v in RECOGNITION.items():
                if any(key in cols for key in v):
                    data[k] = df; break

# --- [5. 메인 분석 로직] ---
if len(data) >= 5:
    with st.spinner('정밀 데이터 맵핑 중...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
        
        today_dt = pd.Timestamp(datetime.now().date())
        base_dt = pd.Timestamp(start_date_val)

        # [품목 마스터 인덱스 매핑 - 사용자 정보 기반]
        # G열(6): 상품코드, M열(12): 최종생산지, N열(13): 이전상품코드
        item_code_idx = 6
        item_site_idx = 12
        item_prev_idx = 13
        
        # 매칭 사전 구축
        master_site_map = df_item.set_index(df_item.iloc[:, item_code_idx].astype(str).str.strip()).iloc[:, item_site_idx - item_code_idx - 1].to_dict()
        master_prev_map = df_item.set_index(df_item.iloc[:, item_code_idx].astype(str).str.strip()).iloc[:, item_prev_idx - item_code_idx - 1].to_dict()
        # 반대 매칭 (이전코드로 현재코드 찾기)
        master_next_map = df_item.set_index(df_item.iloc[:, item_prev_idx].astype(str).str.strip()).iloc[:, item_code_idx - item_prev_idx].to_dict()

        # 1. 수주 데이터 정제 (G열 기준)
        bl_code_col = df_bl.columns[5] # 보통 G열
        df_bl['clean_qty'] = clean_numeric(df_bl['수주잔량'])
        df_bl['dt_clean'] = parse_date_smart(df_bl.iloc[:, 24]) # 보통 Y열
        df_bl = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

        # 2. PO 데이터 정제 (M열 기준)
        po_code_col = df_po.columns[12] # M열
        df_po['m_qty'] = clean_numeric(df_po['PO잔량(미선적)']) * 11.3378 

        def calc_arrival_v55(row):
            pid = str(row[po_code_col]).strip()
            # 마스터에서 생산지 조회
            site_raw = str(master_site_map.get(pid, 'ETC')).upper()
            lt_config = LT_CONFIG.get(site_raw[:2], {'total': 0, 'ship_days': 0})
            
            # 생산예정일 확인
            prod_dt = parse_date_smart(pd.Series([row.get('생산예정일', np.nan)]))[0]
            if pd.notnull(prod_dt):
                return prod_dt + timedelta(days=int(lt_config['ship_days']))
            else:
                po_dt = parse_date_smart(pd.Series([row.get('PO일자', row.get('입고요청일', np.nan))]))[0]
                if pd.isna(po_dt): po_dt = today_dt
                return po_dt + relativedelta(months=int(lt_config['total']))

        df_po['dt_arrival'] = df_po.apply(calc_arrival_v55, axis=1)

        # 3. 현재고 정제 (H열 기준)
        st_code_col = df_st.columns[7] # H열
        df_st['clean_qty'] = clean_numeric(df_st.iloc[:, 17]) # 현재고 수량

        # 4. 기간 설정 및 행렬 루프
        freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
        date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
        time_labels = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range[:12]]

        target_ids = df_bl[df_bl['clean_qty'] > 0][bl_code_col].unique()
        matrix_rows, alert_list = [], []
        idx_no = 1

        for pid in target_ids:
            pid_s = str(pid).strip()
            item_match = df_item[df_item.iloc[:, item_code_idx].astype(str).str.strip() == pid_s]
            p_name = str(item_match['상품명'].iloc[0]) if not item_match.empty else "-"
            if search_query and (search_query.lower() not in p_name.lower() and search_query.lower() not in pid_s.lower()): continue

            # 그룹핑 (이전/현재/다음)
            prev_id = str(master_prev_map.get(pid_s, ""))
            next_id = str(master_next_map.get(pid_s, ""))
            group = list(set([pid_s, prev_id, next_id]))
            group = [g for g in group if g and g not in ["nan", "0", "-"]]

            site_raw = str(master_site_map.get(pid_s, "ETC"))
            lt_total = LT_CONFIG.get(site_raw[:2].upper(), {'total': 0})['total']
            is_retail = " 🏷️" if any(str(g) in df_retail.iloc[:, 8].astype(str).values for g in group) else ""

            # 기초 재고 및 PO 합산
            main_stk = df_st[df_st[st_code_col].astype(str).str.strip().isin(group)]['clean_qty'].sum()
            gap_po_val = df_po[(df_po[po_code_col].astype(str).str.strip().isin(group)) & (df_po['dt_arrival'] >= today_dt) & (df_po['dt_arrival'] < base_dt)]['m_qty'].sum()
            total_start_stk = main_stk + gap_po_val
            po_total_m = df_po[df_po[po_code_col].astype(str).str.strip().isin(group)]['m_qty'].sum()

            overdue_dem = df_bl[(df_bl[bl_code_col].astype(str).str.strip().isin(group)) & (df_bl['dt_clean'] < base_dt)]['clean_qty'].sum()
            running_inv = total_start_stk - overdue_dem
            
            d_row = {"No": idx_no, "품명": p_name, "수주품번": pid_s + is_retail, "본사재고": total_start_stk, "PO잔량(m)": po_total_m, "생산지": f"{site_raw[:2]}({lt_total}M)", "구분": "소요량", "연계정보": f"이전:{prev_id}" if prev_id else "", "납기경과": overdue_dem, "group": group}
            p_row = {"No": idx_no, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "입고량(PO)", "연계정보": "", "납기경과": gap_po_val, "group": group}
            s_row = {"No": idx_no, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "예상재고", "연계정보": f"변경:{next_id}" if next_id else "", "납기경과": running_inv, "group": group}

            for i in range(12):
                start, end = date_range[i], date_range[i+1]
                m_dem = df_bl[(df_bl[bl_code_col].astype(str).str.strip().isin(group)) & (df_bl['dt_clean'] >= start) & (df_bl['dt_clean'] < end)]['clean_qty'].sum()
                m_sup = df_po[(df_po[po_code_col].astype(str).str.strip().isin(group)) & (df_po['dt_arrival'] >= start) & (df_po['dt_arrival'] < end)]['m_qty'].sum()
                running_inv = (running_inv + m_sup) - m_dem
                d_row[time_labels[i]], p_row[time_labels[i]], s_row[time_labels[i]] = m_dem, m_sup, running_inv
                if running_inv < 0 and start < base_dt + relativedelta(months=lt_total):
                    alert_list.append({"품명": p_name, "품번": pid_s, "부족시점": time_labels[i], "부족수량": abs(running_inv)})

            matrix_rows.extend([d_row, p_row, s_row])
            idx_no += 1

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        num_cols = ["본사재고", "PO잔량(m)", "납기경과"] + time_labels
        for c in num_cols: res_df[c] = pd.to_numeric(res_df[c], errors='coerce')

        def style_fn(row):
            g_idx = (row.name // 3)
            bg = '#f9f9f9' if g_idx % 2 == 0 else '#ffffff'
            styles = [f'background-color: {bg}'] * len(row)
            for i, col in enumerate(row.index):
                if col == "구분": styles[i] = 'background-color: #e1f5fe; font-weight: bold'
                elif row['구분'] == "예상재고" and col in num_cols:
                    if row[col] < 0: styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        st.subheader(f"📊 통합 수급 매트릭스 ({freq_opt})")
        st.dataframe(
            res_df.style.apply(style_fn, axis=1).format({c: "{:,.0f}" for c in num_cols}, na_rep=""),
            use_container_width=True, hide_index=True,
            column_order=["No", "품명", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels,
            on_select="rerun", selection_mode="single-row"
        )
