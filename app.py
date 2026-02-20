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
    'SH': {'total': 1, 'ship_days': 30},
    'KD': {'total': 2, 'ship_days': 30},
    'QZ': {'total': 2, 'ship_days': 30}
}

# --- [2. 지능형 유틸리티 함수] ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date_smart(series):
    s = series.astype(str).str.replace('.0', '', regex=False).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

def smart_load_csv(file):
    """헤더 행을 자동으로 찾아내어 로드하는 지능형 함수"""
    # 인식용 핵심 키워드들
    critical_keys = ["수주잔량", "총예상수량", "PO잔량", "미선적", "재고수량", "현재고", "상품코드", "최종생산지"]
    
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            for skip in range(10): # 상단 10줄까지 헤더 탐색
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, skiprows=skip)
                # 컬럼 중 하나라도 핵심 키워드를 포함하고 있다면 해당 행이 헤더임
                if any(any(key in str(col) for key in critical_keys) for col in df.columns):
                    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            # 키워드를 못 찾았더라도 일단 로드 시도
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            return df
        except: continue
    return None

def find_col(df, keywords):
    for k in keywords:
        for col in df.columns:
            if k in str(col).replace(" ", ""):
                return col
    return None

# --- [3. 상세 팝업창] ---
@st.dialog("현장별 상세 수주 내역", width="large")
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.write(f"🔎 분석 대상 품번 그룹: {', '.join(group_ids)}")
    code_col = find_col(df_bl, ['상품코드', '품번', 'Item'])
    detail = df_bl[df_bl[code_col].astype(str).isin(group_ids)].copy()
    detail = detail[(detail['clean_qty'] > 0) & (detail['dt_clean'] >= cutoff_date)]
    if detail.empty:
        st.info("조건에 맞는 수주 데이터가 없습니다.")
        return
    st.dataframe(detail.sort_values('dt_clean', ascending=True), use_container_width=True, hide_index=True)

# --- [4. 메인 UI] ---
st.title("🚀 P·Forecast Stock Manager v5.8")

# 파일 인식 사전 (키워드 보강)
RECOGNITION = {
    "backlog": {"name": "수주예정(Demand)", "keys": ["수주잔량", "총예상수량", "잔량"], "found": False},
    "po": {"name": "구매발주(PO)", "keys": ["PO잔량", "미선적"], "found": False},
    "stock": {"name": "현재고(Stock)", "keys": ["재고수량", "현재고액"], "found": False},
    "item": {"name": "품목정보(Master)", "keys": ["최종생산지", "이전상품코드"], "found": False},
    "retail": {"name": "시판스펙(Retail)", "keys": ["출시예정", "4개월판매량"], "found": False}
}

with st.sidebar:
    st.header("⚙️ 분석 설정")
    start_date_val = st.date_input("검토 시점", datetime.now())
    freq_opt = st.selectbox("집계 단위", ["주별", "월별", "분기별", "년도별"], index=1)
    exclude_months = st.slider("과거 수주 제외 (N개월)", 1, 36, 12)
    cutoff_date = pd.Timestamp(start_date_val) - relativedelta(months=exclude_months)
    st.markdown("---")
    search_query = st.text_input("🔍 키워드 검색 (품명/품번)", "")
    st.markdown("---")
    st.subheader("📁 파일 로드 상태")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)

# 데이터 로딩 및 상태 업데이트
data = {}
if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            cols_text = "|".join(df.columns)
            for k, v in RECOGNITION.items():
                if any(key in cols_text for key in v["keys"]):
                    data[k] = df
                    RECOGNITION[k]["found"] = True
                    break

# 사이드바 리스트 표시
with st.sidebar:
    for k, v in RECOGNITION.items():
        if v["found"]: st.success(f"✅ {v['name']} (완료)")
        else: st.warning(f"⏳ {v['name']} (대기중)")

# --- [5. 메인 분석 로직] ---
if len(data) >= 5:
    with st.spinner('품번 그룹화 및 수급 시뮬레이션 중...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
        today_dt = pd.Timestamp(datetime.now().date())
        base_dt = pd.Timestamp(start_date_val)

        # 컬럼 자동 검색
        it_code = find_col(df_item, ['상품코드', '품번'])
        it_site = find_col(df_item, ['최종생산지'])
        it_prev = find_col(df_item, ['이전상품코드'])
        
        bl_code = find_col(df_bl, ['상품코드', '품번'])
        bl_qty = find_col(df_bl, ['수주잔량', '잔량'])
        bl_date = find_col(df_bl, ['납품예정일'])
        
        po_code = find_col(df_po, ['품번', '상품코드'])
        po_qty = find_col(df_po, ['PO잔량', '미선적'])
        po_prod = find_col(df_po, ['생산예정일'])
        po_date = find_col(df_po, ['PO일자', '발주일자', '입고요청일'])

        st_code = find_col(df_st, ['품번', '상품코드'])
        st_qty = find_col(df_st, ['재고수량', '현재고'])

        master_info = df_item.set_index(df_item[it_code].astype(str).str.strip())
        site_map = master_info[it_site].to_dict()
        prev_map = master_info[it_prev].to_dict()
        next_map = df_item.set_index(df_item[it_prev].astype(str).str.strip())[it_code].to_dict()

        df_bl['clean_qty'] = clean_numeric(df_bl[bl_qty])
        df_bl['dt_clean'] = parse_date_smart(df_bl[bl_date])
        df_bl = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

        df_po['m_qty'] = clean_numeric(df_po[po_qty]) * 11.3378 

        def get_arrival_v58(row):
            pid = str(row[po_code]).strip()
            site_raw = str(row.get('생산지명', site_map.get(pid, 'ETC'))).upper()
            lt = LT_CONFIG.get(site_raw[:2], {'total': 0, 'ship_days': 0})
            p_dt = parse_date_smart(pd.Series([row.get(po_prod, np.nan)]))[0]
            if pd.notnull(p_dt):
                return p_dt + timedelta(days=int(lt['ship_days']))
            else:
                b_dt = parse_date_smart(pd.Series([row.get(po_date, np.nan)]))[0]
                if pd.isna(b_dt): b_dt = today_dt
                return b_dt + relativedelta(months=int(lt['total']))

        df_po['dt_arrival'] = df_po.apply(get_arrival_v58, axis=1)
        df_st['clean_qty'] = clean_numeric(df_st[st_qty])

        freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
        date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
        time_labels = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range[:12]]

        target_ids = df_bl[df_bl['clean_qty'] > 0][bl_code].unique()
        matrix_rows, alert_list = [], []
        idx_no = 1

        for pid in target_ids:
            pid_s = str(pid).strip()
            item_match = df_item[df_item[it_code].astype(str).str.strip() == pid_s]
            p_name = str(item_match['상품명'].iloc[0]) if not item_match.empty else "-"
            if search_query and (search_query.lower() not in p_name.lower() and search_query.lower() not in pid_s.lower()): continue

            prev_id = str(prev_map.get(pid_s, "nan"))
            prev_id = "" if prev_id in ["nan", "None", "0", "-"] else prev_id
            next_id = str(next_map.get(pid_s, "nan"))
            next_id = "" if next_id in ["nan", "None", "0", "-"] else next_id

            group = list(set([pid_s, prev_id, next_id])); group = [g for g in group if g]
            site_name = str(site_map.get(pid_s, "ETC"))
            lt_total = LT_CONFIG.get(site_name[:2].upper(), {'total': 0})['total']
            is_retail = " 🏷️" if any(str(g) in df_retail.iloc[:, 8].astype(str).values for g in group) else ""

            main_stk = df_st[df_st[st_code].astype(str).str.strip().isin(group)]['clean_qty'].sum()
            gap_po = df_po[(df_po[po_code].astype(str).str.strip().isin(group)) & (df_po['dt_arrival'] >= today_dt) & (df_po['dt_arrival'] < base_dt)]['m_qty'].sum()
            total_stk = main_stk + gap_po
            
            overdue_dem = df_bl[(df_bl[bl_code].astype(str).str.strip().isin(group)) & (df_bl['dt_clean'] < base_dt)]['clean_qty'].sum()
            running_inv = total_stk - overdue_dem
            
            d_row = {"No": idx_no, "품명": p_name, "수주품번": pid_s + is_retail, "본사재고": total_stk, "PO잔량(m)": df_po[df_po[po_code].astype(str).str.strip().isin(group)]['m_qty'].sum(), "생산지": f"{site_name[:2]}({lt_total}M)", "구분": "소요량", "연계정보": f"이전:{prev_id}" if prev_id else "", "납기경과": overdue_dem, "group": group}
            p_row = {"No": idx_no, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "입고량(PO)", "연계정보": "", "납기경과": gap_po, "group": group}
            s_row = {"No": idx_no, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "예상재고", "연계정보": f"변경:{next_id}" if next_id else "", "납기경과": running_inv, "group": group}

            for i in range(12):
                start, end = date_range[i], date_range[i+1]
                m_dem = df_bl[(df_bl[bl_code].astype(str).str.strip().isin(group)) & (df_bl['dt_clean'] >= start) & (df_bl['dt_clean'] < end)]['clean_qty'].sum()
                m_sup = df_po[(df_po[po_code].astype(str).str.strip().isin(group)) & (df_po['dt_arrival'] >= start) & (df_po['dt_arrival'] < end)]['m_qty'].sum()
                running_inv = (running_inv + m_sup) - m_dem
                d_row[time_labels[i]], p_row[time_labels[i]], s_row[time_labels[i]] = m_dem, m_sup, running_inv
                if running_inv < 0 and start < base_dt + relativedelta(months=lt_total):
                    alert_list.append({"품명": p_name, "품번": pid_s, "부족시점": time_labels[i], "부족수량": abs(running_inv)})

            matrix_rows.extend([d_row, p_row, s_row]); idx_no += 1

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        num_cols = ["본사재고", "PO잔량(m)", "납기경과"] + time_labels
        for c in num_cols: res_df[c] = pd.to_numeric(res_df[c], errors='coerce')

        def style_fn(row):
            g_idx = (res_df.index.get_loc(row.name) // 3)
            bg = '#f9f9f9' if g_idx % 2 == 0 else '#ffffff'
            styles = [f'background-color: {bg}'] * len(row)
            for i, col in enumerate(row.index):
                if col == "구분": styles[i] = 'background-color: #e1f5fe; font-weight: bold'
                elif row['구분'] == "예상재고" and col in num_cols:
                    if row[col] < 0: styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        if alert_list:
            if st.button(f"⚠️ 긴급 발주 검토 대상 ({len(pd.DataFrame(alert_list)['품번'].unique())}건)"):
                st.table(pd.DataFrame(alert_list).drop_duplicates(subset=['품번'], keep='first').style.format({"부족수량": "{:,.0f}"}))

        st.subheader(f"📊 수급 분석 매트릭스 ({freq_opt})")
        st_df = st.dataframe(
            res_df.style.apply(style_fn, axis=1).format({c: "{:,.0f}" for c in num_cols}, na_rep=""),
            use_container_width=True, hide_index=True,
            column_order=["No", "품명", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels,
            on_select="rerun", selection_mode="single-row"
        )
        if st_df.selection.rows:
            s_idx = st_df.selection.rows[0]
            target = res_df.iloc[s_idx - (s_idx % 3)]
            if st.button(f"🔍 {target['수주품번'].replace('🏷️','').strip()} 상세 보기"):
                show_detail_popup(target['group'], df_bl, cutoff_date)
