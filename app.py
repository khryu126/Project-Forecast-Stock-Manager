import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- [1. 설정 및 리드타임 마스터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_CONFIG = {
    'SE': {'total': 6, 'ship_days': 90},
    'SR': {'total': 8, 'ship_days': 90},
    'SRL': {'total': 8, 'ship_days': 90},
    'SP': {'total': 8, 'ship_days': 90},
    'SH': {'total': 1, 'ship_days': 30},
    'KD': {'total': 2, 'ship_days': 30},
    'QZ': {'total': 2, 'ship_days': 30}
}

# --- [2. 핵심 유틸리티 함수] ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date_smart(series):
    s = series.astype(str).str.replace('.0', '', regex=False).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

def find_col_precise(df, keywords, exclude_keywords=None, default_idx=None):
    for k in keywords:
        for col in df.columns:
            col_upper = str(col).replace(" ", "").upper()
            if k in col_upper:
                if exclude_keywords:
                    if any(ex.upper() in col_upper for ex in exclude_keywords): continue
                return col
    if default_idx is not None and len(df.columns) > default_idx:
        return df.columns[default_idx]
    return None

def smart_load_csv(file):
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            if df.columns.str.contains('Unnamed').sum() > len(df.columns) * 0.3:
                for i in range(1, 20):
                    file.seek(0)
                    df = pd.read_csv(file, skiprows=i, encoding=enc)
                    if not df.columns.str.contains('Unnamed').all(): break
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except: continue
    return None

# --- [3. 상세 수주 팝업] ---
@st.dialog("상세 수주/수급 내역", width="large")
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.markdown(f"#### 🔍 분석 대상 품번 그룹")
    st.caption(f"{', '.join(group_ids)}")
    code_col = find_col_precise(df_bl, ['상품코드', '품번'], default_idx=5)
    qty_col = find_col_precise(df_bl, ['수주잔량', '잔량'], default_idx=30)
    group_upper = [g.upper() for g in group_ids]
    detail = df_bl[df_bl[code_col].astype(str).str.upper().str.strip().isin(group_upper)].copy()
    detail['clean_qty'] = clean_numeric(detail[qty_col])
    detail['dt_clean_popup'] = pd.to_datetime(detail.iloc[:, 24].astype(str).str.replace('.0',''), format='%Y%m%d', errors='coerce')
    detail = detail[(detail['clean_qty'] > 0) & (detail['dt_clean_popup'] >= cutoff_date)]
    if detail.empty:
        st.info("해당 품번으로 조건에 맞는 수주 데이터가 없습니다.")
        return
    st.dataframe(detail.sort_values('dt_clean_popup', ascending=True), use_container_width=True, hide_index=True)

# --- [4. 메인 분석 엔진 (캐싱 적용)] ---
# 데이터가 바뀌지 않으면 로딩바 없이 즉시 결과를 반환함
def run_simulation(data, start_date_val, freq_opt, exclude_months, search_query):
    df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
    today_dt = pd.Timestamp(datetime.now().date())
    base_dt = pd.Timestamp(start_date_val)
    cutoff_date = base_dt - relativedelta(months=exclude_months)

    # 마스터/데이터 정제 (로직 유지)
    it_code = find_col_precise(df_item, ['상품코드', '품번'], exclude_keywords=['대표'], default_idx=6)
    it_site = find_col_precise(df_item, ['최종생산지명', '생산지'], default_idx=12)
    it_prev = find_col_precise(df_item, ['이전상품코드'], default_idx=13)
    it_date = find_col_precise(df_item, ['생성일자'], default_idx=3)
    it_name = find_col_precise(df_item, ['상품명', '품명'], default_idx=1)

    master_proc = df_item.copy()
    master_proc['clean_date'] = parse_date_smart(master_proc[it_date])
    master_proc['key_u'] = master_proc[it_code].astype(str).str.upper().str.strip()
    master_proc = master_proc.sort_values(by=['key_u', 'clean_date'], ascending=[True, False])
    master_unique = master_proc.drop_duplicates(subset='key_u', keep='first')

    site_map = master_unique.set_index('key_u')[it_site].to_dict()
    prev_map = master_unique.set_index('key_u')[it_prev].to_dict()
    next_map = master_unique.set_index(master_unique[it_prev].astype(str).str.upper().str.strip())[it_code].to_dict()

    bl_code_col = find_col_precise(df_bl, ['상품코드', '품번'], default_idx=5)
    df_bl['clean_qty'] = clean_numeric(df_bl[find_col_precise(df_bl, ['수주잔량', '총예상수량'], default_idx=30)])
    df_bl['dt_clean'] = parse_date_smart(df_bl[find_col_precise(df_bl, ['납품예정일'], default_idx=24)])
    df_bl_filtered = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

    po_code_col = find_col_precise(df_po, ['품번', '상품코드'], default_idx=12)
    df_po['m_qty'] = clean_numeric(df_po[find_col_precise(df_po, ['PO잔량', '미선적'], default_idx=19)]) * 11.3378 

    def calc_arrival(row):
        pid_u = str(row[po_code_col]).upper().strip()
        site_v = str(row.get(find_col_precise(df_po, ['생산지명', '거래처'], default_idx=10), site_map.get(pid_u, 'ETC'))).upper()
        site_k = 'SR' if 'SR' in site_v else site_v[:2]
        lt = LT_CONFIG.get(site_k, LT_CONFIG.get(site_v[:2], {'total': 1, 'ship_days': 30}))
        p_dt = parse_date_smart(pd.Series([row.get(find_col_precise(df_po, ['생산예정일'], default_idx=28), np.nan)]))[0]
        if pd.notnull(p_dt): arrival = p_dt + pd.DateOffset(days=int(lt['ship_days']))
        else:
            b_dt = parse_date_smart(pd.Series([row.get(find_col_precise(df_po, ['PO일자', '발주일자'], default_idx=3), today_dt)]))[0]
            if pd.isna(b_dt): b_dt = today_dt
            arrival = b_dt + pd.DateOffset(months=int(lt['total']))
        if pd.isnull(arrival) or arrival < base_dt:
            arrival = today_dt + pd.DateOffset(days=int(lt['ship_days']))
            if arrival < base_dt: arrival = base_dt
        return arrival

    df_po['dt_arrival'] = df_po.apply(calc_arrival, axis=1)
    df_st['clean_qty'] = clean_numeric(df_st[find_col_precise(df_st, ['재고수량', '현재고'], default_idx=7)])

    freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
    date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
    time_labels = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range[:12]]

    target_ids = df_bl_filtered[df_bl_filtered['clean_qty'] > 0][bl_code_col].unique()
    matrix_rows, alert_list = [], []
    
    # --- 로딩 바는 분석이 실제로 필요할 때만 노출 ---
    progress_placeholder = st.empty()
    bar = progress_placeholder.progress(0, text="📊 데이터 분석 중...")
    
    for i, pid in enumerate(target_ids):
        pid_s = str(pid).strip(); pid_u = pid_s.upper()
        item_match = master_unique[master_unique['key_u'] == pid_u]
        p_name = str(item_match[it_name].iloc[0]) if not item_match.empty else "-"
        
        if search_query and (search_query.lower() not in p_name.lower() and search_query.lower() not in pid_s.lower()):
            continue
        
        bar.progress((i + 1) / len(target_ids), text=f"🔍 분석 중: {p_name[:15]}...")

        def clean_p(v):
            s = str(v).strip().upper()
            return s if s not in ["NAN", "NONE", "0", "-", ""] else ""
        p_id = clean_p(prev_map.get(pid_u, "")); n_id = clean_p(next_map.get(pid_u, ""))
        group = list(set([pid_u, p_id, n_id])); group = [g for g in group if g]

        site_name = str(site_map.get(pid_u, "ETC"))
        site_key = 'SR' if 'SR' in site_name.upper() else site_name[:2].upper()
        lt_total = LT_CONFIG.get(site_key, {'total': 0})['total']

        main_stk = df_st[df_st[find_col_precise(df_st, ['품번', '상품코드'], default_idx=7)].astype(str).str.upper().str.strip().isin(group)]['clean_qty'].sum()
        overdue_dem = df_bl_filtered[(df_bl_filtered[bl_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_bl_filtered['dt_clean'] < base_dt)]['clean_qty'].sum()
        running_inv = main_stk - overdue_dem
        
        d_row = {"No": i+1, "품명": p_name, "수주품번": pid_s, "본사재고": main_stk, "PO잔량(m)": df_po[df_po[po_code_col].astype(str).str.upper().str.strip().isin(group)]['m_qty'].sum(), "생산지": f"{site_key}({lt_total}M)", "구분": "소요량", "연계정보": f"이전:{p_id}" if p_id else "", "납기경과": overdue_dem, "group": group}
        p_row = {"No": i+1, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "입고량(PO)", "연계정보": "", "납기경과": 0, "group": group}
        s_row = {"No": i+1, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "예상재고", "연계정보": f"변경:{n_id}" if n_id else "", "납기경과": running_inv, "group": group}

        for j in range(12):
            start, end = date_range[j], date_range[j+1]
            m_dem = df_bl_filtered[(df_bl_filtered[bl_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_bl_filtered['dt_clean'] >= start) & (df_bl_filtered['dt_clean'] < end)]['clean_qty'].sum()
            m_sup = df_po[(df_po[po_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_po['dt_arrival'] >= start) & (df_po['dt_arrival'] < end)]['m_qty'].sum()
            running_inv = (running_inv + m_sup) - m_dem
            lbl = time_labels[j]
            d_row[lbl], p_row[lbl], s_row[lbl] = m_dem, m_sup, running_inv
            if running_inv < 0 and start < base_dt + pd.DateOffset(months=lt_total):
                alert_list.append({"품명": p_name, "품번": pid_s, "생산지": site_key, "LT": lt_total, "부족시점": lbl, "부족수량": abs(running_inv), "group": group})
        matrix_rows.extend([d_row, p_row, s_row])

    progress_placeholder.empty()
    return pd.DataFrame(matrix_rows), pd.DataFrame(alert_list), time_labels

# --- [5. 메인 UI 실행] ---
st.title("🚀 P·Forecast Stock Manager v6.7")

data = {}
if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            cols_text = "|".join(df.columns).upper()
            for k, v in RECOGNITION.items():
                if any(key in cols_text for key in v["keys"]):
                    data[k] = df; RECOGNITION[k]["found"] = True; break

if len(data) >= 5:
    # [v6.7 핵심] 세션 스테이트를 사용하여 로딩 횟수 최소화
    if 'sim_result' not in st.session_state or st.sidebar.button("♻️ 데이터 새로고침"):
        res, alerts, labels = run_simulation(data, start_date_val, freq_opt, exclude_months, search_query)
        st.session_state.sim_result = (res, alerts, labels)

    res_df, alert_df, time_labels = st.session_state.sim_result

    # 긴급 발주 대시보드
    st.subheader("🚨 수급 안정성 검토")
    if not alert_df.empty:
        alert_clean = alert_df.drop_duplicates(subset=['품번'], keep='first').copy()
        st.error(f"리드타임 내 재고 부족 예상 품목: {len(alert_clean)}건")
        
        def get_dday(row):
            deadline = pd.to_datetime(row['부족시점']) - pd.DateOffset(months=int(row['LT']))
            days = (deadline - pd.Timestamp(datetime.now().date())).days
            return f"D-{days}일" if days >= 0 else f"지남({abs(days)}일 전)"
        
        alert_clean['발주기한'] = alert_clean.apply(get_dday, axis=1)
        
        # [v6.7] 긴급 리스트 클릭 시 즉시 상세보기 활성화
        sel_alert = st.dataframe(
            alert_clean[['품명', '품번', '생산지', '부족시점', '부족수량', '발주기한']], 
            use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
        )
        if sel_alert.selection.rows:
            target = alert_clean.iloc[sel_alert.selection.rows[0]]
            # 별도 버튼 없이 바로 아래에 상세보기 버튼을 노출 (동선 단축)
            if st.button(f"🔍 {target['품번']} 수주 상세 보기 (팝업)", type="primary"):
                show_detail_popup(target['group'], data['backlog'], cutoff_date)
    else:
        st.success("안전: 리드타임 내 부족 품목이 없습니다.")

    # 메인 매트릭스
    st.subheader(f"📊 통합 수급 시뮬레이션")
    num_cols = ["본사재고", "PO잔량(m)", "납기경과"] + time_labels
    
    def style_fn(row):
        g_idx = (row.name // 3); bg = '#f9f9f9' if g_idx % 2 == 0 else '#ffffff'
        styles = [f'background-color: {bg}'] * len(row)
        for i, col in enumerate(row.index):
            if col == "구분": styles[i] = 'background-color: #e1f5fe; font-weight: bold'
            elif row['구분'] == "예상재고" and col in num_cols and row[col] < 0:
                styles[i] = 'background-color: #ff4b4b; color: white'
        return styles

    st_df = st.dataframe(
        res_df.style.apply(style_fn, axis=1).format({c: "{:,.0f}" for c in num_cols}, na_rep=""),
        use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row",
        column_order=["No", "품명", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels
    )
    
    if st_df.selection.rows:
        target = res_df.iloc[st_df.selection.rows[0] - (st_df.selection.rows[0] % 3)]
        if st.button(f"🔍 {str(target['수주품번']).strip()} 상세 내역 보기"):
            show_detail_popup(target['group'], data['backlog'], cutoff_date)
else:
    st.info("사이드바에 5종 파일을 모두 업로드해주세요.")
