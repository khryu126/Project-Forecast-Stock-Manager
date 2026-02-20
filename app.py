import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- [1. 설정 및 리드타임 마스터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

# 유럽은 선적 리드타임 3개월로 일괄 적용
LT_CONFIG = {
    'SE': {'total': 6, 'ship': 3},   # 독일
    'SRL': {'total': 8, 'ship': 3},  # 이태리
    'SP': {'total': 8, 'ship': 3},   # 폴란드
    'SH': {'total': 1, 'ship': 0.5}, # 상해
    'KD': {'total': 2, 'ship': 1},   # 중국
    'QZ': {'total': 2, 'ship': 1}    # 광저우
}

# --- [2. 데이터 정제 유틸리티] ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None', 'nan ', ' nan'], np.nan)
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

# --- [3. 상세 팝업창 (크기 및 필터 고도화)] ---
@st.dialog("현장별 상세 수주 내역", width="large")
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.write(f"🔎 분석 대상 품번 그룹: {', '.join(group_ids)}")
    code_col = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
    
    # 필터: 수주잔량 > 0 이고 설정된 제외기간 이후 데이터만
    detail = df_bl[df_bl[code_col].astype(str).isin(group_ids)].copy()
    detail = detail[(detail['clean_qty'] > 0) & (detail['dt_clean'] >= cutoff_date)]
    
    if detail.empty:
        st.info("조건에 맞는 수주 데이터가 없습니다.")
        return

    # 납기 가장 빠른 현장 순 정렬
    st.dataframe(detail.sort_values('dt_clean', ascending=True), use_container_width=True, hide_index=True)

# --- [4. 메인 UI 및 파일 상태창] ---
st.title("🚀 P·Forecast Stock Manager v4.8")

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
    search_query = st.text_input("🔍 품명/품번 키워드 검색", "")
    
    st.markdown("---")
    st.subheader("📁 파일 로드 상태")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)

# 데이터 로드
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

# 사이드바 상태 표시 (대기중 -> 완료)
with st.sidebar:
    for k, v in RECOGNITION.items():
        if v["found"]: st.success(f"✅ {v['name']} (업로드 완료)")
        else: st.warning(f"⏳ {v['name']} (업로드 대기중)")

# --- [5. 메인 분석 로직] ---
if len(data) >= 5:
    with st.spinner('데이터를 정밀 분석 중입니다...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
        
        # 1. 수주/재고 정제
        bl_code = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
        df_bl['clean_qty'] = clean_numeric(df_bl['수주잔량'])
        df_bl['dt_clean'] = parse_date_smart(df_bl['납품예정일' if '납품예정일' in df_bl.columns else df_bl.columns[24]])
        df_bl = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

        # 2. PO 데이터 선적 LT 반영 로직
        po_code = '품번' if '품번' in df_po.columns else df_po.columns[12]
        df_po['clean_qty'] = clean_numeric(df_po['PO잔량(미선적)'])
        def calculate_arrival(row):
            # 생산예정일 -> 입고요청일 -> PO일자 순 추적
            target_dt = parse_date_smart(pd.Series([row.get('생산예정일', np.nan)]))[0]
            if pd.isna(target_dt):
                target_dt = parse_date_smart(pd.Series([row.get('입고요청일', row.get('PO일자', np.nan))]))[0]
            
            site = str(row.get('생산지명', ''))[:2].upper()
            ship_lt = LT_CONFIG.get(site, {'ship': 0})['ship']
            return target_dt + relativedelta(months=int(ship_lt)) if pd.notnull(target_dt) else pd.NaT

        df_po['dt_arrival'] = df_po.apply(calculate_arrival, axis=1)

        st_code = '품번' if '품번' in df_st.columns else df_st.columns[7]
        df_st['clean_qty'] = clean_numeric(df_st['재고수량' if '재고수량' in df_st.columns else df_st.columns[17]])

        # 3. 기간 설정 및 매트릭스 생성
        base_dt = pd.Timestamp(start_date)
        freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
        date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
        time_labels = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range[:12]]

        target_ids = df_bl[df_bl['clean_qty'] > 0][bl_code].unique()
        matrix_rows, alert_list = [], []
        idx_no = 1

        for pid in target_ids:
            pid_s = str(pid)
            item_match = df_item[df_item['상품코드'].astype(str) == pid_s]
            p_name = str(item_match['상품명'].iloc[0]) if not item_match.empty else "-"
            
            if search_query and (search_query.lower() not in p_name.lower() and search_query.lower() not in pid_s.lower()):
                continue

            # 연계 품번 및 생산지 정보
            prev = str(item_match['이전상품코드'].iloc[0]) if not item_match.empty else ""
            chng = str(item_match['변경상품코드'].iloc[0]) if not item_match.empty else ""
            prev = prev if prev not in ["nan", "0", "-"] else ""
            chng = chng if chng not in ["nan", "0", "-"] else ""

            def get_site_lt(code):
                if not code: return ""
                m = df_item[df_item['상품코드'].astype(str) == code]
                if not m.empty:
                    s = str(m['최종생산지명'].iloc[0])[:2]
                    l = LT_CONFIG.get(s.upper(), {'total': 0})['total']
                    return f"({s}/{l}M)"
                return ""

            group = [g for g in [pid_s, prev, chng] if g]
            site_raw = str(item_match['최종생산지명'].iloc[0]) if not item_match.empty else "ETC"
            lt_total = LT_CONFIG.get(site_raw[:2].upper(), {'total': 0})['total']
            is_retail = " 🏷️" if any(str(g) in df_retail.iloc[:, 8].astype(str).values for g in group) else ""

            # 재고 수지 계산
            main_stk = df_st[df_st[st_code].astype(str).isin(group)]['clean_qty'].sum()
            po_kg = df_po[df_po[po_code].astype(str).isin(group)]['clean_qty'].sum()
            po_m = (po_kg * 1000) / (70 * 1.26)

            overdue_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt_clean'] < base_dt)]['clean_qty'].sum()
            running_inv = main_stk - overdue_dem
            d_vals, s_vals = {"납기경과": overdue_dem}, {"납기경과": running_inv}

            for i in range(12):
                start, end = date_range[i], date_range[i+1]
                m_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt_clean'] >= start) & (df_bl['dt_clean'] < end)]['clean_qty'].sum()
                m_po_rows = df_po[(df_po[po_code].astype(str).isin(group)) & (df_po['dt_arrival'] >= start) & (df_po['dt_arrival'] < end)]
                m_sup = sum([(r['clean_qty'] * 1000) / (70 * 1.26) for _, r in m_po_rows.iterrows()])
                
                running_inv = (running_inv + m_sup) - m_dem
                d_vals[time_labels[i]], s_vals[time_labels[i]] = round(m_dem, 0), round(running_inv, 0)
                
                # 긴급 발주 알람 판단
                if running_inv < 0 and start < base_dt + relativedelta(months=lt_total):
                    alert_list.append({"품명": p_name, "품번": pid_s, "부족시점": time_labels[i], "부족수량": round(abs(running_inv), 0)})

            common = {"No": idx_no, "품명": p_name, "수주품번": pid_s + is_retail, "본사재고": main_stk, "PO잔량(m)": po_m, "생산지": f"{site_raw[:2]}({lt_total}M)", "group": group}
            matrix_rows.append({**common, "구분": "소요량", "연계정보": f"이전:{prev} {get_site_lt(prev)}" if prev else "", **d_vals})
            matrix_rows.append({"No": idx_no, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "group": group, "구분": "예상재고", "연계정보": f"변경:{chng} {get_site_lt(chng)}" if chng else "", **s_vals})
            idx_no += 1

    # [6. 긴급 발주 버튼 및 요약 표]
    if alert_list:
        if st.button(f"⚠️ 긴급 발주 검토 대상 보기 ({len(pd.DataFrame(alert_list)['품번'].unique())}건)"):
            st.error("리드타임 이내 재고 고갈이 예상되는 품목 리스트입니다.")
            sum_df = pd.DataFrame(alert_list).drop_duplicates(subset=['품번'], keep='first')
            st.table(sum_df.style.format({"부족수량": "{:,.0f}"}))

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        def style_fn(row):
            g_idx = (row.name // 2)
            bg = '#f5f5f5' if g_idx % 2 == 0 else '#ffffff'
            styles = [f'background-color: {bg}'] * len(row)
            for i, col in enumerate(row.index):
                if col == "구분": styles[i] = 'background-color: #e1f5fe; font-weight: bold'
                elif row['구분'] == "예상재고" and (col == "납기경과" or col in time_labels):
                    if isinstance(row[col], (int, float)) and row[col] < 0:
                        styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        st.subheader(f"📊 수급 분석 매트릭스 ({freq_opt} 합산)")
        # 소수점 제거 및 콤마 포맷
        fmt_dict = {col: "{:,.0f}" for col in ["본사재고", "PO잔량(m)", "납기경과"] + time_labels}
        st_df = st.dataframe(
            res_df.style.apply(style_fn, axis=1).format(fmt_dict, na_rep=""),
            use_container_width=True, hide_index=True,
            column_order=["No", "품명", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels,
            on_select="rerun", selection_mode="single-row"
        )
        if st_df.selection.rows:
            s_idx = st_df.selection.rows[0]
            target = res_df.iloc[s_idx if res_df.iloc[s_idx]['수주품번'] != '' else s_idx-1]
            if st.button(f"🔍 {target['수주품번'].replace('🏷️','')} 상세 보기"):
                show_detail_popup(target['group'], df_bl, cutoff_date)
else:
    st.info("사이드바에 5종 파일을 모두 업로드해주세요.")
