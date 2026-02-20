import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- [1. 설정 및 리드타임 마스터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

# 리드타임 설정: SR(0) 등 변칙 코드 대응을 위해 'SR' 키워드 기준 관리
LT_CONFIG = {
    'SE': {'total': 6, 'ship_days': 90},
    'SR': {'total': 8, 'ship_days': 90},  # SR(0), SRL 모두 포함
    'SRL': {'total': 8, 'ship_days': 90},
    'SP': {'total': 8, 'ship_days': 90},
    'SH': {'total': 1, 'ship_days': 30},
    'KD': {'total': 2, 'ship_days': 30},
    'QZ': {'total': 2, 'ship_days': 30}
}

# --- [2. 지능형 유틸리티 함수] ---
def clean_numeric(series):
    if series.dtype == 'object':
        # 숫자, 마침표, 마이너스 기호 제외한 모든 문자 제거
        series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date_smart(series):
    """날짜 형식을 안전하게 변환 (오류 시 NaT 반환)"""
    s = series.astype(str).str.replace('.0', '', regex=False).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

def find_col_precise(df, keywords, exclude_keywords=None, default_idx=None):
    """
    키워드로 컬럼명을 찾되, 제외 키워드(예: '대표')가 포함된 열은 피함.
    품목정보 파일에서 '상품코드'와 '대표상품코드'를 구분하기 위함.
    """
    for k in keywords:
        for col in df.columns:
            col_upper = str(col).replace(" ", "").upper()
            # 메인 키워드가 포함되어 있고
            if k in col_upper:
                # 제외 키워드가 포함되어 있지 않아야 함
                if exclude_keywords:
                    if any(ex.upper() in col_upper for ex in exclude_keywords):
                        continue
                return col
    # 못 찾으면 기본 인덱스 활용
    if default_idx is not None and len(df.columns) > default_idx:
        return df.columns[default_idx]
    return None

def smart_load_csv(file):
    """v5.3/v6.3에서 검증된 안정적인 로딩 로직"""
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            # Unnamed가 많으면 헤더가 데이터 아래에 있다고 판단하여 스킵 탐색
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
@st.dialog("현장별 상세 수주 내역", width="large")
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.write(f"🔎 분석 대상 품번 그룹: {', '.join(group_ids)}")
    # 수주예정등록 파일에서 상품코드와 수량 컬럼 탐색
    code_col = find_col_precise(df_bl, ['상품코드', '품번'], default_idx=5)
    qty_col = find_col_precise(df_bl, ['수주잔량', '잔량'], default_idx=30)
    
    group_upper = [g.upper() for g in group_ids]
    detail = df_bl[df_bl[code_col].astype(str).str.upper().str.strip().isin(group_upper)].copy()
    detail['clean_qty'] = clean_numeric(detail[qty_col])
    
    # 날짜 컬럼(보통 인덱스 24) 안전하게 파싱
    date_col_idx = 24
    detail['dt_clean_popup'] = pd.to_datetime(detail.iloc[:, date_col_idx].astype(str).str.replace('.0',''), format='%Y%m%d', errors='coerce')
    detail = detail[(detail['clean_qty'] > 0) & (detail['dt_clean_popup'] >= cutoff_date)]
    
    if detail.empty:
        st.info("조건에 맞는 수주 데이터가 없습니다.")
        return
    st.dataframe(detail.sort_values('dt_clean_popup', ascending=True), use_container_width=True, hide_index=True)

# --- [4. 메인 UI] ---
st.title("🚀 P·Forecast Stock Manager v6.4")

# 파일 인식을 위한 핵심 키워드 (v5.3 기반 안정성 유지)
RECOGNITION = {
    "backlog": {"name": "수주예정(Demand)", "keys": ["수주잔량", "총예상수량"], "found": False},
    "po": {"name": "구매발주(PO)", "keys": ["PO잔량", "미선적"], "found": False},
    "stock": {"name": "현재고(Stock)", "keys": ["재고수량", "현재고액"], "found": False},
    "item": {"name": "품목정보(Master)", "keys": ["최종생산지명", "이전상품코드"], "found": False},
    "retail": {"name": "시판스펙(Retail)", "keys": ["출시예정", "4개월판매량"], "found": False}
}

with st.sidebar:
    st.header("⚙️ 분석 설정")
    # 기본 분석 시점: 다음달 1일
    default_start = (datetime.now().replace(day=1) + relativedelta(months=1))
    start_date_val = st.date_input("검토 시점(조회 시작일)", default_start)
    freq_opt = st.selectbox("집계 단위", ["주별", "월별", "분기별", "년도별"], index=1)
    exclude_months = st.slider("과거 수주 제외 (N개월)", 1, 36, 12)
    cutoff_date = pd.Timestamp(start_date_val) - relativedelta(months=exclude_months)
    st.markdown("---")
    search_query = st.text_input("🔍 품명/품번 키워드 검색", "")
    st.markdown("---")
    st.subheader("📁 파일 로드 상태")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)

# 데이터 로딩 실행
data = {}
if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            cols_text = "|".join(df.columns).upper()
            for k, v in RECOGNITION.items():
                if any(key in cols_text for key in v["keys"]):
                    data[k] = df
                    RECOGNITION[k]["found"] = True
                    break

# 사이드바 로드 상태 표시
with st.sidebar:
    for k, v in RECOGNITION.items():
        if v["found"]: st.success(f"✅ {v['name']}")
        else: st.warning(f"⏳ {v['name']}")

# --- [5. 메인 분석 로직] ---
if len(data) >= 5:
    with st.spinner('최신 마스터 정보 반영 및 수급 시뮬레이션 중...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
        today_dt = pd.Timestamp(datetime.now().date())
        base_dt = pd.Timestamp(start_date_val)

        # 1. 품목 마스터 정밀 구축 (v6.4 핵심: 최신 생성일자 우선)
        # '대표상품코드'를 피해서 '상품코드' 열을 정확히 탐색
        it_code = find_col_precise(df_item, ['상품코드', '품번'], exclude_keywords=['대표'], default_idx=6)
        it_site = find_col_precise(df_item, ['최종생산지명', '생산지'], default_idx=12)
        it_prev = find_col_precise(df_item, ['이전상품코드'], default_idx=13)
        it_chng = find_col_precise(df_item, ['변경상품코드'], default_idx=8)
        it_date = find_col_precise(df_item, ['생성일자'], default_idx=3)
        it_name = find_col_precise(df_item, ['상품명', '품명'], default_idx=1)

        # 마스터 데이터 전처리: 생성일자 기준 정렬 후 중복 제거
        master_proc = df_item.copy()
        master_proc['clean_date'] = parse_date_smart(master_proc[it_date])
        master_proc['key_u'] = master_proc[it_code].astype(str).str.upper().str.strip()
        # 생성일자 내림차순 정렬하여 가장 최신 데이터가 위로 오게 함
        master_proc = master_proc.sort_values(by=['key_u', 'clean_date'], ascending=[True, False])
        # 중복된 품번 중 가장 최신 것(첫 번째)만 남김
        master_unique = master_proc.drop_duplicates(subset='key_u', keep='first')

        site_map = master_unique.set_index('key_u')[it_site].to_dict()
        prev_map = master_unique.set_index('key_u')[it_prev].to_dict()
        # 이전코드로 현재코드를 찾는 역방향 맵 (체인 연결용)
        next_map = master_unique.set_index(master_unique[it_prev].astype(str).str.upper().str.strip())[it_code].to_dict()

        # 2. 각 소스 데이터 정제
        # 수주예정
        bl_code_col = find_col_precise(df_bl, ['상품코드', '품번'], default_idx=5)
        bl_qty_col = find_col_precise(df_bl, ['수주잔량', '총예상수량'], default_idx=30)
        bl_date_col = find_col_precise(df_bl, ['납품예정일'], default_idx=24)
        df_bl['clean_qty'] = clean_numeric(df_bl[bl_qty_col])
        df_bl['dt_clean'] = parse_date_smart(df_bl[bl_date_col])
        df_bl = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

        # PO (KG -> M 환산)
        po_code_col = find_col_precise(df_po, ['품번', '상품코드'], default_idx=12)
        po_qty_col = find_col_precise(df_po, ['PO잔량', '미선적'], default_idx=19)
        po_site_col = find_col_precise(df_po, ['생산지명', '거래처'], default_idx=10)
        po_prod_col = find_col_precise(df_po, ['생산예정일'], default_idx=28)
        po_date_col = find_col_precise(df_po, ['PO일자', '발주일자'], default_idx=3)
        df_po['m_qty'] = clean_numeric(df_po[po_qty_col]) * 11.3378 

        # [v6.4] 지능형 입고일 계산 (SR 인식 및 사각지대 전진 배치)
        def calc_arrival_v64(row):
            pid_u = str(row[po_code_col]).upper().strip()
            # PO파일에 없으면 마스터 파일(최신순 정렬됨)에서 생산지 조회
            site_v = str(row.get(po_site_col, site_map.get(pid_u, 'ETC'))).upper()
            
            # SR(0), SRL 등을 모두 'SR' 키워드로 통합 인식
            site_k = 'SR' if 'SR' in site_v else site_v[:2]
            lt = LT_CONFIG.get(site_k, LT_CONFIG.get(site_v[:2], {'total': 1, 'ship_days': 30}))
            
            p_dt = parse_date_smart(pd.Series([row.get(po_prod_col, np.nan)]))[0]
            if pd.notnull(p_dt):
                arrival = p_dt + pd.DateOffset(days=int(lt['ship_days']))
            else:
                b_dt = parse_date_smart(pd.Series([row.get(po_date_col, today_dt)]))[0]
                if pd.isna(b_dt): b_dt = today_dt
                arrival = b_dt + pd.DateOffset(months=int(lt['total']))
            
            # [수정] 조회 시작일(base_dt) 이전 물량은 '첫 분석달' 가용 물량으로 강제 전진 배치
            if pd.isnull(arrival) or arrival < base_dt:
                # 오늘 선적 지시 시 가장 빠른 도착일 계산
                arrival = today_dt + pd.DateOffset(days=int(lt['ship_days']))
                # 그래도 시작일보다 빠르면 시작일 당일로 맞춤
                if arrival < base_dt: arrival = base_dt
            return arrival

        df_po['dt_arrival'] = df_po.apply(calc_arrival_v64, axis=1)

        # 재고
        st_code_col = find_col_precise(df_st, ['품번', '상품코드'], default_idx=7)
        st_qty_col = find_col_precise(df_st, ['재고수량', '현재고'], default_idx=17)
        df_st['clean_qty'] = clean_numeric(df_st[st_qty_col])

        # 3. 타임라인 매트릭스 생성
        freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
        date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
        time_labels = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range[:12]]

        target_ids = df_bl[df_bl['clean_qty'] > 0][bl_code_col].unique()
        matrix_rows, alert_list = [], []
        idx_no = 1

        for pid in target_ids:
            pid_s = str(pid).strip(); pid_u = pid_s.upper()
            item_match = master_unique[master_unique['key_u'] == pid_u]
            p_name = str(item_match[it_name].iloc[0]) if not item_match.empty else "-"
            
            if search_query and (search_query.lower() not in p_name.lower() and search_query.lower() not in pid_s.lower()):
                continue

            # 품번 연계 그룹핑 및 'NAN' 클리닝
            def clean_pid_str(v):
                s = str(v).strip().upper()
                return s if s not in ["NAN", "NONE", "0", "-", ""] else ""
            
            p_id = clean_pid_str(prev_map.get(pid_u, ""))
            n_id = clean_pid_str(next_map.get(pid_u, ""))
            group = list(set([pid_u, p_id, n_id])); group = [g for g in group if g]

            # 생산지 및 LT 정보 (v6.4: SR 포함 여부로 판별)
            site_name = str(site_map.get(pid_u, "ETC"))
            site_key = 'SR' if 'SR' in site_name.upper() else site_name[:2].upper()
            lt_total = LT_CONFIG.get(site_key, {'total': 0})['total']
            
            # 시판스펙 여부
            is_retail = " 🏷️" if any(str(g).upper() in df_retail.iloc[:, 8].astype(str).str.upper().values for g in group) else ""

            # 기초 재고 수지 계산
            main_stk = df_st[df_st[st_code_col].astype(str).str.upper().str.strip().isin(group)]['clean_qty'].sum()
            overdue_dem = df_bl[(df_bl[bl_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_bl['dt_clean'] < base_dt)]['clean_qty'].sum()
            running_inv = main_stk - overdue_dem
            
            # 3행 1세트 데이터 구조화
            d_row = {"No": idx_no, "품명": p_name, "수주품번": pid_s + is_retail, "본사재고": main_stk, "PO잔량(m)": df_po[df_po[po_code_col].astype(str).str.upper().str.strip().isin(group)]['m_qty'].sum(), "생산지": f"{site_key}({lt_total}M)", "구분": "소요량", "연계정보": f"이전:{p_id}" if p_id else "", "납기경과": overdue_dem, "group": group}
            p_row = {"No": idx_no, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "입고량(PO)", "연계정보": "", "납기경과": 0, "group": group}
            s_row = {"No": idx_no, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "예상재고", "연계정보": f"변경:{n_id}" if n_id else "", "납기경과": running_inv, "group": group}

            for i in range(12):
                start, end = date_range[i], date_range[i+1]
                m_dem = df_bl[(df_bl[bl_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_bl['dt_clean'] >= start) & (df_bl['dt_clean'] < end)]['clean_qty'].sum()
                m_sup = df_po[(df_po[po_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_po['dt_arrival'] >= start) & (df_po['dt_arrival'] < end)]['m_qty'].sum()
                running_inv = (running_inv + m_sup) - m_dem
                
                label = time_labels[i]
                d_row[label], p_row[label], s_row[label] = m_dem, m_sup, running_inv
                
                # 리드타임 내 재고 부족 시 알림 리스트 추가
                if running_inv < 0 and start < base_dt + pd.DateOffset(months=lt_total):
                    alert_list.append({"품명": p_name, "품번": pid_s, "부족시점": label, "부족수량": abs(running_inv)})

            matrix_rows.extend([d_row, p_row, s_row]); idx_no += 1

    # 결과 매트릭스 출력
    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        num_cols = ["본사재고", "PO잔량(m)", "납기경과"] + time_labels
        for c in num_cols: res_df[c] = pd.to_numeric(res_df[c], errors='coerce')

        def style_fn(row):
            g_idx = (row.name // 3); bg = '#f9f9f9' if g_idx % 2 == 0 else '#ffffff'
            styles = [f'background-color: {bg}'] * len(row)
            for i, col in enumerate(row.index):
                if col == "구분": styles[i] = 'background-color: #e1f5fe; font-weight: bold'
                elif row['구분'] == "예상재고" and col in num_cols:
                    if row[col] < 0: styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        st.subheader(f"📊 통합 수급 시뮬레이션 ({freq_opt})")
        st_df = st.dataframe(
            res_df.style.apply(style_fn, axis=1).format({c: "{:,.0f}" for c in num_cols}, na_rep=""),
            use_container_width=True, hide_index=True,
            column_order=["No", "품명", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels,
            on_select="rerun", selection_mode="single-row"
        )
        
        # 선택된 품번 상세 수주 내역 팝업 연동
        if st_df.selection.rows:
            s_idx = st_df.selection.rows[0]
            target = res_df.iloc[s_idx - (s_idx % 3)]
            if st.button(f"🔍 {str(target['수주품번']).replace('🏷️','').strip()} 상세 보기"):
                show_detail_popup(target['group'], df_bl, cutoff_date)
else:
    st.info("사이드바에 5종 파일을 모두 업로드해주세요.")
