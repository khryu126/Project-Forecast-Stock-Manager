import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- [1. 기본 설정 및 마스터 데이터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

# 생산지별 리드타임(LT) 설정
LT_MASTER = {
    'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8
}

# --- [2. 데이터 정제 및 로드 유틸리티] ---
def clean_numeric(series):
    """문자열 숫자(콤마 포함)를 실수형으로 변환 및 결측치 0 처리"""
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date(series):
    """다양한 날짜 형식을 datetime 객체로 변환"""
    return pd.to_datetime(series, errors='coerce')

def smart_load_csv(file):
    """빈 줄 건너뛰기 및 인코딩 자동 감지 기능이 포함된 CSV 로더"""
    try:
        encodings = ['utf-8', 'cp949', 'euc-kr']
        for enc in encodings:
            try:
                file.seek(0)
                # 일단 읽어보고 컬럼이 유효한지 확인
                df = pd.read_csv(file, encoding=enc)
                # 만약 첫 줄이 데이터가 아니면(Unnamed가 많으면) 한 줄씩 건너뛰며 재시도
                if df.columns.str.contains('Unnamed').sum() > len(df.columns) * 0.5:
                    for i in range(1, 5):
                        file.seek(0)
                        df = pd.read_csv(file, skiprows=i, encoding=enc)
                        if not df.columns.str.contains('Unnamed').all():
                            break
                return df
            except UnicodeDecodeError:
                continue
        return None
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None

def get_pattern_group(df_item, target_id):
    """실린더 이전에 따른 이전/변경 품번 연계 추적 (Code Chain)"""
    target_id = str(target_id).strip()
    related = {target_id}
    if df_item is not None:
        # 해당 품번이 포함된 모든 행 찾기
        links = df_item[(df_item['상품코드'] == target_id) | 
                        (df_item.get('이전상품코드') == target_id) | 
                        (df_item.get('변경상품코드') == target_id)]
        for _, row in links.iterrows():
            for col in ['상품코드', '이전상품코드', '변경상품코드']:
                if col in df_item.columns:
                    val = str(row[col]).strip()
                    if val and val.lower() != 'nan' and val != '0':
                        related.add(val)
    return list(related)

# --- [3. 상세 팝업창 (Drill-down)] ---
@st.dialog("현장별 수주 상세 내역 (유령잔량 확인용)")
def show_detail_dialog(group_ids, df_bl):
    st.write(f"🔍 분석 대상 품번: {', '.join(group_ids)}")
    detail = df_bl[df_bl['상품코드'].isin(group_ids)].copy()
    
    if detail.empty:
        st.info("수주 데이터가 없습니다.")
        return

    today = datetime.now()
    detail['상태'] = detail['납품예정일'].apply(lambda x: "⚠️ 납기경과" if pd.notnull(x) and x < today else "정상")
    
    # 주요 정보 위주로 표시
    cols = ['상태', '현장명', '건설사', '수주잔량', '납품예정일', '메모']
    actual_cols = [c for c in cols if c in detail.columns]
    st.dataframe(detail[actual_cols].sort_values('납품예정일'), use_container_width=True, hide_index=True)
    st.caption("※ 납기경과 물량은 실제 지연인지 전산 유령 데이터인지 현업 확인이 필요합니다.")

# --- [4. 메인 UI 및 데이터 로직] ---
st.title("📦 P·Forecast Stock Manager")
st.caption("건설 특판 모양지 통합 오더 및 재고 수지 관리 시스템")

uploaded_files = st.sidebar.file_uploader("5종의 CSV 파일을 업로드하세요", accept_multiple_files=True)

data = {}
file_map = {
    "backlog": {"name": "수주예정(Demand)", "key": "수주잔량", "loaded": False},
    "po": {"name": "구매발주(PO)", "key": "PO잔량", "loaded": False},
    "stock": {"name": "현재고(Stock)", "key": "재고수량", "loaded": False},
    "item": {"name": "품목정보(Master)", "key": "최종생산지", "loaded": False},
    "retail": {"name": "시판스펙(Retail)", "key": "출시예정", "loaded": False}
}

if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            cols_text = "".join(df.columns)
            
            # 키워드 매칭을 통한 파일 식별
            if file_map["backlog"]["key"] in cols_text: 
                data['backlog'] = df; file_map["backlog"]["loaded"] = True
            elif file_map["po"]["key"] in cols_text: 
                data['po'] = df; file_map["po"]["loaded"] = True
            elif file_map["stock"]["key"] in cols_text or "현재고" in cols_text: 
                data['stock'] = df; file_map["stock"]["loaded"] = True
            elif file_map["item"]["key"] in cols_text or "상품명" in cols_text: 
                data['item'] = df; file_map["item"]["loaded"] = True
            elif file_map["retail"]["key"] in cols_text or "시판" in cols_text: 
                data['retail'] = df; file_map["retail"]["loaded"] = True

# 사이드바 상태 표시
st.sidebar.markdown("---")
st.sidebar.subheader("📁 데이터 로드 상태")
for k, v in file_map.items():
    if v["loaded"]: st.sidebar.success(f"✅ {v['name']}")
    else: st.sidebar.error(f"❌ {v['name']} (미인식)")

# 메인 분석 로직
if len(data) >= 5:
    df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']

    # 1. 숫자 및 날짜 데이터 정제
    for df in [df_bl, df_po, df_st, df_retail]:
        for col in df.columns:
            if any(k in col for k in ['잔량', '수량', '현재고', 'weight', '평량']):
                df[col] = clean_numeric(df[col])
    
    df_bl['납품예정일'] = parse_date(df_bl['납품예정일'])
    df_po['입고요청일'] = parse_date(df_po.get('입고요청일', df_po.get('PO일자')))

    # 2. 타임라인 설정
    today_base = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = [today_base + pd.DateOffset(months=i) for i in range(12)]
    month_cols = [m.strftime('%Y-%m') for m in months]

    # 3. 분석 대상 품번 (수주잔량 > 0)
    target_ids = df_bl[df_bl['수주잔량'] > 0]['상품코드'].unique()
    matrix_rows = []
    processed_groups = set()

    for pid in target_ids:
        group = sorted(get_pattern_group(df_item, pid))
        group_key = tuple(group)
        if group_key in processed_groups: continue
        processed_groups.add(group_key)

        # 기초 정보 및 태그
        item_info = df_item[df_item['상품코드'].isin(group)].iloc[0] if not df_item[df_item['상품코드'].isin(group)].empty else {}
        site_code = str(item_info.get('최종생산지명', 'ETC'))
        lt = LT_MASTER.get(site_code, 0)
        
        is_retail = "🏷️" if any(str(g) in df_retail.iloc[:, 8].astype(str).values for g in group) else ""
        has_chain = "🔄" if len(group) > 1 else ""
        
        # 재고 수지 계산
        # 현재고 합산
        total_curr_stock = df_st[df_st.get('품번', df_st.columns[7]).isin(group)]['재고수량'].sum()
        # 납기경과 소요 합산
        overdue_demand = df_bl[(df_bl['상품코드'].isin(group)) & (df_bl['납품예정일'] < today_base)]['수주잔량'].sum()
        
        running_inv = total_curr_stock - overdue_demand
        row_dem = {"납기경과": overdue_demand}
        row_stk = {"납기경과": running_inv}

        for m_date in months:
            m_str = m_date.strftime('%Y-%m')
            # 해당 월 소요
            m_dem_val = df_bl[(df_bl['상품코드'].isin(group)) & (df_bl['납품예정일'] >= m_date) & (df_bl['납품예정일'] < m_date + pd.DateOffset(months=1))]['수주잔량'].sum()
            
            # 해당 월 입고 (PO kg -> m 환산)
            m_po_data = df_po[(df_po.get('품번', df_po.columns[12]).isin(group)) & (df_po['입고요청일'] >= m_date) & (df_po['입고요청일'] < m_date + pd.DateOffset(months=1))]
            m_sup_val = 0
            for _, r in m_po_data.iterrows():
                bw = clean_numeric(pd.Series([r.get('B/P weight', 70)]))[0]
                m_sup_val += (clean_numeric(pd.Series([r.get('PO잔량(미선적)', 0)]))[0] * 1000) / ((bw if bw > 0 else 70) * 1.26)
            
            running_inv = (running_inv + m_sup_val) - m_dem_val
            row_dem[m_str] = round(m_dem_val, 0)
            row_stk[m_str] = round(running_inv, 0)

        title = f"{pid} {is_retail}{has_chain}{'⚠️' if overdue_demand > 0 else ''}"
        common = {"품번": title, "생산지(LT)": f"{site_code}({lt}M)", "group": group}
        matrix_rows.append({**common, "구분": "소요량(m)", **row_dem})
        matrix_rows.append({**common, "구분": "예상재고(m)", **row_stk})

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        
        def style_matrix(row):
            styles = [''] * len(row)
            if row['구분'] == "예상재고(m)":
                lt_val = int(row['생산지(LT)'].split('(')[1].replace('M)', ''))
                for i, col in enumerate(row.index):
                    if col == "납기경과" and row[col] < 0:
                        styles[i] = 'background-color: #9e0000; color: white'
                    elif '-' in col and row[col] < 0:
                        col_dt = datetime.strptime(col, '%Y-%m')
                        limit_dt = today_base + pd.DateOffset(months=lt_val)
                        if col_dt <= limit_dt: styles[i] = 'background-color: #ff4b4b; color: white'
                        else: styles[i] = 'background-color: #ffeb3b; color: black'
            return styles

        st.subheader("📊 통합 수급 분석 매트릭스")
        
        selection = st.dataframe(
            res_df.style.apply(style_matrix, axis=1),
            use_container_width=True, hide_index=True,
            column_order=["품번", "생산지(LT)", "구분", "납기경과"] + month_cols,
            on_select="rerun", selection_mode="single_row"
        )

        if selection.selection.rows:
            sel_idx = selection.selection.rows[0]
            if st.button(f"🔍 {res_df.iloc[sel_idx]['품번']} 상세 현장 정보 보기"):
                show_detail_dialog(res_df.iloc[sel_idx]['group'], df_bl)
else:
    st.info("사이드바에 5종의 파일을 모두 업로드하면 분석이 시작됩니다.")
