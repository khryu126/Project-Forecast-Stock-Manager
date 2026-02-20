import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- [1. 설정 및 리드타임 마스터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_MASTER = {
    'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8
}

# --- [2. 데이터 정제 유틸리티 (에러 방지용)] ---
def clean_numeric_data(series):
    """문자열 숫자(콤마 포함)를 실수형으로 변환하고, 빈 값은 0으로 채움"""
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.strip()
        # 빈 문자열('')을 NaN으로 바꾼 후 숫자로 변환
        series = series.replace('', np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date(series):
    """다양한 날짜 형식을 datetime 객체로 표준화"""
    return pd.to_datetime(series, errors='coerce')

def get_pattern_group(df_item, target_id):
    """품번 이원화(Code Chain) 추적: 연계된 모든 품번 리스트 반환"""
    target_id = str(target_id).strip()
    related = {target_id}
    
    # 품목정보에서 이전/변경 코드 연결고리 탐색
    links = df_item[(df_item['상품코드'] == target_id) | 
                    (df_item['이전상품코드'] == target_id) | 
                    (df_item['변경상품코드'] == target_id)]
    
    for _, row in links.iterrows():
        for col in ['상품코드', '이전상품코드', '변경상품코드']:
            val = str(row[col]).strip()
            if val and val.lower() != 'nan' and val != '0':
                related.add(val)
    return list(related)

# --- [3. 상세 팝업창 로직 (Drill-down)] ---
@st.dialog("상세 수주 및 납기 현황")
def show_detail_popup(group_ids, df_bl):
    st.write(f"🔎 연계 품번 그룹: {', '.join(group_ids)}")
    
    detail = df_bl[df_bl['상품코드'].isin(group_ids)].copy()
    if detail.empty:
        st.info("현재 수주 잔량이 없습니다.")
        return

    today = datetime.now()
    # 납기 상태 구분 로직
    detail['상태'] = detail['납품예정일'].apply(lambda x: "⚠️ 납기경과" if x < today else "정상")
    
    # 필요한 컬럼만 출력
    cols = ['상태', '현장명', '건설사', '수주잔량', '납품예정일', '메모']
    st.dataframe(detail[cols].sort_values('납품예정일'), use_container_width=True, hide_index=True)
    st.warning("납기경과 물량은 실제 지연인지 전산상 유령 잔량인지 현업 확인이 필요합니다.")

# --- [4. 메인 앱 UI] ---
st.title("📦 P·Forecast Stock Manager")
st.caption("건설 특판 모양지 통합 오더 및 재고 수지 관리 시스템")

# 파일 업로드 섹션
uploaded_files = st.sidebar.file_uploader("5종의 CSV 파일을 모두 선택하세요", accept_multiple_files=True)

data = {}
if uploaded_files:
    for f in uploaded_files:
        df = pd.read_csv(f).rename(columns=lambda x: x.strip())
        # 컬럼명을 합쳐서 어떤 파일인지 자동 판별
        cols_text = "".join(df.columns)
        if "수주잔량" in cols_text: data['backlog'] = df
        elif "PO" in cols_text or "미선적" in cols_text: data['po'] = df
        elif "현재고" in cols_text or "재고수량" in cols_text: data['stock'] = df
        elif "시판" in cols_text: data['retail'] = df
        elif "최종생산지" in cols_text: data['item'] = df

# 데이터 처리 시작
if len(data) >= 5:
    # 데이터 표준화 작업 (Bulletproof)
    df_item = data['item']
    df_bl = data['backlog']
    df_po = data['po']
    df_st = data['stock']
    df_retail = data['retail']

    # 모든 숫자형 컬럼 강제 정제 (PyArrow 에러 원천 차단)
    for df in [df_bl, df_po, df_st, df_retail]:
        for col in df.columns:
            if any(k in col for k in ['잔량', '수량', '현재고', 'weight', '평량']):
                df[col] = clean_numeric_data(df[col])

    df_bl['납품예정일'] = parse_date(df_bl['납품예정일'])
    df_po['입고요청일'] = parse_date(df_po['입고요청일'])

    # 타임라인 설정 (오늘부터 12개월)
    today_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_range = [today_start + pd.DateOffset(months=i) for i in range(12)]
    month_cols = [m.strftime('%Y-%m') for m in month_range]

    # 분석 대상 품번 추출
    target_ids = df_bl[df_bl['수주잔량'] > 0]['상품코드'].unique()
    
    matrix_rows = []
    processed_groups = set()

    for pid in target_ids:
        # 1. 품번 그룹화 (Code Chain)
        group = sorted(get_pattern_group(df_item, pid))
        group_key = tuple(group)
        if group_key in processed_groups: continue
        processed_groups.add(group_key)

        # 2. 기초 정보 및 태그
        item_info = df_item[df_item['상품코드'].isin(group)].iloc[0] if not df_item[df_item['상품코드'].isin(group)].empty else {}
        site_code = str(item_info.get('최종생산지명', 'ETC'))
        lt = LT_MASTER.get(site_code, 0)
        
        is_retail = "🏷️" if any(str(g) in df_retail['품번'].astype(str).values for g in group) else ""
        has_chain = "🔄" if len(group) > 1 else ""
        
        # 3. 재고 수지 계산
        curr_stock = df_st[df_st['품번'].isin(group)]['재고수량'].sum()
        overdue_demand = df_bl[(df_bl['상품코드'].isin(group)) & (df_bl['납품예정일'] < today_start)]['수주잔량'].sum()
        has_overdue = "⚠️" if overdue_demand > 0 else ""
        
        running_inv = curr_stock - overdue_demand
        row_demand = {"납기경과": overdue_demand}
        row_stock = {"납기경과": running_inv}

        for m_date in month_range:
            m_str = m_date.strftime('%Y-%m')
            # 소요량
            m_dem = df_bl[(df_bl['상품코드'].isin(group)) & 
                          (df_bl['납품예정일'] >= m_date) & 
                          (df_bl['납품예정일'] < m_date + pd.DateOffset(months=1))]['수주잔량'].sum()
            
            # 입고량 (PO 환산)
            m_po_data = df_po[(df_po['품번'].isin(group)) & 
                              (df_po['입고요청일'] >= m_date) & 
                              (df_po['입고요청일'] < m_date + pd.DateOffset(months=1))]
            
            m_sup = 0
            for _, r in m_po_data.iterrows():
                bw = r.get('B/P weight', 70)
                bw = 70 if bw == 0 else bw
                m_sup += (r.get('PO잔량(미선적)', 0) * 1000) / (bw * 1.26)
            
            running_inv = (running_inv + m_sup) - m_dem
            row_demand[m_str] = round(m_dem, 0)
            row_stock[m_str] = round(running_inv, 0)

        # 4. 결과 행 추가
        title = f"{pid} {is_retail}{has_chain}{has_overdue}"
        common_info = {"품번": title, "생산지(LT)": f"{site_code}({lt}M)", "group": group}
        matrix_rows.append({**common_info, "구분": "소요량(m)", **row_demand})
        matrix_rows.append({**common_info, "구분": "예상재고(m)", **row_stock})

    # 최종 테이블 생성
    result_df = pd.DataFrame(matrix_rows)

    # 스타일 적용 (리드타임 기반 알람)
    def style_stock(row):
        styles = [''] * len(row)
        if row['구분'] == "예상재고(m)":
            lt_val = int(row['생산지(LT)'].split('(')[1].replace('M)', ''))
            for i, col in enumerate(row.index):
                if col == "납기경과" and row[col] < 0:
                    styles[i] = 'background-color: #9e0000; color: white' # 강제 경고
                elif '-' in col and row[col] < 0:
                    col_dt = datetime.strptime(col, '%Y-%m')
                    limit_dt = today_start + pd.DateOffset(months=lt_val)
                    if col_dt <= limit_dt:
                        styles[i] = 'background-color: #ff4b4b; color: white' # 리드타임 내 고갈
                    else:
                        styles[i] = 'background-color: #ffeb3b; color: black' # 리드타임 외 고갈
        return styles

    st.subheader("📊 통합 수급 분석 매트릭스")
    st.info("💡 아래 표에서 행을 클릭한 뒤 [상세보기] 버튼을 누르면 현장별 납기 정보를 확인할 수 있습니다.")

    # 테이블 출력
    selection = st.dataframe(
        result_df.style.apply(style_stock, axis=1),
        use_container_width=True,
        hide_index=True,
        column_order=["품번", "생산지(LT)", "구분", "납기경과"] + month_cols,
        on_select="rerun",
        selection_mode="single_row"
    )

    # 상세보기 버튼 (팝업 호출)
    if selection.selection.rows:
        sel_idx = selection.selection.rows[0]
        sel_group = result_df.iloc[sel_idx]['group']
        if st.button(f"🔍 {result_df.iloc[sel_idx]['품번']} 현장별 상세 내역 보기"):
            show_detail_popup(sel_group, df_bl)

else:
    st.warning("분석을 위해 왼쪽 사이드바에서 5종의 CSV 파일을 업로드해 주세요.")
