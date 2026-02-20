import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- [1. 기본 설정] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_MASTER = {'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8}

# --- [2. 데이터 정제 유틸리티] ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date(series):
    return pd.to_datetime(series, errors='coerce')

def smart_load_csv(file):
    encodings = ['cp949', 'utf-8-sig', 'utf-8', 'euc-kr']
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            # 수주예정등록 등 빈 줄이 있는 경우 처리
            if df.columns.str.contains('Unnamed').sum() > len(df.columns) * 0.4:
                for i in range(1, 5):
                    file.seek(0)
                    df = pd.read_csv(file, skiprows=i, encoding=enc)
                    if not df.columns.str.contains('Unnamed').all(): break
            return df
        except: continue
    return None

# --- [3. 상세 팝업] ---
@st.dialog("현장별 상세 내역")
def show_detail_dialog(group_ids, df_bl):
    st.write(f"🔎 분석 품번: {', '.join(group_ids)}")
    # 스니펫 기반: 상품코드는 보통 5번 혹은 '상품코드' 열
    code_col = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
    detail = df_bl[df_bl[code_col].astype(str).isin(group_ids)].copy()
    
    if detail.empty:
        st.info("해당 품번의 수주 데이터가 없습니다.")
        return

    today = datetime.now()
    date_col = '납품예정일' if '납품예정일' in df_bl.columns else df_bl.columns[24]
    detail['상태'] = pd.to_datetime(detail[date_col], errors='coerce').apply(
        lambda x: "⚠️ 납기경과" if pd.notnull(x) and x < today else "정상"
    )
    st.dataframe(detail.sort_values(date_col), use_container_width=True, hide_index=True)

# --- [4. 메인 UI] ---
st.title("📦 P·Forecast Stock Manager")
st.caption("건설 특판 모양지 통합 수급 예측 시스템")

uploaded_files = st.sidebar.file_uploader("5종의 CSV 파일을 선택하세요", accept_multiple_files=True)

data = {}
RECOGNITION_MAP = {
    "backlog": {"name": "수주예정(Demand)", "keys": ["수주잔량", "총예상수량", "수주번호"]},
    "po": {"name": "구매발주(PO)", "keys": ["PO잔량", "미선적", "PO번호"]},
    "stock": {"name": "현재고(Stock)", "keys": ["재고수량", "현재고액", "본사창고"]},
    "item": {"name": "품목정보(Master)", "keys": ["최종생산지", "이전상품코드", "상품코드"]},
    "retail": {"name": "시판스펙(Retail)", "keys": ["출시예정", "4개월판매량", "시판"]}
}

if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            cols_text = "|".join(df.columns)
            for k, v in RECOGNITION_MAP.items():
                if any(key in cols_text for key in v["keys"]):
                    data[k] = df
                    break

# 사이드바 상태 표시
st.sidebar.markdown("---")
for k, v in RECOGNITION_MAP.items():
    if k in data: st.sidebar.success(f"✅ {v['name']}")
    else: st.sidebar.error(f"❌ {v['name']} (미인식)")

# 분석 실행
if len(data) >= 5:
    with st.spinner('데이터를 분석하고 매트릭스를 생성 중입니다. 잠시만 기다려주세요...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']

        # 열 인덱스 자동 매핑 (스니펫 기준)
        bl_code_col = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
        bl_qty_col = '수주잔량'
        bl_date_col = '납품예정일' if '납품예정일' in df_bl.columns else df_bl.columns[24]
        
        df_bl[bl_qty_col] = clean_numeric(df_bl[bl_qty_col])
        df_bl['dt_temp'] = parse_date(df_bl[bl_date_col])

        po_code_col = '품번' if '품번' in df_po.columns else df_po.columns[12]
        po_qty_col = 'PO잔량(미선적)'
        po_date_col = '입고요청일'
        df_po[po_qty_col] = clean_numeric(df_po[po_qty_col])
        df_po['dt_temp'] = parse_date(df_po[po_date_col])

        st_code_col = '품번' if '품번' in df_st.columns else df_st.columns[7]
        st_qty_col = '재고수량' if '재고수량' in df_st.columns else df_st.columns[17]
        df_st[st_qty_col] = clean_numeric(df_st[st_qty_col])

        # 타임라인 설정
        today_base = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_cols = [(today_base + pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(12)]

        target_ids = df_bl[df_bl[bl_qty_col] > 0][bl_code_col].unique()
        matrix_rows = []
        processed_groups = set()

        for pid in target_ids:
            # 품번 연계 (간소화 버전)
            pid_str = str(pid)
            group = [pid_str]
            if '변경상품코드' in df_item.columns:
                match = df_item[df_item['상품코드'].astype(str) == pid_str]
                if not match.empty and pd.notnull(match['변경상품코드'].iloc[0]):
                    group.append(str(match['변경상품코드'].iloc[0]))
            
            group = sorted(list(set(group)))
            group_key = tuple(group)
            if group_key in processed_groups: continue
            processed_groups.add(group_key)

            # 생산지 및 LT
            item_match = df_item[df_item['상품코드'].astype(str).isin(group)]
            site = str(item_match['최종생산지명'].iloc[0]) if not item_match.empty else "ETC"
            lt = LT_MASTER.get(site[:2].upper(), 0)

            # 수지 계산
            total_stk = df_st[df_st[st_code_col].astype(str).isin(group)][st_qty_col].sum()
            overdue_dem = df_bl[(df_bl[bl_code_col].astype(str).isin(group)) & (df_bl['dt_temp'] < today_base)][bl_qty_col].sum()
            
            running_inv = total_stk - overdue_dem
            row_dem = {"납기경과": overdue_dem}
            row_stk = {"납기경과": running_inv}

            for m_str in month_cols:
                m_dt = datetime.strptime(m_str, '%Y-%m')
                m_d = df_bl[(df_bl[bl_code_col].astype(str).isin(group)) & (df_bl['dt_temp'].dt.strftime('%Y-%m') == m_str)][bl_qty_col].sum()
                m_p_df = df_po[(df_po[po_code_col].astype(str).isin(group)) & (df_po['dt_temp'].dt.strftime('%Y-%m') == m_str)]
                m_s = sum([(r[po_qty_col] * 1000) / (70 * 1.26) for _, r in m_p_df.iterrows()])
                
                running_inv = (running_inv + m_s) - m_d
                row_dem[m_str] = round(m_d, 0)
                row_stk[m_str] = round(running_inv, 0)

            common = {"품번": f"{pid}", "생산지": f"{site}({lt}M)", "group": group}
            matrix_rows.append({**common, "구분": "소요량(m)", **row_dem})
            matrix_rows.append({**common, "구분": "예상재고(m)", **row_stk})

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        
        def style_matrix(row):
            styles = [''] * len(row)
            if row['구분'] == "예상재고(m)":
                for i, col in enumerate(row.index):
                    if (col == "납기경과" or '-' in col) and isinstance(row[col], (int, float)) and row[col] < 0:
                        styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        st.subheader("📊 통합 수급 분석 매트릭스")
        
        # [수정] selection_mode를 가장 표준적인 문자열로 변경
        # 만약 여기서 또 오류가 난다면 selection 기능을 끄고 표시만 하도록 안전장치 마련
        try:
            selection = st.dataframe(
                res_df.style.apply(style_matrix, axis=1),
                use_container_width=True, hide_index=True,
                column_order=["품번", "생산지", "구분", "납기경과"] + month_cols,
                on_select="rerun", 
                selection_mode="single_row" # 리스트에서 문자열로 복구
            )

            if selection.selection.rows:
                sel_idx = selection.selection.rows[0]
                if st.button(f"🔍 {res_df.iloc[sel_idx]['품번']} 상세 정보 보기"):
                    show_detail_dialog(res_df.iloc[sel_idx]['group'], df_bl)
        except Exception as e:
            # 선택 기능 오류 시 테이블만 표시
            st.dataframe(res_df.style.apply(style_matrix, axis=1), use_container_width=True, hide_index=True)
            st.error(f"UI 선택 기능 오류로 테이블만 표시합니다. (사유: {e})")
else:
    st.info("5종의 파일을 모두 업로드해주세요.")
