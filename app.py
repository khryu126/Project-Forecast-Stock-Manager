import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="성지라미텍 특판 오더 관리 시스템", layout="wide")

st.markdown("""
    <style>
    .shortage { background-color: #ffcccc; color: #cc0000; font-weight: bold; }
    .safe { background-color: #ccffcc; color: #006600; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 유틸리티 함수 (자동 파일 식별 및 로드) ---

def identify_and_load(uploaded_files):
    """파일명이 달라도 컬럼명을 분석해 어떤 데이터인지 식별"""
    data = {}
    for file in uploaded_files:
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            try:
                file.seek(0)
                # 수주 파일처럼 첫 줄이 비어있는 경우를 대비해 유연하게 로드
                df = pd.read_csv(file, encoding=enc)
                if df.columns[0].startswith('Unnamed') or len(df.columns) < 2:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc, skiprows=1)
                
                df.columns = df.columns.str.strip()
                
                # 컬럼 특성에 따른 식별
                cols = "".join(df.columns)
                if '수주잔량' in cols and '납품예정일' in cols:
                    data['exp'] = df
                elif '재고수량' in cols and '현재고액' in cols:
                    data['stk'] = df
                elif 'PO 수량' in cols or 'PO번호' in cols:
                    data['po'] = df
                elif 'B/P무게' in cols or 'B/P weight' in cols:
                    data['itm'] = df
                elif '4개월판매량' in cols:
                    data['rtl'] = df
                break
            except:
                continue
    return data

def to_num(val):
    if pd.isna(val): return 0
    return pd.to_numeric(str(val).replace(',', '').strip(), errors='coerce') or 0

# --- 3. 사이드바: 파일 업로드 ---
st.sidebar.header("📁 소스 파일 통합 업로드")
files = st.sidebar.file_uploader("모든 관련 CSV 파일을 한꺼번에 선택해서 올려주세요.", type="csv", accept_multiple_files=True)

st.title("📊 특판 모양지 오더 관리 대시보드")

if files:
    loaded_data = identify_and_load(files)
    
    # 필수 데이터 확인 (수주, 현재고)
    if 'exp' in loaded_data and 'stk' in loaded_data:
        df_exp = loaded_data['exp']
        df_stk = loaded_data['stk']
        df_itm = loaded_data.get('itm')
        df_po = loaded_data.get('po')
        df_rtl = loaded_data.get('rtl')

        # --- 데이터 전처리 ---
        # 1. 평량 맵 생성
        weight_map = {}
        if df_itm is not None:
            w_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight'
            weight_map = df_itm.set_index('상품코드')[w_col].to_dict()

        # 2. 기간 단위 선택
        st.sidebar.divider()
        unit = st.sidebar.radio("🗓️ 분석 단위 선택", ["월별", "분기별"])
        
        # 3. 분석 대상 품번 추출 (수주에 있는 것 기준)
        items = df_exp['상품코드'].unique()
        
        # --- 핵심 계산 로직 ---
        report_rows = []
        
        # 미래 기간 생성 (현재부터 6개월/4분기)
        now = datetime.now()
        if unit == "월별":
            periods = [(now + timedelta(days=30*i)).strftime("%Y-%m") for i in range(7)]
        else:
            periods = [f"{now.year} Q{(now.month-1)//3 + 1 + i}" for i in range(4)] # 간단 분기 계산

        for item in items:
            # 초기 재고
            current_inv = to_num(df_stk[df_stk['품번'] == item]['재고수량'].sum())
            
            # PO 잔량 환산
            po_m = 0
            if df_po is not None:
                po_data = df_po[df_po['품번'] == item]
                bw = weight_map.get(item, 70)
                po_m = (to_numeric(po_data['PO 수량']).sum() * 1000) / (bw * 1.26)
            
            # 시판 월 소요량
            retail_m = 0
            if df_rtl is not None:
                rtl_row = df_rtl[df_rtl['품번'] == item]
                if not rtl_row.empty:
                    retail_m = to_num(rtl_row['4개월판매량'].values[0]) / 4

            # 수주 잔량 (시계열 배분은 단순화를 위해 첫 달에 몰거나 예정일 파싱 가능)
            total_spec = to_num(df_exp[df_exp['상품코드'] == item]['수주잔량'].sum())

            # 행 데이터 생성
            row = {"품번": item, "현재고(m)": current_inv + po_m}
            balance = current_inv + po_m
            
            for p in periods:
                # 여기에 기간별 수주 예정일을 매칭하여 balance 차감 가능
                balance -= retail_m # 일단 시판 수요 매달 차감
                row[f"{p} 재고"] = balance
            
            report_rows.append(row)

        # --- 4. 대시보드 출력 ---
        final_df = pd.DataFrame(report_rows)

        # 스타일 함수: 재고 부족 시 빨간색
        def style_inventory(val):
            if isinstance(val, (int, float)) and val < 0:
                return 'background-color: #ffcccc; color: #cc0000'
            elif isinstance(val, (int, float)) and val > 0:
                return 'background-color: #ccffcc; color: #006600'
            return ''

        st.subheader(f"📅 {unit} 재고 수지 현황")
        st.dataframe(final_df.style.applymap(style_inventory, subset=[c for c in final_df.columns if '재고' in c]), use_container_width=True)

        # 엑셀 다운로드 버튼
        st.download_button("📊 분석 결과 다운로드 (CSV)", final_df.to_csv(index=False).encode('utf-8-sig'), "특판_재고분석.csv", "text/csv")

    else:
        st.warning("분석을 위해 최소한 '수주예정등록'과 '현재고' 파일이 포함되어야 합니다.")
else:
    st.info("사이드바에서 관련 CSV 파일들을 한꺼번에 업로드해 주세요 (파일명 상관없음).")
