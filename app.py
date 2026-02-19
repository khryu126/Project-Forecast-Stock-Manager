import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, utils
import google.generativeai as genai
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(page_title="성지라미텍 특판 리스크 관리 시스템", layout="wide")

# --- CSS: 스타일링 ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 유틸리티 함수: 데이터 로드 ---
def safe_read_csv(file, skiprows=0):
    if file is not None:
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, skiprows=skiprows)
                # 컬럼명 앞뒤 공백 제거
                df.columns = df.columns.str.strip()
                return df
            except:
                continue
    return None

def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)

# --- 사이드바: 파일 업로드 ---
st.sidebar.header("📁 데이터 소스 업로드")
f_expected = st.sidebar.file_uploader("1. 수주예정등록.csv (첫 줄 공백 포함)", type="csv")
f_stock = st.sidebar.file_uploader("2. 현재고.csv", type="csv")
f_history = st.sidebar.file_uploader("3. 출고내역.csv", type="csv")
f_item = st.sidebar.file_uploader("4. 품목정보.csv", type="csv")
f_retail = st.sidebar.file_uploader("5. 시판스펙관리.csv", type="csv")
f_po = st.sidebar.file_uploader("6. PO.csv", type="csv")

# 데이터 프레임 로드
df_exp = safe_read_csv(f_expected, skiprows=1)
df_stk = safe_read_csv(f_stock)
df_his = safe_read_csv(f_history)
df_itm = safe_read_csv(f_item)
df_rtl = safe_read_csv(f_retail)
df_po = safe_read_csv(f_po)

st.title("🛡️ 성지라미텍 특판 리스크 관리")
st.info("왼쪽 사이드바에 파일을 업로드하면 자동으로 분석이 시작됩니다.")

if df_exp is not None and df_stk is not None:
    # --- 데이터 전처리 ---
    # 수주잔량 및 예상수량 수치화
    df_exp['수주잔량_n'] = to_numeric(df_exp['수주잔량'])
    df_stk['재고수량_n'] = to_numeric(df_stk['재고수량'])
    
    # 평량(Basis Weight) 매핑
    # 품목정보의 'B/P무게' 또는 'B/P weight' 컬럼 참조
    weight_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight' if 'B/P weight' in df_itm.columns else None
    if weight_col:
        weight_map = df_itm.set_index('상품코드')[weight_col].to_dict()
    else:
        weight_map = {}

    tab1, tab2 = st.tabs(["📍 현장 누락 방지 점검", "📅 오더 시점 및 재고 예측"])

    # --- TAB 1: 현장 누락 방지 ---
    with tab1:
        st.subheader("M/H 및 S/H 출고 기반 등록 여부 확인")
        keywords = ['M/H', 'MH', '모델하우스', 'S/H', 'SH', '샘플']
        
        # 출고내역에서 모델하우스 관련 건 필터링
        mh_mask = df_his['현장명'].str.contains('|'.join(keywords), na=False, case=False) | \
                  df_his['비고'].str.contains('|'.join(keywords), na=False, case=False)
        mh_deliveries = df_his[mh_mask].copy()
        
        if not mh_deliveries.empty:
            unique_sites = mh_deliveries['현장명'].unique()
            expected_sites = df_exp['현장명'].unique()
            
            matching_results = []
            for site in unique_sites:
                # 텍스트 유사도 매칭 (RapidFuzz)
                match = process.extractOne(str(site), expected_sites, processor=utils.default_process)
                score = match[1] if match else 0
                match_name = match[0] if match else "매칭 없음"
                
                status = "✅ 등록됨" if score > 85 else "⚠️ 누락 의심"
                matching_results.append({
                    "출고 현장명": site,
                    "가장 유사한 수주 등록명": match_name,
                    "유사도": f"{score:.1f}%",
                    "상태": status
                })
            
            st.dataframe(pd.DataFrame(matching_results), use_container_width=True)
            st.caption("※ 유사도가 낮거나 '누락 의심'인 건은 현장 주소 정보를 통해 수주 등록 여부를 재확인하세요.")
        else:
            st.write("분석된 M/H 출고 데이터가 없습니다.")

    # --- TAB 2: 오더 시점 예측 ---
    with tab2:
        st.subheader("모양지 발주 검토 (독일 리드타임 4개월 기준)")
        
        # 품번 선택
        target_item = st.selectbox("점검할 품번(상품코드)을 선택하세요", df_exp['상품코드'].unique())
        
        # 1. 가용 재고 계산 (현재고)
        current_inv = df_stk[df_stk['품번'] == target_item]['재고수량_n'].sum()
        
        # 2. PO 입고 예정 물량 (kg -> m 환산)
        po_total_m = 0
        if df_po is not None:
            # PO 파일에서 해당 품번 추출
            po_items = df_po[df_po['품번'] == target_item].copy()
            if not po_items.empty:
                basis_weight = weight_map.get(target_item, 70) # 없으면 기본 70g
                # 환산 공식: m = (kg * 1000) / (평량 * 1.26)
                po_items['qty_m'] = (to_numeric(po_items['PO 수량']) * 1000) / (basis_weight * 1.26)
                po_total_m = po_items['qty_m'].sum()

        # 3. 수요 데이터 집계
        # 특판 수요
        special_demand = df_exp[df_exp['상품코드'] == target_item]['수주잔량_n'].sum()
        
        # 시판 수요 (시판스펙관리)
        retail_monthly = 0
        if df_rtl is not None:
            rtl_data = df_rtl[df_rtl['품번'] == target_item]
            if not rtl_data.empty:
                # 4개월 판매량을 월평균으로 환산
                retail_monthly = to_numeric(rtl_data['4개월판매량']).values[0] / 4

        # 대시보드 지표
        col1, col2, col3 = st.columns(3)
        col1.metric("현재고 (m)", f"{current_inv:,.0f}")
        col2.metric("PO 예정량 (m)", f"{po_total_m:,.0f}")
        col3.metric("총 수주잔량 (m)", f"{special_demand:,.0f}")

        # 시뮬레이션: 월별 재고 흐름
        st.write("### 📉 향후 재고 소진 예측")
        months = [datetime.now() + timedelta(days=30*i) for i in range(1, 7)]
        sim_list = []
        temp_balance = current_inv + po_total_m
        
        # 납기일별 특판 물량 배분 (임시: 납기예정일 컬럼 활용)
        # 실제 데이터의 납기예정일 형식을 파싱해야 함 (예: 20250601)
        for m in months:
            # 해당 월의 시판 수요 차감
            temp_balance -= retail_monthly
            # 해당 월의 특판 수요는 데이터 기반으로 더 정교화 가능
            sim_list.append({"월": m.strftime("%Y-%m"), "예상재고": temp_balance})
            
        st.line_chart(pd.DataFrame(sim_list).set_index("월"))

        # 알람 로직
        if temp_balance < special_demand:
            st.error(f"🚨 재고 부족 위험! (예상 부족분: {special_demand - temp_balance:,.0f} m)")
            st.warning("독일 수입 리드타임(4개월)을 고려하여 오더 시점을 점검하십시오.")
        else:
            st.success("현재고 및 PO 물량으로 수주 물량 대응이 가능할 것으로 보입니다.")

else:
    st.warning("데이터 분석을 위해 상단 '수주예정등록'과 '현재고' 파일 업로드가 필수입니다.")
