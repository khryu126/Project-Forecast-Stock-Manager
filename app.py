import streamlit as st
import pandas as pd
from rapidfuzz import process, utils
from datetime import datetime, timedelta

st.set_page_config(page_title="성지라미텍 특판 발주 검토 시스템", layout="wide")

# --- [이미지: 데이터 흐름도] ---
# 

st.title("📦 특판 모양지 발주 및 현장 관리 시스템")
st.sidebar.header("데이터 업로드")

# 1. 파일 업로드 섹션
uploaded_files = {
    "수주": st.sidebar.file_uploader("수주예정등록.csv", type="csv"),
    "재고": st.sidebar.file_uploader("현재고.csv", type="csv"),
    "출고": st.sidebar.file_uploader("출고내역.csv", type="csv"),
    "품목": st.sidebar.file_uploader("품목정보.csv", type="csv"),
    "시판": st.sidebar.file_uploader("시판스펙관리.csv", type="csv"),
    "PO": st.sidebar.file_uploader("PO.csv", type="csv")
}

def load_data(file, skiprows=0):
    if file is not None:
        try:
            return pd.read_csv(file, encoding='cp949', skiprows=skiprows)
        except:
            return pd.read_csv(file, encoding='utf-8', skiprows=skiprows)
    return None

# 데이터 로드 (도안 구조 반영)
df_expected = load_data(uploaded_files["수주"], skiprows=1)
df_stock = load_data(uploaded_files["재고"])
df_history = load_data(uploaded_files["출고"])
df_item_info = load_data(uploaded_files["품목"])
df_retail = load_data(uploaded_files["시판"])
df_po = load_data(uploaded_files["PO"])

if df_expected is not None and df_history is not None:
    tab1, tab2 = st.tabs(["📍 현장 누락 방지 점검", "📅 오더 시점 예측"])

    # --- TAB 1: 현장 누락 방지 ---
    with tab1:
        st.subheader("M/H 및 S/H 출고 기반 누락 현장 탐지")
        
        # M/H, S/H 관련 키워드 필터링
        keywords = ['M/H', 'MH', '모델하우스', 'S/H', 'SH', '샘플']
        mh_history = df_history[df_history['현장명'].str.contains('|'.join(keywords), na=False, case=False)]
        
        unique_mh_sites = mh_history['현장명'].unique()
        expected_sites = df_expected['현장명'].unique()
        
        results = []
        for site in unique_mh_sites:
            # 유사도 매칭 (RapidFuzz 활용)
            match = process.extractOne(site, expected_sites, processor=utils.default_process)
            score = match[1] if match else 0
            match_site = match[0] if match else "없음"
            
            status = "✅ 등록됨" if score > 85 else "⚠️ 누락 의심"
            results.append({"출고 현장명": site, "매칭 수주명": match_site, "유사도": score, "상태": status})
        
        st.table(pd.DataFrame(results))

    # --- TAB 2: 오더 시점 예측 (재고 수지 분석) ---
    with tab2:
        st.subheader("모양지 발주 검토 및 시뮬레이션")
        
        # 1. 평량(Basis Weight) 매핑 테이블 생성 (품목정보 기준)
        # 평량이 품목정보에 없을 경우 기본값 70g 가정
        weight_map = {row['상품코드']: 70 for _, row in df_item_info.iterrows()} 
        
        # 2. PO 데이터 kg -> m 환산
        if df_po is not None:
            def convert_to_meters(row):
                item_code = row['품번']
                kg = row['PO 수량']
                weight = weight_map.get(item_code, 70)
                # 환산 공식: m = (kg * 1000) / (평량 * 1.26)
                return (kg * 1000) / (weight * 1.26)
            
            df_po['PO_m'] = df_po.apply(convert_to_meters, axis=1)

        # 3. 통합 분석 (특정 품번 선택 시 시나리오 보여주기)
        target_item = st.selectbox("분석할 품번을 선택하세요", df_expected['상품코드'].unique())
        
        curr_stock = df_stock[df_stock['품번'] == target_item]['재고수량'].sum()
        po_stock = df_po[df_po['품번'] == target_item]['PO_m'].sum() if df_po is not None else 0
        
        # 특판 수요(수주잔량)
        special_demand = df_expected[df_expected['상품코드'] == target_item]['수주잔량'].replace(',', '', regex=True).astype(float).sum()
        
        # 시판 수요(시판스펙관리)
        retail_row = df_retail[df_retail['품번'] == target_item]
        retail_monthly = (retail_row['4개월판매량'].values[0] / 4) if not retail_row.empty else 0

        st.metric("현재 총 가용량 (현재고 + PO)", f"{curr_stock + po_stock:,.0f} m")
        
        # 간단한 월별 시뮬레이션 (4개월 리드타임 고려)
        st.write("### 📅 향후 6개월 재고 흐름 예측 (독일 리드타임 4개월)")
        
        months = [datetime.now() + timedelta(days=30*i) for i in range(7)]
        sim_data = []
        balance = curr_stock + po_stock
        
        for i, m in enumerate(months):
            if i == 0: continue
            # 매달 시판 수요 차감 + 해당 월 납기인 특판 물량 차감 (샘플 로직)
            balance -= retail_monthly
            sim_data.append({"월": m.strftime("%Y-%m"), "예상재고": balance})
        
        st.line_chart(pd.DataFrame(sim_data).set_index("월"))
        
        if balance < special_demand:
            st.error(f"🚨 경고: 4개월 내 재고 쇼트 발생 위험! (부족분: {special_demand - balance:,.0f} m)")
            st.warning("독일 수입 리드타임을 고려하여 이번 달 내로 오더가 필요합니다.")

else:
    st.info("왼쪽 사이드바에서 모든 소스 파일(.csv)을 업로드해 주세요.")