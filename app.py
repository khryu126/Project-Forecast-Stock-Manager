import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, utils
import re
import google.generativeai as genai
from datetime import datetime, timedelta

# --- 1. 페이지 설정 및 디자인 ---
st.set_page_config(page_title="성지라미텍 특판 리스크 관리 시스템", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AI 검색 엔진 설정 (Gemini) ---
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # 인터넷 검색 도구가 활성화된 모델 설정
    model = genai.GenerativeModel('gemini-1.5-pro', tools=[{"google_search": {}}])
else:
    st.sidebar.warning("⚠️ Secrets에 GOOGLE_API_KEY가 설정되지 않아 AI 검색 기능을 사용할 수 없습니다.")

def get_ai_search_result(site_name):
    """AI를 통해 현장명/지번으로 실제 아파트 단지명과 시공사 정보를 검색"""
    prompt = f"건설 현장명 또는 지번 '{site_name}'에 대해 인터넷에서 검색하여 실제 아파트 단지명, 브랜드명, 시공사 정보를 짧고 명확하게 요약해줘."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 검색 중 오류 발생: {str(e)}"

# --- 3. 유틸리티 함수 ---
def safe_read_csv(file, skiprows=0):
    if file is not None:
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, skiprows=skiprows)
                df.columns = df.columns.str.strip()
                return df
            except:
                continue
    return None

def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)

def clean_site_name(name):
    if not name or pd.isna(name): return ""
    name = re.sub(r'\(주\)|주식회사|신축공사|현장|일대|M/H|MH|S/H|SH|샘플', '', str(name))
    name = " ".join(name.split())
    return name

# --- 4. 사이드바: 데이터 업로드 ---
st.sidebar.header("📁 데이터 소스 업로드")
f_expected = st.sidebar.file_uploader("1. 수주예정등록.csv", type="csv")
f_stock = st.sidebar.file_uploader("2. 현재고.csv", type="csv")
f_history = st.sidebar.file_uploader("3. 출고내역.csv", type="csv")
f_item = st.sidebar.file_uploader("4. 품목정보.csv", type="csv")
f_retail = st.sidebar.file_uploader("5. 시판스펙관리.csv", type="csv")
f_po = st.sidebar.file_uploader("6. PO.csv", type="csv")

df_exp = safe_read_csv(f_expected, skiprows=1)
df_stk = safe_read_csv(f_stock)
df_his = safe_read_csv(f_history)
df_itm = safe_read_csv(f_item)
df_rtl = safe_read_csv(f_retail)
df_po = safe_read_csv(f_po)

st.title("🛡️ 성지라미텍 특판 리스크 관리 시스템")

if df_exp is not None and df_stk is not None:
    # 데이터 전처리
    df_exp['수주잔량_n'] = to_numeric(df_exp['수주잔량'])
    df_stk['재고수량_n'] = to_numeric(df_stk['재고수량'])
    
    weight_map = {}
    if df_itm is not None:
        w_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight' if 'B/P weight' in df_itm.columns else None
        if w_col:
            weight_map = df_itm.set_index('상품코드')[w_col].to_dict()

    tab1, tab2 = st.tabs(["📍 현장 누락 방지 점검", "📅 오더 시점 및 재고 예측"])

    # --- TAB 1: 현장 누락 방지 ---
    with tab1:
        st.subheader("🏢 특판 현장(M/H, S/H) 출고 기반 등록 여부 확인")
        if df_his is not None:
            target_keywords = ['M/H', 'MH', 'S/H', 'SH']
            mh_pattern = '|'.join(target_keywords)
            mh_deliveries = df_his[
                df_his['현장명'].str.contains(mh_pattern, na=False, case=False) |
                df_his['비고'].str.contains(mh_pattern, na=False, case=False)
            ].copy()

            if not mh_deliveries.empty:
                unique_sites = mh_deliveries['현장명'].unique()
                expected_sites = df_exp['현장명'].unique()
                clean_exp_list = [clean_site_name(s) for s in expected_sites]
                exp_map = {clean_site_name(s): s for s in expected_sites}

                results = []
                for site in unique_sites:
                    c_site = clean_site_name(site)
                    match = process.extractOne(c_site, clean_exp_list, processor=utils.default_process)
                    score = match[1] if match else 0
                    match_original = exp_map.get(match[0]) if match else "없음"
                    
                    status = "✅ 등록됨" if score > 85 else "⚠️ 누락 의심" if score > 50 else "🔴 미등록"
                    
                    # 결과 행 구성
                    col_a, col_b, col_c, col_d = st.columns([3, 3, 1, 2])
                    with col_a: st.write(f"**출고명:** {site}")
                    with col_b: st.write(f"**수주매칭:** {match_original} ({score:.1f}%)")
                    with col_c: st.write(status)
                    with col_d:
                        if status != "✅ 등록됨" and "GOOGLE_API_KEY" in st.secrets:
                            if st.button(f"🔍 AI 검색", key=f"btn_{site}"):
                                with st.spinner('AI가 인터넷 정보를 분석 중입니다...'):
                                    search_info = get_ai_search_result(site)
                                    st.info(f"**AI 분석 결과:**\n{search_info}")

                st.divider()
                st.caption("※ 유사도가 낮거나 미등록인 건은 AI 검색 버튼을 눌러 실제 아파트 이름을 확인해 보세요.")
            else:
                st.write("분석된 M/H/SH 출고 내역이 없습니다.")
        else:
            st.warning("출고내역.csv 파일을 업로드해 주세요.")

    # --- TAB 2: 오더 시점 및 재고 예측 ---
    with tab2:
        st.subheader("모양지 발주 검토 (독일 리드타임 4개월)")
        target_item = st.selectbox("점검할 품번(상품코드)을 선택하세요", df_exp['상품코드'].unique())
        
        curr_inv = df_stk[df_stk['품번'] == target_item]['재고수량_n'].sum()
        po_m = 0
        if df_po is not None:
            po_data = df_po[df_po['품번'] == target_item].copy()
            if not po_data.empty:
                bw = weight_map.get(target_item, 70) 
                # 공식: m = (kg * 1000) / (평량 * 1.26)
                po_m = (to_numeric(po_data['PO 수량']).sum() * 1000) / (bw * 1.26)
        
        spec_demand = df_exp[df_exp['상품코드'] == target_item]['수주잔량_n'].sum()
        retail_monthly = 0
        if df_rtl is not None:
            rtl_match = df_rtl[df_rtl['품번'] == target_item]
            if not rtl_match.empty:
                retail_monthly = to_numeric(rtl_match['4개월판매량']).values[0] / 4

        c1, c2, c3 = st.columns(3)
        c1.metric("현재고 (m)", f"{curr_inv:,.0f}")
        c2.metric("PO 예정량 (m)", f"{po_m:,.0f}")
        c3.metric("특판 수주잔량 (m)", f"{spec_demand:,.0f}")

        st.write("### 📉 향후 6개월 재고 시뮬레이션 (시판 수요 포함)")
        months = [(datetime.now() + timedelta(days=30*i)).strftime("%Y-%m") for i in range(1, 7)]
        sim_balance = curr_inv + po_m
        graph_data = []
        for m in months:
            sim_balance -= retail_monthly 
            graph_data.append({"월": m, "예상재고": max(0, sim_balance)})
        
        st.line_chart(pd.DataFrame(graph_data).set_index("월"))
        
        if sim_balance < spec_demand:
            st.error(f"🚨 위험: 4개월 내 재고 쇼트 발생 가능성 높음! (부족분: {spec_demand - sim_balance:,.0f} m)")
        else:
            st.success("안정권: 수주 물량 대응이 가능합니다.")

else:
    st.warning("사이드바에서 '수주예정등록.csv'와 '현재고.csv'를 먼저 업로드해 주세요.")
