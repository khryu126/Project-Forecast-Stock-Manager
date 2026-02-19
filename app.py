import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="성지라미텍 오더 관리 시스템", layout="wide")

# --- 2. 유틸리티 함수 (에러 방지용 안전 설계) ---

def to_num(series):
    """문자열 숫자를 안전하게 실수형으로 변환"""
    if series is None: return pd.Series(0.0)
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.replace(' ', '').str.strip(), errors='coerce').fillna(0.0)

def identify_data(uploaded_files):
    """파일 내용(컬럼명)을 분석해 자동으로 분류"""
    data_map = {}
    for file in uploaded_files:
        identified = False
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            if identified: break
            for sr in [0, 1, 2]: # 최대 2줄 건너뜀
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc, skiprows=sr)
                    df.columns = [str(c).strip() for c in df.columns]
                    cols = " ".join(df.columns)
                    
                    if '수주잔량' in cols and '납품예정일' in cols:
                        data_map['exp'] = df; identified = True; break
                    elif '재고수량' in cols and '현재고액' in cols:
                        data_map['stk'] = df; identified = True; break
                    elif 'PO 수량' in cols or 'PO잔량' in cols:
                        data_map['po'] = df; identified = True; break
                    elif 'B/P무게' in cols or 'B/P weight' in cols:
                        data_map['itm'] = df; identified = True; break
                    elif '4개월판매량' in cols:
                        data_map['rtl'] = df; identified = True; break
                except: continue
    return data_map

# --- 3. 메인 로직 ---

st.title("🛡️ 특판 모양지 통합 오더 관리 시스템 (안정화 버전)")

# 파일 업로드
uploaded_files = st.sidebar.file_uploader("CSV 파일들을 한꺼번에 선택해서 올려주세요", type="csv", accept_multiple_files=True)

if uploaded_files:
    data = identify_data(uploaded_files)
    
    # 필수 파일(수주, 재고) 체크
    if 'exp' in data and 'stk' in data:
        df_exp, df_stk = data['exp'], data['stk']
        df_po, df_itm, df_rtl = data.get('po'), data.get('itm'), data.get('rtl')
        
        # 컬럼 표준화
        exp_col = '상품코드' if '상품코드' in df_exp.columns else '품번'
        stk_col = '품번' if '품번' in df_stk.columns else '상품코드'
        
        # 수주 데이터 전처리 (IndexError 방지)
        df_exp['수주잔량_n'] = to_num(df_exp['수주잔량'])
