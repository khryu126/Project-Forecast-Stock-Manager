import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# 앱 제목 및 설정
st.set_page_config(page_title="자재 패턴 검색기", page_icon="🔍")
st.title("🔍 실시간 자재 패턴 검색")
st.write("현장에서 찍은 사진을 올리면 가장 유사한 자재를 찾아드립니다.")

# 1. 데이터 로드 함수 (에러 메시지 강화 버전)
@st.cache_resource
def load_resources():
    # 파일명 정의 (대리님이 올려주신 이름과 정확히 일치해야 함)
    pkl_file = '자재_지문_장부_light.pkl'
    spec_file = '스펙인코드_25.12.08.csv'
    link_file = '제목 없는 스프레드시트 - 시트1.csv'

    # 파일 존재 여부 확인
    missing_files = []
    for f in [pkl_file, spec_file, link_file]:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        # 어떤 파일이 없는지 화면에 정확히 표시
        st.error(f"⚠️ 아래 파일들을 찾을 수 없습니다: {', '.join(missing_files)}")
        st.info(f"현재 폴더의 파일 목록: {os.listdir()}")
        return None, None, None, None

    # 모델 로드
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # 피클 로드
    with open(pkl_file, 'rb') as f:
        feature_dict = pickle.load(f)
    
    # 엑셀 로드 (인코딩 자동 시도)
    def read_csv_safe(path):
        for enc in ['cp949', 'utf-8-sig', 'euc-kr']:
            try:
                return pd.read_csv(path, encoding=enc)
            except:
                continue
        return None

    spec_df = read_csv_safe(spec_file)
    link_df = read_csv_safe(link_file)
    
    return model, feature_dict, spec_df, link_df

# 리소스 불러오기 실행
model, feature_dict, spec_df, link_df = load_resources()

# 데이터가 모두 로드되었을 때만 실행
if model is not None:
    # 2. 사진 업로드 섹션
    uploaded_file = st.file_uploader("가구 사진을 촬영하거나 업로드하세요", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='업로드된 사진', use_column_width=True)
        
        with st.spinner('유사한 자재를 분석 중입니다...'):
            img_resized = img.resize((224, 224))
            x = image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            target_feat = model.predict(x).flatten()
            
            # 대조 작업
            scores = [(f, cosine_similarity([target_feat], [feat])[0][0]) for f, feat in feature_dict.items()]
            top_results = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            
            st.subheader("✨ 분석 결과 Top 3")
            for i, (fname, score) in enumerate(top_results):
                m = link_df[link_df['파일명'] == fname]
                if not m.empty:
                    pumbun = m.iloc[0]['추출된_품번']
                    url = m.iloc[0]['카카오톡_전송용_URL']
                    s = spec_df[spec_df['품번'] == str(pumbun).strip()]
                    name = s.iloc[0]['품명'] if not s.empty else "정보없음"
                    
                    with st.expander(f"{i+1}순위: {name} (일치율 {score*100:.1f}%)"):
                        st.write(f"**품번:** {pumbun}")
                        st.link_button("구글 드라이브 사진 확인", url)
else:
    st.warning("⚠️ 파일을 찾을 수 없어 분석을 시작할 수 없습니다. GitHub에 올린 파일명과 위 코드에 적힌 이름이 완전히 똑같은지 확인해주세요.")
