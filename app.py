import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import ssl
import torch
import torchvision.transforms as T
import cv2
import requests
import base64
from PIL import Image, ImageEnhance, ImageDraw
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as k_image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# [0] 환경 설정 및 보안 대응 (DINOv2 및 이미지 로딩 안정화)
ssl._create_default_https_context = ssl._create_unverified_context

# --- [1] 유틸리티 및 데이터 로드 로직 ---
def get_direct_url(url):
    """구글 드라이브 URL 변환"""
    if not url or str(url) == 'nan' or 'drive.google.com' not in url: return url
    if 'file/d/' in url: file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url: file_id = url.split('id=')[1].split('&')[0]
    else: return url
    return f'https://drive.google.com/uc?export=download&id={file_id}'

def get_image_as_base64(url):
    """구글 보안 우회 및 엑박 방지 (Base64 Proxy)"""
    try:
        r = requests.get(get_direct_url(url), timeout=10)
        img_str = base64.b64encode(r.content).decode()
        return f"data:image/png;base64,{img_str}"
    except: return None

def load_csv_smart(target_name):
    """4가지 인코딩 자동 시도로 한글 깨짐 방지"""
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    return pd.DataFrame()

def get_digits(text):
    return "".join(re.findall(r'\d+', str(text))) if text else ""

@st.cache_resource
def init_resources():
    # 1. AI 모델 로드 (Hybrid: ResNet50 + DINOv2)
    model_res = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_dino.eval()
    
    # 2. 데이터베이스 로드
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
        
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    # 3. [v2.6 이식] 정밀 재고 매칭 로직
    agg_stock, stock_date = {}, "확인불가"
    if not df_stock.empty:
        # 규칙: 콤마 제거 및 숫자화
        df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        # 규칙: astype(str).strip().upper() 완벽 키 생성
        df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
        # 규칙: 중복 품번 합산
        agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
        if '정산일자' in df_stock.columns:
            stock_date = str(int(df_stock['정산일자'].max()))
            
    return model_res, model_dino, feature_db, df_path, df_info, agg_stock, stock_date

res_model, dino_model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

# --- [2] 이미지 처리 엔진 (Advanced Corrections) ---
def apply_advanced_correction(img, angle, bri, con, shp, sat, temp, exp, hue):
    """사용자 요청 5대 보정 옵션 및 회전 기능"""
    # 회전
    if angle != 0: img = img.rotate(angle, expand=True)
    # 기본 필터
    img = ImageEnhance.Brightness(img).enhance(bri)
    img = ImageEnhance.Contrast(img).enhance(con)
    img = ImageEnhance.Sharpness(img).enhance(shp)
    img = ImageEnhance.Color(img).enhance(sat)
    # 노출 및 색온도 (Numpy)
    img_np = np.array(img).astype(np.float32)
    img_np *= exp # 노출
    if temp > 1.0: img_np[:,:,0] *= temp; img_np[:,:,2] /= temp # Warm
    elif temp < 1.0: img_np[:,:,2] *= (2.0-temp); img_np[:,:,0] /= (2.0-temp) # Cool
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    # 색조 (HSV)
    if hue != 0:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + hue) % 180
        img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_np)

def four_point_transform(image, pts):
    """LANCZOS4 고화질 워핑"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    w = max(int(np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))), int(np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))))
    h = max(int(np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))), int(np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LANCZOS4)

# --- [3] DecoMatch UI & Branding ---
st.set_page_config(layout="wide", page_title="DecoMatch - Schattdecor")

# Schattdecor 테마 적용
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #B67741; color: white; border-radius: 4px; border: none; }
    .stExpander { border: 1px solid #B67741; border-radius: 5px; background-color: white; }
    h1 { color: #B67741; font-family: 'Arial Black', sans-serif; }
    </style>
    """, unsafe_allow_value=True)

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://brandfetch.com/schattdecor.com?view=library", width=120)
with col_title:
    st.title("DecoMatch")
    st.caption("Advanced Hybrid Surface Pattern Recognition")

st.sidebar.markdown(f"📦 **재고 정산일:** \n{stock_date}")

if 'points' not in st.session_state: st.session_state['points'] = []
if 'search_done' not in st.session_state: st.session_state['search_done'] = False
if 'refresh_count' not in st.session_state: st.session_state['refresh_count'] = 0

uploaded = st.file_uploader("📷 자재 사진 업로드 (Upload Material Image)", type=['jpg','png','jpeg'])

if uploaded:
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
        st.session_state.update({'points': [], 'search_done': False, 'current_img_name': uploaded.name, 'proc_img': Image.open(uploaded).convert('RGB')})
        st.rerun()

    working_img = st.session_state['proc_img']
    w, h = working_img.size

    # 고급 옵션 Expander
    with st.expander("🛠️ 고급 이미지 보정 및 사진 회전 (Advanced Settings)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            angle = st.slider("사진 회전 (Rotation)", 0, 360, 0)
            bri = st.slider("밝기 (Brightness)", 0.5, 2.0, 1.0)
            con = st.slider("대비 (Contrast)", 0.5, 2.0, 1.0)
        with c2:
            shp = st.slider("선명도 (Sharpness)", 0.0, 3.0, 1.5)
            sat = st.slider("채도 (Saturation)", 0.0, 2.0, 1.0)
            exp = st.slider("노출 (Exposure)", 0.5, 2.0, 1.0)
        with c3:
            temp = st.slider("색온도 (Color Temp)", 0.5, 1.5, 1.0)
            hue = st.slider("색조 (Hue Shift)", 0, 180, 0)
            if st.button("🔄 점 전체 초기화"): st.session_state['points'] = []; st.rerun()

    # 영역 지정 및 보기 크기 제어
    scale = st.radio("🔍 보기 크기 (View Scale):", [0.1, 0.3, 0.5, 0.7, 1.0], index=2, horizontal=True)
    
    col_ui, col_pad = st.columns([1, 2])
    with col_ui:
        source_type = st.radio("원본 구분", ['📸 현장 사진', '💻 디지털 파일'], horizontal=True)
        mat_type = st.selectbox("🧱 자재 종류", ['일반', '마루/우드 (Wood)', '하이그로시/유광 (Glossy)', '벽지/패브릭 (Texture)', '석재/콘크리트 (Stone)'])
        s_mode = st.radio("검색 모드", ["종합(컬러+패턴)", "패턴 중심(흑백)"], horizontal=True)
        if st.button("🔄 이미지 안나옴 (새로고침)"): st.session_state['refresh_count'] += 1; st.rerun()

    with col_pad:
        d_img = working_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(d_img)
        # 점 시각화 및 숫자 부여
        for i, p in enumerate(st.session_state['points']):
            px, py = p[0]*scale, p[1]*scale
            draw.ellipse((px-8, py-8, px+8, py+8), fill='#B67741', outline='white', width=2)
            draw.text((px+10, py-10), str(i+1), fill='red')
        # 4점 가이드라인
        if len(st.session_state['points']) == 4:
            draw.polygon([tuple((p[0]*scale, p[1]*scale)) for p in st.session_state['points']], outline='#00FF00', width=3)

        coords = streamlit_image_coordinates(d_img, key=f"deco_{st.session_state['refresh_count']}")
        if coords and len(st.session_state['points']) < 4:
            new_p = (coords['x']/scale, coords['y']/scale)
            if not st.session_state['points'] or st.session_state['points'][-1] != new_p:
                st.session_state['points'].append(new_p); st.rerun()

    if len(st.session_state['points']) == 4:
        warped = four_point_transform(np.array(working_img), np.array(st.session_state['points'], dtype="float32"))
        final_img = Image.fromarray(warped)
        final_img = apply_advanced_correction(final_img, angle, bri, con, shp, sat, temp, exp, hue)
        if s_mode == "패턴 중심(흑백)": final_img = final_img.convert("L").convert("RGB")
        
        st.image(final_img, width=300, caption="DecoMatch Analysis Target")
        
        if st.button("🔍 Run DecoMatch Search", type="primary", use_container_width=True):
            with st.spinner('ResNet(결 60%) + DINO(구조 40%) 하이브리드 분석 중...'):
                # 특징 추출
                x_res = k_image.img_to_array(final_img.resize((224, 224)))
                q_res = res_model.predict(preprocess_input(np.expand_dims(x_res, axis=0)), verbose=0).flatten()
                d_in = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval() # 일시 로드 방지 위해 캐싱 활용 권장
                d_in = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(final_img).unsqueeze(0)
                with torch.no_grad(): q_dino = dino_model(d_in).cpu().numpy().flatten()

                # [v2.6 이식] 정밀 매칭 및 순위 산출
                results = []
                for fn, db_vec in feature_db.items():
                    s_res = cosine_similarity([q_res], [db_vec[:2048]])[0][0]
                    s_dino = cosine_similarity([q_dino], [db_vec[2048:]])[0][0]
                    score = (s_res * 0.6) + (s_dino * 0.4)
                    
                    d_key = get_digits(fn)
                    # 품목 마스터 맵 연동
                    match_info = df_info[df_info['상품코드'].apply(get_digits) == d_key]
                    f_code = match_info.iloc[0]['상품코드'] if not match_info.empty else fn.split('.')[0]
                    
                    # [v2.6 핵심] strip().upper() 재고 조회
                    f_key = str(f_code).strip().upper()
                    qty = agg_stock.get(f_key, 0)
                    
                    url_row = df_path[df_path['추출된_품번'].apply(get_digits) == d_key]
                    url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                    
                    if url:
                        results.append({'formal': f_code, 'name': match_info.iloc[0]['상품명'] if not match_info.empty else "정보없음", 'score': score, 'url': url, 'stock': qty})

                results.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['search_results'] = results[:15]
                st.session_state['search_done'] = True; st.rerun()

# --- [4] 결과 출력 (Ranked & Expandable) ---
if st.session_state.get('search_done'):
    st.markdown("---")
    st.subheader("🏆 DecoMatch Ranking (Top 15)")
    res = st.session_state['search_results']
    cols = st.columns(5)
    for i, item in enumerate(res):
        with cols[i % 5]:
            # 카드 헤더 (순위 및 품번)
            st.markdown(f"#### Rank {i+1}")
            st.markdown(f"**{item['formal']}**")
            st.caption(f"Similarity: {item['score']:.1%}")
            
            # 결과 이미지 접기/펼치기
            with st.expander("🖼️ 이미지/상세 정보", expanded=False):
                # 구글 우회 출력
                b64_img = get_image_as_base64(item['url'])
                if b64_img: st.image(b64_img, use_container_width=True)
                else: st.warning("이미지 로드 실패")
                st.write(f"**품명:** {item['name']}")
                # 재고 표시
                if item['stock'] >= 100: st.success(f"현재고: {item['stock']:,}m")
                else: st.info(f"현재고: {item['stock']:,}m")
