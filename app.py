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

# [0] 환경 설정: SSL 우회 및 이미지 처리 보안 대응
ssl._create_default_https_context = ssl._create_unverified_context

# --- [1] 유틸리티 및 리소스 로드 ---
def get_direct_url(url):
    """구글 드라이브 URL을 직접 다운로드 가능하게 변환"""
    if not url or str(url) == 'nan' or 'drive.google.com' not in url: return url
    if 'file/d/' in url: file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url: file_id = url.split('id=')[1].split('&')[0]
    else: return url
    return f'https://drive.google.com/uc?export=download&id={file_id}'

def get_image_as_base64(url):
    """구글 보안 우회: 서버 측에서 이미지를 가져와 base64로 변환"""
    try:
        r = requests.get(get_direct_url(url), timeout=10)
        img_str = base64.b64encode(r.content).decode()
        return f"data:image/png;base64,{img_str}"
    except: return None

def load_csv_smart(target_name):
    """4가지 인코딩 자동 시도로 UnicodeDecodeError 방어"""
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    st.error(f"❌ {target_name} 파일을 찾을 수 없습니다.")
    st.stop()

def get_digits(text):
    return "".join(re.findall(r'\d+', str(text))) if text else ""

@st.cache_resource
def init_resources():
    # 하이브리드 모델 로드 (ResNet50 + DINOv2)
    model_res = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_dino.eval()
    
    # 데이터베이스 및 CSV 로드
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
        
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    # 재고 집계 로직 (유 대리님 기존 로직 유지)
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    stock_date = str(int(df_stock['정산일자'].max())) if '정산일자' in df_stock.columns else "확인불가"
    
    return model_res, model_dino, feature_db, df_path, df_info, agg_stock, stock_date

# 리소스 초기화
res_model, dino_model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

# DINOv2 전용 이미지 변환
dino_transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- [2] 이미지 고도화 처리 엔진 (CLAHE & Warp) ---
def apply_clahe(img):
    """CLAHE(밝기 균일화) 도입: 조명 차이로 인한 인식률 저하 방지"""
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_np = cv2.merge((cl, a, b))
    return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """LANCZOS4 보간법 적용 워핑: 나뭇결 뭉개짐 방지"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    w1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    w2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    h1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    h2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    mW, mH = max(int(w1), int(w2)), max(int(h1), int(h2))
    dst = np.array([[0, 0], [mW - 1, 0], [mW - 1, mH - 1], [0, mH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (mW, mH), flags=cv2.INTER_LANCZOS4)

def apply_smart_filters(img, category, lighting, brightness, sharpness):
    """UI 복구: 자재 카테고리(석재 포함) 및 필터 적용"""
    if lighting == '백열등 (누런 조명)':
        r, g, b = img.split(); b = b.point(lambda i: i * 1.2); img = Image.merge('RGB', (r, g, b))
    img = apply_clahe(img) # CLAHE 자동 적용
    en_con = ImageEnhance.Contrast(img); en_shp = ImageEnhance.Sharpness(img); en_bri = ImageEnhance.Brightness(img)
    if category != '일반':
        img = en_shp.enhance(2.0); img = en_con.enhance(1.1)
    if brightness != 1.0: img = en_bri.enhance(brightness)
    if sharpness != 1.0: img = en_shp.enhance(sharpness)
    return img

# --- [3] 메인 UI (전면 복구) ---
st.set_page_config(layout="wide", page_title="v3.4 통합 자재 검색")
st.title("🏭 하이브리드 자재 패턴 검색 (v3.4)")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

if 'points' not in st.session_state: st.session_state['points'] = []
if 'search_done' not in st.session_state: st.session_state['search_done'] = False
if 'refresh_count' not in st.session_state: st.session_state['refresh_count'] = 0

# UI 복구: 파일 업로드 및 제어 버튼
uploaded = st.file_uploader("📸 분석할 자재 사진을 업로드하세요", type=['jpg','png','jpeg'])

if uploaded:
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
        st.session_state['points'] = []; st.session_state['search_done'] = False
        st.session_state['current_img_name'] = uploaded.name
        st.session_state['proc_img'] = Image.open(uploaded).convert('RGB')
        st.rerun()

    working_img = st.session_state['proc_img']
    w, h = working_img.size
    
    st.markdown("### 1️⃣ 환경 설정")
    source_type = st.radio("📂 원본 종류", ['📸 현장 사진', '💻 디지털 파일'], horizontal=True) # 복구
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1: mat_type = st.selectbox("🧱 자재 종류", ['일반', '마루/우드 (Wood)', '하이그로시/유광 (Glossy)', '벽지/패브릭 (Texture)', '석재/콘크리트 (Stone)']) # 석재 복구
    with c_opt2: s_mode = st.radio("🔎 검색 기준", ["🎨 컬러+패턴 종합(6:4)", "🦓 패턴 중심 (흑백)"], horizontal=True) # 복구

    st.markdown("### 2️⃣ 영역 지정")
    # UI 복구: 보기 크기 10% 단위 제어
    scale = st.radio("🔍 보기 크기 (축소 가능):", [0.1, 0.3, 0.5, 0.7, 1.0], format_func=lambda x: f"{int(x*100)}%", index=3, horizontal=True)
    
    c_ref, c_del, c_auto = st.columns([1, 1, 2])
    with c_ref: 
        if st.button("🔄 이미지 안나옴"): # 복구
            st.session_state['refresh_count'] += 1; st.rerun()
    with c_del:
        if st.button("❌ 선택 초기화"): st.session_state['points'] = []; st.rerun()
    with c_auto:
        if st.button("⏹️ 전체 선택"):
            st.session_state['points'] = [(0, 0), (w, 0), (w, h), (0, h)]; st.rerun()

    # UI 복구: 숫자 및 가이드라인 시각화
    d_img = working_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(d_img)
    for i, p in enumerate(st.session_state['points']):
        px, py = p[0]*scale, p[1]*scale
        draw.ellipse((px-8, py-8, px+8, py+8), fill='red', outline='white', width=2)
        draw.text((px + 10, py - 10), str(i + 1), fill='red') # 숫자 복구

    if len(st.session_state['points']) == 4:
        draw.polygon([tuple((p[0]*scale, p[1]*scale)) for p in order_points(np.array(st.session_state['points']))], outline='#00FF00', width=3) # 가이드라인 복구

    value = streamlit_image_coordinates(d_img, key=f"click_{st.session_state['refresh_count']}")
    if value and len(st.session_state['points']) < 4:
        new_p = (value['x']/scale, value['y']/scale)
        if not st.session_state['points'] or st.session_state['points'][-1] != new_p:
            st.session_state['points'].append(new_p); st.rerun()

    if len(st.session_state['points']) == 4:
        st.markdown("#### 🔍 분석 영역")
        warped = four_point_transform(np.array(working_img), np.array(st.session_state['points'], dtype="float32"))
        final_img = Image.fromarray(warped)
        final_img = apply_smart_filters(final_img, mat_type, '일반', 1.0, 1.5)
        if s_mode == "🦓 패턴 중심 (흑백)": final_img = final_img.convert("L").convert("RGB")
        
        st.image(final_img, width=300, caption="AI 분석 대상")
        
        if st.button("🔍 하이브리드 검색 시작", type="primary", use_container_width=True):
            with st.spinner('ResNet(결 60%) + DINO(구조 40%) 하이브리드 분석 중...'): # 가중치 복구
                # 사용자 이미지 특징 추출
                x_res = k_image.img_to_array(final_img.resize((224, 224)))
                q_res = res_model.predict(preprocess_input(np.expand_dims(x_res, axis=0)), verbose=0).flatten()
                
                d_in = dino_transform(final_img).unsqueeze(0)
                with torch.no_grad():
                    q_dino = dino_model(d_in).cpu().numpy().flatten()

                # 하이브리드 유사도 계산 (0.6:0.4)
                all_results = []
                for fn, db_vec in feature_db.items():
                    db_res = db_vec[:2048]; db_dino = db_vec[2048:]
                    s_res = cosine_similarity([q_res], [db_res])[0][0]
                    s_dino = cosine_similarity([q_dino], [db_dino])[0][0]
                    total_sim = (s_res * 0.6) + (s_dino * 0.4)
                    
                    # 정보 매칭
                    d_key = get_digits(fn)
                    url_row = df_path[df_path['추출된_품번'].apply(get_digits) == d_key]
                    url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                    
                    if url:
                        qty = agg_stock.get(d_key, 0)
                        all_results.append({'formal': fn.split('.')[0], 'score': total_sim, 'url': url, 'stock': qty})

                all_results.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['search_results'] = all_results[:15]
                st.session_state['search_done'] = True; st.rerun()

# --- [4] 결과 출력 (구글 드라이브 액박 해결) ---
if st.session_state.get('search_done'):
    st.markdown("---")
    res_data = st.session_state['search_results']
    cols = st.columns(5)
    for i, item in enumerate(res_data):
        with cols[i % 5]:
            # 액박 해결: 서버에서 base64로 이미지를 인코딩하여 출력
            b64_img = get_image_as_base64(item['url'])
            if b64_img: st.image(b64_img, use_container_width=True)
            else: st.warning("🖼️ 이미지 로드 불가")
            st.markdown(f"**{item['formal']}**")
            st.caption(f"유사도: {item['score']:.1%}")
            st.info(f"재고: {item['stock']:,}m")
