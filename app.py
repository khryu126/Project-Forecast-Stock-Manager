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

# [0] 환경 설정 및 보안 대응
ssl._create_default_https_context = ssl._create_unverified_context

# --- [1] 유틸리티 함수 ---
def get_direct_url(url):
    """구글 드라이브 URL 변환"""
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
    """인코딩 대응 CSV 로드"""
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    return pd.DataFrame()

def get_digits(text):
    return "".join(re.findall(r'\d+', str(text))) if text else ""

# --- [2] 리소스 로딩 (캐싱) ---
@st.cache_resource
def init_resources():
    # 모델 로드 (ResNet50 + DINOv2)
    model_res = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_dino.eval()
    
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
        
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    # [v2.6 이식] 정밀 재고 매칭 로직
    agg_stock, stock_date = {}, "확인불가"
    if not df_stock.empty:
        df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
        agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
        if '정산일자' in df_stock.columns:
            stock_date = str(int(df_stock['정산일자'].max()))
    
    return model_res, model_dino, feature_db, df_path, df_info, agg_stock, stock_date

res_model, dino_model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

# DINOv2 전용 변환
dino_transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_data
def get_master_map():
    mapping = {}
    for _, row in df_info.iterrows():
        f = str(row.get('상품코드', '')).strip()
        n = str(row.get('상품명', '')).strip()
        d = get_digits(f)
        if d: mapping[d] = {'formal': f, 'name': n}
    return mapping

master_map = get_master_map()

# --- [3] 이미지 고도화 처리 엔진 ---
def apply_advanced_correction(img, angle, bri, con, shp, sat, temp, exp, hue):
    """요청하신 5대 보정 및 회전 기능"""
    if angle != 0: img = img.rotate(angle, expand=True)
    img = ImageEnhance.Brightness(img).enhance(bri)
    img = ImageEnhance.Contrast(img).enhance(con)
    img = ImageEnhance.Sharpness(img).enhance(shp)
    img = ImageEnhance.Color(img).enhance(sat)
    
    img_np = np.array(img).astype(np.float32)
    img_np *= exp # 노출 조정
    if temp > 1.0: img_np[:, :, 0] *= temp; img_np[:, :, 2] /= temp # Warm
    elif temp < 1.0: img_np[:, :, 2] *= (2.0-temp); img_np[:, :, 0] /= (2.0-temp) # Cool
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    if hue != 0:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
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
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LANCZOS4)

# --- [4] DecoMatch UI 레이아웃 ---
st.set_page_config(layout="wide", page_title="DecoMatch - Schattdecor")

# [⚠️ 수정완료] unsafe_allow_html=True 로 변경하여 TypeError 해결
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #B67741; color: white; border-radius: 4px; border: none; }
    .stExpander { border: 1px solid #B67741; border-radius: 5px; background-color: white; }
    h1 { color: #B67741; font-family: 'Arial Black', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://brandfetch.com/schattdecor.com?view=library", width=120)
with col_title:
    st.title("DecoMatch")
    st.caption("Advanced Hybrid Surface Pattern Recognition")

st.sidebar.markdown(f"📦 **Inventory Date:** \n{stock_date}")

if 'points' not in st.session_state: st.session_state['points'] = []
if 'search_done' not in st.session_state: st.session_state['search_done'] = False
if 'refresh_count' not in st.session_state: st.session_state['refresh_count'] = 0

uploaded = st.file_uploader("📷 Upload Material Image", type=['jpg','png','jpeg'])

if uploaded:
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
        st.session_state.update({'points': [], 'search_done': False, 'current_img_name': uploaded.name, 'proc_img': Image.open(uploaded).convert('RGB')})
        st.rerun()

    working_img = st.session_state['proc_img']
    w, h = working_img.size

    # 1. 고급 옵션 Expander
    with st.expander("🛠️ Advanced Image Correction & Rotation", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            angle = st.slider("Rotation Angle", 0, 360, 0)
            bri = st.slider("Brightness", 0.5, 2.0, 1.0)
            con = st.slider("Contrast", 0.5, 2.0, 1.0)
        with c2:
            shp = st.slider("Sharpness", 0.0, 3.0, 1.5)
            sat = st.slider("Saturation", 0.0, 2.0, 1.0)
            exp = st.slider("Exposure", 0.5, 2.0, 1.0)
        with c3:
            temp = st.slider("Color Temp", 0.5, 1.5, 1.0)
            hue = st.slider("Hue Shift", 0, 180, 0)
            if st.button("🔄 Reset Points"): st.session_state['points'] = []; st.rerun()

    # 2. 영역 지정 UI
    scale = st.radio("🔍 View Scale:", [0.1, 0.3, 0.5, 0.7, 1.0], index=2, horizontal=True)
    
    col_ui, col_pad = st.columns([1, 2])
    with col_ui:
        source_type = st.radio("Source Type", ['📸 Photo', '💻 Digital'], horizontal=True)
        mat_type = st.selectbox("Material Category", ['Normal', 'Wood', 'Glossy', 'Texture', 'Stone'])
        s_mode = st.radio("Search Mode", ["Hybrid(Color+Pattern)", "Pattern Only(B&W)"], horizontal=True)
        if st.button("🔄 Image Refresh"): st.session_state['refresh_count'] += 1; st.rerun()

    with col_pad:
        d_img = working_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(d_img)
        for i, p in enumerate(st.session_state['points']):
            px, py = p[0]*scale, p[1]*scale
            draw.ellipse((px-8, py-8, px+8, py+8), fill='#B67741', outline='white', width=2)
            draw.text((px+10, py-10), str(i+1), fill='red')
        
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
        if s_mode == "Pattern Only(B&W)": final_img = final_img.convert("L").convert("RGB")
        
        st.image(final_img, width=300, caption="DecoMatch Analysis Target")
        
        if st.button("🔍 Run DecoMatch Search", type="primary", use_container_width=True):
            with st.spinner('Analyzing Texture(60%) & Structure(40%)...'):
                x_res = k_image.img_to_array(final_img.resize((224, 224)))
                q_res = res_model.predict(preprocess_input(np.expand_dims(x_res, axis=0)), verbose=0).flatten()
                
                d_in = dino_transform(final_img).unsqueeze(0)
                with torch.no_grad():
                    q_dino = dino_model(d_in).cpu().numpy().flatten()

                # [v2.6 정밀 매칭] strip().upper() 기반 재고 연동
                results = []
                for fn, db_vec in feature_db.items():
                    s_res = cosine_similarity([q_res], [db_vec[:2048]])[0][0]
                    s_dino = cosine_similarity([q_dino], [db_vec[2048:]])[0][0]
                    score = (s_res * 0.6) + (s_dino * 0.4)
                    
                    d_key = get_digits(fn)
                    info = master_map.get(d_key, {'formal': fn.split('.')[0], 'name': 'Unknown'})
                    
                    # f_key 생성 및 재고 조회
                    f_key = str(info['formal']).strip().upper()
                    qty = agg_stock.get(f_key, 0)
                    
                    url_row = df_path[df_path['추출된_품번'].apply(get_digits) == d_key]
                    url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                    
                    if url:
                        results.append({'formal': info['formal'], 'name': info['name'], 'score': score, 'url': url, 'stock': qty})

                results.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['search_results'] = results[:15]
                st.session_state['search_done'] = True; st.rerun()

# --- [5] 결과 리스트 (Ranked & Expandable) ---
if st.session_state.get('search_done'):
    st.markdown("---")
    st.subheader("🏆 Matching Results (Ranked Top 15)")
    res = st.session_state['search_results']
    cols = st.columns(5)
    for i, item in enumerate(res):
        with cols[i % 5]:
            st.markdown(f"#### Rank {i+1}")
            st.markdown(f"**{item['formal']}**")
            st.caption(f"Similarity: {item['score']:.1%}")
            
            with st.expander("🖼️ View Detail", expanded=False):
                b64 = get_image_as_base64(item['url'])
                if b64: st.image(b64, use_container_width=True)
                else: st.warning("Image Load Failed")
                st.write(f"**Name:** {item['name']}")
                if item['stock'] >= 100: st.success(f"Stock: {item['stock']:,}m")
                else: st.info(f"Stock: {item['stock']:,}m")
