# -*- coding: utf-8 -*-
"""
California Housing Price Predictor
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="🏠 California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* ===== Developer Card Styles ===== */
    .developer-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #667eea;
    }
    
    .developer-avatar {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        margin-bottom: 1rem;
    }
    
    .developer-name {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .developer-id {
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.3rem 0;
    }
    
    .developer-role {
        color: #7f8c8d;
        font-size: 0.9rem;
        font-style: italic;
        margin: 0.3rem 0;
    }
    
    .footer-developer {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Load Model (แก้ไข path แล้ว!) ====================
@st.cache_resource
def load_model():
    """โหลดโมเดลและ scaler จากไฟล์"""
    try:
        # ✅ แก้ไข: ลบ model_files/ ออก เพราะไฟล์อยู่ที่ root
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        st.stop()

model, scaler, feature_names = load_model()

# ==================== Developer Info ====================
DEVELOPER_NAME = "นายจตุรภัทร สถาปิคานนท์"
DEVELOPER_ID = "024"
# 📌 วางรูปชื่อ "developer.jpg" ไว้ในโฟลเดอร์เดียวกับไฟล์นี้
# หรือเปลี่ยนเป็น URL รูปที่ต้องการ เช่น "https://..."
DEVELOPER_IMAGE = "developer.jpg"

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("## 🏠 California Housing")
    st.markdown("### Price Predictor")
    st.markdown("---")
    
    # ===== Developer Profile in Sidebar =====
    st.markdown("### 👨‍💻 ผู้พัฒนา")
    
    # แสดงรูปผู้พัฒนา (ใช้ placeholder ถ้าไม่มีไฟล์รูป)
    if os.path.exists(DEVELOPER_IMAGE):
        st.image(DEVELOPER_IMAGE, width=150)
    else:
        # ใช้ placeholder avatar จาก UI Avatars
        placeholder_url = f"https://ui-avatars.com/api/?name=Jaturapat&size=150&background=667eea&color=fff&bold=true"
        st.image(placeholder_url, width=150)
    
    st.markdown(f"""
    <div style='text-align: center; margin-top: 0.5rem;'>
        <p style='font-size: 1.1rem; font-weight: bold; margin: 0.3rem 0;'>
            {DEVELOPER_NAME}
        </p>
        <p style='font-size: 1rem; color: #a8d0ff; margin: 0.3rem 0;'>
            รหัสนักศึกษา: {DEVELOPER_ID}
        </p>
        <p style='font-size: 0.85rem; color: #bdc3c7; font-style: italic; margin: 0.3rem 0;'>
            ML with Python Developer
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    **โมเดล:** Random Forest Regressor  
    **Dataset:** California Housing  
    **Accuracy:** R² = 0.81
    """)
    
    with st.expander("📖 คำอธิบาย Features"):
        st.markdown("""
        - **MedInc**: รายได้เฉลี่ย (×$10,000)
        - **HouseAge**: อายุบ้านเฉลี่ย
        - **AveRooms**: จำนวนห้องเฉลี่ย
        - **AveBedrms**: จำนวนห้องนอนเฉลี่ย
        - **Population**: จำนวนประชากร
        - **AveOccup**: จำนวนคนต่อครัวเรือน
        - **Latitude**: ละติจูด
        - **Longitude**: ลองจิจูด
        """)

# ==================== Main Content ====================
st.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1 style='color: #2c3e50; font-size: 3rem; margin-bottom: 0.5rem;'>
        🏠 California Housing Price Predictor
    </h1>
    <p style='color: #7f8c8d; font-size: 1.2rem;'>
        ทำนายราคาบ้านในแคลิฟอร์เนียด้วย Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== Input Section ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📍 Location & Demographics")
    latitude = st.slider("🌐 Latitude", 32.0, 42.0, 35.0, 0.01)
    longitude = st.slider("🌐 Longitude", -124.0, -114.0, -119.0, 0.01)
    population = st.slider(" Population", 1.0, 30000.0, 1500.0, 10.0)
    ave_occup = st.slider("👨‍👩‍👧‍👦 Avg. Occupancy", 1.0, 10.0, 3.0, 0.1)

with col2:
    st.markdown("### 🏡 House Characteristics")
    med_inc = st.slider("💰 Median Income (×$10k)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("🏚️ House Age (years)", 1.0, 52.0, 20.0, 1.0)
    ave_rooms = st.slider(" Avg. Rooms", 1.0, 15.0, 5.0, 0.1)
    ave_bedrms = st.slider("️ Avg. Bedrooms", 0.5, 5.0, 1.1, 0.1)

# ==================== Prediction Button ====================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 ทำนายราคาบ้าน", use_container_width=True)

# ==================== Prediction Logic ====================
if predict_button:
    input_data = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_usd = prediction * 100000
    
    st.markdown("---")
    st.markdown(f"""
    <div class='result-card'>
        <h2 style='margin: 0;'>💰 ราคาบ้านที่ทำนายได้</h2>
        <div class='result-value'>${prediction_usd:,.0f}</div>
        <p style='font-size: 1.1rem; margin: 0;'>
            หรือประมาณ {prediction:,.2f} × $100,000
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 รายละเอียดการทำนาย")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #667eea; margin: 0;'> Predicted Price</h3>
            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                ${prediction_usd:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        price_per_room = prediction_usd / ave_rooms if ave_rooms > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #764ba2; margin: 0;'>🚪 Price/Room</h3>
            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                ${price_per_room:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        price_per_person = prediction_usd / population if population > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #f093fb; margin: 0;'>👥 Price/Person</h3>
            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                ${price_per_person:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        confidence = 81
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #4facfe; margin: 0;'>🎯 Confidence</h3>
            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                {confidence}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis',
            title='🔍 ความสำคัญของ Features ในการทำนาย'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 ดูข้อมูล Input ที่ใช้ทำนาย"):
        input_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data[0]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ==================== Footer with Developer Info ====================
st.markdown("---")

# แสดง Developer Card แบบเต็มใน Footer
st.markdown(f"""
<div class='footer-developer'>
    <div style='display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;'>
        <div>
            <img src='{DEVELOPER_IMAGE if os.path.exists(DEVELOPER_IMAGE) else "https://ui-avatars.com/api/?name=Jaturapat&size=120&background=667eea&color=fff&bold=true"}' 
                 class='developer-avatar' 
                 onerror="this.src='https://ui-avatars.com/api/?name=Jaturapat&size=120&background=667eea&color=fff&bold=true'">
        </div>
        <div style='text-align: left;'>
            <h3 style='color: #2c3e50; margin: 0;'>👨‍💻 พัฒนาโดย</h3>
            <p class='developer-name'>{DEVELOPER_NAME}</p>
            <p class='developer-id'>🆔 รหัสนักศึกษา: {DEVELOPER_ID}</p>
            <p class='developer-role'>📚 วิชา: การโปรแกรมสำหรับการเรียนรู้ด้วยเครื่องด้วยภาษาไพทอน</p>
        </div>
    </div>
    <hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #e0e0e0;'>
    <p style='color: #7f8c8d; margin: 0;'>
        📅 กรกฎาคม 2026 | 🛠️ Built with <b>Streamlit</b> + <b>scikit-learn</b> + <b>Plotly</b>
    </p>
</div>
""", unsafe_allow_html=True)