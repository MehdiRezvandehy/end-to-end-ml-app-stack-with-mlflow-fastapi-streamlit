import streamlit as st
import requests
import os
import plotly.graph_objects as go

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(page_title="Building Energy Load Predictor", page_icon="📚", layout="wide")
if "snow_done" not in st.session_state:
    st.snow()
    st.session_state.snow_done = True

# ------------------------------------------------------
# Ultra-Light Theme CSS
# ------------------------------------------------------
st.markdown("""
<style>
body, .stApp { background: #ffffff !important; color: #0f172a; }
.card { background: #ffffff; padding: 24px; border-radius: 16px; box-shadow: 0px 4px 12px rgba(0,0,0,0.08); margin-bottom: 20px; }
.info-card { background: #e0f2fe; padding: 14px; border-radius: 10px; text-align: center; color: #0f172a; }
.info-label { font-size: 14px; color: #475569; margin-bottom: -6px; }
.info-value { font-size: 18px; font-weight: bold; color: #0f172a; }
h2, h3 { color: #0f172a !important; }
.predict-btn-container { display: flex; justify-content: flex-end; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Title
# ------------------------------------------------------
st.title("🏠 Building Energy Load — Predictor")
st.markdown("<p style='font-size:16px; color:#475569;'>Binary classification on Heating Load using ML + FastAPI.</p>",
            unsafe_allow_html=True)

# ------------------------------------------------------
# Dataset & Methodology
# ------------------------------------------------------
st.markdown("""
<div class="card">
<h3>📘 Dataset & Methodology</h3>
<p style="color:#475569;">
This project uses the <b>UCI Energy Efficiency Dataset</b> (Tsanas & Xifara, 2012),
containing 768 building simulations with 8 structural features.  
We perform <b>binary classification</b> on Heating Load to determine if a building requires high heating energy.
</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Columns Layout
# ------------------------------------------------------
col1, col2 = st.columns([1.1, 1])

# ------------------------------------------------------
# Inputs (Left)
# ------------------------------------------------------
with col1:
    st.markdown("### Building Features")
    rel_compact = st.number_input("Relative Compactness", 0.01, 0.98, 0.75, 0.1)
    surface_area = st.number_input("Surface Area", 100.0, 1500.0, 670.0, step=10.0)
    wall_area = st.number_input("Wall Area", 50.0, 800.0, 300.0, step=5.0)
    roof_area = st.number_input("Roof Area", 20.0, 500.0, 200.0, step=5.0)
    overall_height = st.number_input("Overall Height", 3.5, 5.0, 3.5, step=0.1)
    orientation = st.selectbox("Orientation", [2, 3, 4, 5])
    glazing_area = st.number_input("Glazing Area", 0.0, 0.5, 0.25, step=0.1)
    glazing_dist = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# Prediction + Gauge (Right)
# ------------------------------------------------------
with col2:
    st.markdown('<div class="predict-btn-container">', unsafe_allow_html=True)
    predict_button = st.button("🔍 Predict High Load", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_button:
        with st.spinner("Contacting FastAPI backend..."):
            payload = {
                "rel_compact": rel_compact,
                "surface_area": surface_area,
                "wall_area": wall_area,
                "roof_area": roof_area,
                "overall_height": overall_height,
                "orientation": orientation,
                "glazing_area": glazing_area,
                "glazing_dist": glazing_dist
            }
            try:
                url = f"{os.getenv('API_URL','http://fastapi:8000').rstrip('/')}/predict"
                resp = requests.post(url, json=payload)
                resp.raise_for_status()
                st.session_state.pred = resp.json()
            except:
                st.warning("Backend error — using fallback prediction.")
                st.session_state.pred = {"predict_load_probability":0.73}

    st.markdown("## Prediction Results")

    if "pred" in st.session_state:
        prob = st.session_state.pred.get("predict_load_probability",0)

        # ---------------- Gauge Chart ----------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={'suffix':'%','font':{'size':36}},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':'#3b82f6'},
                'bgcolor':'#e0f2fe',
                'borderwidth':2,
                'bordercolor':'#ffffff',
                'steps': [
                    {'range':[0,50],'color':'#bae6fd'},
                    {'range':[50,75],'color':'#7dd3fc'},
                    {'range':[75,100],'color':'#38bdf8'}
                ]
            }
        ))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)

        a,b = st.columns(2)
        with a:
            st.markdown('<div class="info-card"><p class="info-label">Model</p><p class="info-value">RandomForest</p></div>',
                        unsafe_allow_html=True)
        with b:
            st.markdown('<div class="info-card"><p class="info-label">Scaled</p><p class="info-value">Yes</p></div>',
                        unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#64748b; margin-top:40px;'>Enter features and press <b>Predict High Load</b>.</p>",
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# Footer
# ------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>Powered by FastAPI + Streamlit</p>", unsafe_allow_html=True)
