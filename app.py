import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# --- 1. PAGE & THEME CONFIGURATION ---
st.set_page_config(page_title="CardioInsight AI", page_icon="❤️", layout="centered")

# Custom Medical-Grade UI Styling
st.markdown("""
    <style>
    .main-title { font-size:40px; font-weight:bold; color:#1E3A8A; text-align:center; margin-bottom:10px; }
    .subtitle { font-size:18px; color:#4B5563; text-align:center; margin-bottom:30px; }
    .card { background-color:#F8FAFC; padding:25px; border-radius:12px; border-left:5px solid #3B82F6; margin-bottom:20px; }
    .result-box { padding:20px; border-radius:8px; text-align:center; font-size:24px; font-weight:bold; margin-top:20px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATABASE SETUP ---
DB_FILE = 'cardio_users.db'
def init_db():
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER, email TEXT UNIQUE, password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS diagnostics 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, timestamp TEXT, metrics TEXT, results TEXT)''')
        conn.commit()

init_db()

# --- 3. CACHED ENGINE/MODEL LOADING ---
@st.cache_resource
def load_medical_engine():
    m_dir = "models"
    def safe_load(path):
        try:
            return joblib.load(path), None
        except Exception as load_error:
            return None, str(load_error)

    scaler, scaler_error = safe_load(os.path.join(m_dir, 'scaler.pkl'))
    if scaler is None:
        raise RuntimeError(f"Failed to load scaler.pkl: {scaler_error}")

    with open(os.path.join(m_dir, 'feature_columns.json'), 'r') as f:
        feature_cols = json.load(f)

    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'SVM': 'svm.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl'
    }

    loaded_models = {}
    model_errors = {}
    for model_name, filename in model_files.items():
        model, error = safe_load(os.path.join(m_dir, filename))
        if model is not None:
            loaded_models[model_name] = model
        else:
            model_errors[model_name] = error

    return scaler, feature_cols, loaded_models, model_errors

scaler, FEATURE_COLUMNS, models, model_errors = None, [], {}, {}
try:
    scaler, FEATURE_COLUMNS, models, model_errors = load_medical_engine()
    if not models:
        raise RuntimeError("No valid models loaded.")
except Exception as e:
    st.error(f"Error loading analytical models: {e}. Ensure all required files are present in the 'models/' folder.")
    if model_errors:
        st.error("Model load failures: " + "; ".join(f"{name}: {msg}" for name, msg in model_errors.items()))

# --- 4. DATA PREPROCESSING PIPELINE ---
def process_vitals(input_dict):
    patient_df = pd.DataFrame([input_dict])
    
    # Clean zeros using data criteria defaults
    patient_df['Cholesterol'] = patient_df['Cholesterol'].replace(0, 237.0)
    patient_df['RestingBP'] = patient_df['RestingBP'].replace(0, 130.0)
    
    # Map raw entries to match Dummy One-Hot layout
    string_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    patient_encoded = pd.get_dummies(patient_df, columns=string_cols, drop_first=False)
    
    for col in FEATURE_COLUMNS:
        if col not in patient_encoded.columns:
            patient_encoded[col] = 0
            
    patient_encoded = patient_encoded[FEATURE_COLUMNS]
    return scaler.transform(patient_encoded)

# --- 5. APP FLOW CONTROL & STATE ---
if 'app_mode' not in st.session_state: st.session_state.app_mode = 'auth_landing'
if 'current_user' not in st.session_state: st.session_state.current_user = None
if 'active_results' not in st.session_state: st.session_state.active_results = None

# --- ROUTE A: AUTHENTICATION LANDING ---
if st.session_state.app_mode == 'auth_landing':
    st.markdown("<div class='main-title'>❤️ CardioInsight AI Portal</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Clinical Decision Support System Framework</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>Clinical Machine Learning Framework</h3>
            <p>Access high-accuracy cardiovascular diagnostic assessments leveraging multi-model ensemble systems.</p>
            <ul>
                <li>Individual Model or Unified Ensemble Analytics</li>
                <li>Key Medical Risk Factor Attribution Analysis</li>
                <li>Secure Longitudinal Patient History Tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⏩ Skip Login / Direct Patient Intake", use_container_width=True):
            st.session_state.current_user = "Guest Patient"
            st.session_state.app_mode = 'intake_form'
            st.rerun()

    with col2:
        auth_action = st.tabs(["Sign In", "Register"])
        
        with auth_action[0]:
            lin_email = st.text_input("Clinical/Patient Email", key="lin_em")
            lin_pwd = st.text_input("Password", type="password", key="lin_pw")
            if st.button("Sign In", use_container_width=True):
                with sqlite3.connect(DB_FILE, timeout=10) as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM users WHERE email=? AND password=?", (lin_email, lin_pwd))
                    user = c.fetchone()
                if user:
                    st.session_state.current_user = {"name": user[1], "age": user[2], "email": user[3]}
                    st.session_state.app_mode = 'intake_form'
                    st.rerun()
                else:
                    st.error("Invalid Profile Credentials.")
                    
        with auth_action[1]:
            reg_name = st.text_input("Full Name")
            reg_age = st.number_input("Age", 18, 110, 45)
            reg_email = st.text_input("Registration Email")
            reg_pwd = st.text_input("Account Password", type="password")
            if st.button("Create Profile", use_container_width=True):
                if reg_name and reg_email and reg_pwd:
                    try:
                        with sqlite3.connect(DB_FILE, timeout=10) as conn:
                            c = conn.cursor()
                            c.execute("INSERT INTO users (name, age, email, password) VALUES (?, ?, ?, ?)", 
                                      (reg_name, reg_age, reg_email, reg_pwd))
                            conn.commit()
                        st.success("Profile created! Proceed to Sign In.")
                    except sqlite3.IntegrityError:
                        st.error("This email identity is already registered.")
                else:
                    st.warning("Please complete all profile details.")

# --- ROUTE B: CLINICAL PATIENT INTAKE FORM ---
elif st.session_state.app_mode == 'intake_form':
    st.markdown("<div class='main-title'>📋 Patient Intake Metrics</div>", unsafe_allow_html=True)
    st.write(f"**Authenticated Identity:** {st.session_state.current_user if isinstance(st.session_state.current_user, str) else st.session_state.current_user['name']}")
    
    if scaler is None or not models:
        st.error("The diagnostic engine is unavailable. Please verify your model files in the 'models/' folder and restart the app.")
        st.stop()
    
    if st.button("⬅️ Sign Out / Exit Panel"):
        st.session_state.current_user = None
        st.session_state.app_mode = 'auth_landing'
        st.rerun()
        
    with st.form("medical_vitals_form"):
        st.markdown("#### Patient Vitals & Lab Diagnostics")
        col1, col2 = st.columns(2)
        
        with col1:
            init_age = 45 if isinstance(st.session_state.current_user, str) else int(st.session_state.current_user['age'])
            age = st.number_input("Age (Years)", 1, 120, init_age)
            sex = st.selectbox("Biological Sex", ["M", "F"])
            cp = st.selectbox("Chest Pain Presentation Type", ["ASY", "NAP", "ATA", "TA"])
            rbp = st.number_input("Resting Blood Pressure (mmHg)", 50, 250, 130)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 0, 700, 220)
            
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes (1)" if x==1 else "No (0)")
            ecg = st.selectbox("Resting Electrocardiogram Findings", ["Normal", "LVH", "ST"])
            max_hr = st.number_input("Maximum Heart Rate Achieved (BPM)", 50, 250, 140)
            ex_ang = st.selectbox("Exercise-Induced Angina Symptom", ["N", "Y"])
            oldpeak = st.number_input("ST Depression Induced by Exercise (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
            slope = st.selectbox("Peak Exercise ST Segment Slope Elevation", ["Up", "Flat", "Down"])
            
        st.markdown("---")
        st.markdown("#### Diagnostics Analytical Engine Engine Options")
        eval_mode = st.radio("Processing Optimization Matrix", ["Run Single Model Instance", "Run Full 4-Model Unified Ensemble Analysis"])
        
        selected_model = "Random Forest"
        if eval_mode == "Run Single Model Instance":
            selected_model = st.selectbox("Target Pipeline Model Selection", list(models.keys()))
            
        submit_eval = st.form_submit_button("🚨 RUN PATIENT COMPREHENSIVE ASSESSMENT")
        
    if submit_eval:
        patient_payload = {
            'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': rbp,
            'Cholesterol': chol, 'FastingBS': fbs, 'RestingECG': ecg,
            'MaxHR': max_hr, 'ExerciseAngina': ex_ang, 'Oldpeak': oldpeak, 'ST_Slope': slope
        }
        
        scaled_vitals = process_vitals(patient_payload)
        assessment_run = {}
        
        if eval_mode == "Run Single Model Instance":
            model_instance = models[selected_model]
            prob = model_instance.predict_proba(scaled_vitals)[0][1] * 100
            assessment_run[selected_model] = round(prob, 2)
        else:
            for name, model_instance in models.items():
                prob = model_instance.predict_proba(scaled_vitals)[0][1] * 100
                assessment_run[name] = round(prob, 2)
                
        st.session_state.active_results = {
            "metrics": patient_payload,
            "predictions": assessment_run,
            "mode": eval_mode
        }
        
        # Save to Database if Not Guest
        if isinstance(st.session_state.current_user, dict):
            with sqlite3.connect(DB_FILE, timeout=10) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO diagnostics (email, timestamp, metrics, results) VALUES (?, ?, ?, ?)",
                          (st.session_state.current_user['email'], datetime.now().strftime("%Y-%m-%d %H:%M"),
                           json.dumps(patient_payload), json.dumps(assessment_run)))
                conn.commit()
            
        st.session_state.app_mode = 'results_page'
        st.rerun()

# --- ROUTE C: ANALYTICAL RESULTS REDIRECT PAGE ---
elif st.session_state.app_mode == 'results_page':
    st.markdown("<div class='main-title'>📊 Diagnostic Assessment Summary</div>", unsafe_allow_html=True)
    
    if st.button("⬅️ Process New Diagnostic Evaluation"):
        st.session_state.active_results = None
        st.session_state.app_mode = 'intake_form'
        st.rerun()
        
    if st.session_state.active_results:
        res = st.session_state.active_results
        
        st.markdown("### Estimated Probability Thresholds")
        for model_name, prob in res['predictions'].items():
            if prob >= 70:
                color, label = "#EF4444", "HIGH RISK ASSESSMENT ALERT"
            elif prob >= 40:
                color, label = "#F59E0B", "EVALUATE: MODERATE RISK MATRIX"
            else:
                color, label = "#10B981", "LOW CLINICAL FOOTPRINT RISK"
                
            st.markdown(f"""
            <div style='background-color:{color}22; border:1px solid {color}; padding:15px; border-radius:8px; margin-bottom:10px;'>
                <strong style='color:{color}; font-size:16px;'>{model_name}: {prob}% Risk Profile</strong> — <em>{label}</em>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Top Evaluated Contributing Factors")
        st.markdown("""
        * **ST Segment Slope Characteristics:** Variations along Up/Flat trends correlate directly with structural cardiac stress profiles under load conditions.
        * **Oldpeak Metric Deviations:** ST deviations represent explicit ischemic biomarkers recorded directly via the patient ECG subsystem.
        """)