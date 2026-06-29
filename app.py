# app.py
import streamlit as st
import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 1. PAGE SETUP & CUSTOM CSS
st.set_page_config(page_title="Real F1 DRS Predictor", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for F1 aesthetics
st.markdown("""
    <style>
    .f1-title {
        font-size: 48px !important;
        font-weight: 900;
        color: #E10600; /* Official F1 Red */
        margin-bottom: -15px;
    }
    .subtitle {
        font-size: 18px;
        color: #B0B0B0;
        margin-bottom: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #E10600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="f1-title">🏎️ Pit Wall: Live DRS Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by real Brazil GP 2023 telemetry & SMOTE-balanced Random Forest.</p>', unsafe_allow_html=True)
st.divider()

# 2. MODEL TRAINING (Cached)
@st.cache_resource(show_spinner=False)
def train_real_f1_model():
    fastf1.Cache.enable_cache('f1_cache')
    session = fastf1.get_session(2023, 'Brazil', 'R')
    session.load(telemetry=False, weather=False) 

    laps = session.laps.copy()
    
    valid_laps = laps[(laps['LapNumber'] > 1) & (laps['PitOutTime'].isnull()) & (laps['PitInTime'].isnull())]
    valid_laps['Pos_Change'] = valid_laps.groupby('Driver')['Position'].diff()
    valid_laps['Overtake_Success'] = (valid_laps['Pos_Change'] < 0).astype(int)
    
    lap_avg_speed = valid_laps.groupby('LapNumber')['SpeedST'].transform('mean')
    valid_laps['Speed_Advantage_kmh'] = valid_laps['SpeedST'] - lap_avg_speed
    
    valid_laps['LapTime_Sec'] = valid_laps['LapTime'].dt.total_seconds()
    lap_avg_time = valid_laps.groupby('LapNumber')['LapTime_Sec'].transform('mean')
    valid_laps['Pace_Advantage_Sec'] = lap_avg_time - valid_laps['LapTime_Sec']
    
    df_real = valid_laps[['Speed_Advantage_kmh', 'TyreLife', 'Pace_Advantage_Sec', 'Overtake_Success']].dropna()
    df_real.rename(columns={'TyreLife': 'Tire_Age'}, inplace=True)
    
    X = df_real[['Speed_Advantage_kmh', 'Tire_Age', 'Pace_Advantage_Sec']]
    y = df_real['Overtake_Success']
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_balanced, y_balanced)
    
    return rf_model, df_real

with st.spinner("📡 Connecting to AWS F1 Servers..."):
    model, df_real = train_real_f1_model()

# 3. SIDEBAR CONTROLS
with st.sidebar:
    st.image("https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Track%20icons%204x3/Brazil.png", caption="Interlagos Circuit")
    st.header("⚙️ Telemetry Inputs")
    st.markdown("Adjust the live delta to the car ahead:")
    
    current_speed_adv = st.slider("Speed Trap Advantage (km/h)", -15.0, 25.0, 8.0, 1.0)
    current_tire_age = st.slider("Current Tire Age (Laps)", 1, 40, 10, 1)
    current_pace_adv = st.slider("Pace Advantage (Seconds)", -3.0, 3.0, 0.5, 0.1)

# 4. MAIN DASHBOARD AREA
# Display inputs as a HUD
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Speed Delta", value=f"{current_speed_adv} km/h")
with col2:
    st.metric(label="Tire Age", value=f"{current_tire_age} Laps")
with col3:
    st.metric(label="Pace Delta", value=f"{current_pace_adv} sec")

st.markdown("<br>", unsafe_allow_html=True)

# Package inputs and predict
input_data = pd.DataFrame([[current_speed_adv, current_tire_age, current_pace_adv]], 
                          columns=['Speed_Advantage_kmh', 'Tire_Age', 'Pace_Advantage_Sec'])
probability = model.predict_proba(input_data)[0][1]

# Display results dynamically
st.subheader("🎯 Overtake Probability Engine")
col_prob, col_bar = st.columns([1, 3])

with col_prob:
    st.metric(label="Success Rate", value=f"{probability * 100:.1f}%")

with col_bar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(float(probability))

# Dynamic Strategy Call
st.markdown("### 📻 Race Engineer Strategy Call")
if probability < 0.35:
    st.error("🟥 **HOLD POSITION.** We lack the overspeed. Save the battery and tires.")
elif probability < 0.60:
    st.warning("🟨 **50/50 CHANCE.** You'll need to brake extremely late. Driver's call.")
else:
    st.success("🟩 **SEND IT!** Overtake delta is optimal. Full battery deployment!")

# 5. DATA INSIGHTS EXPANDER
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Look Under The Hood (Data & Model Analytics)"):
    total_laps = len(df_real)
    total_overtakes = df_real['Overtake_Success'].sum()
    st.write(f"- **Data Source:** {total_laps} valid racing laps from the 2023 Brazilian Grand Prix.")
    st.write(f"- **Real World Imbalance:** Only **{total_overtakes}** overtakes occurred natively ({((total_overtakes/total_laps)*100):.1f}%).")
    st.write("- **SMOTE Application:** Synthetic overtakes were mathematically generated in training to balance the Random Forest and prevent zero-variance guessing.")