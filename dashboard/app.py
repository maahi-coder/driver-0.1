import streamlit as st
import pandas as pd
import time
import os
import sys

# Page Config
st.set_page_config(
    page_title="Driver Monitoring Dashboard",
    page_icon="🚗",
    layout="wide",
)

# Title
st.title("🚗 Driver Cognitive Monitoring Dashboard")

# Paths
# Assuming running from root: streamlit run dashboard/app.py
LOG_FILE = os.path.join("data", "logs.csv")

if not os.path.exists(LOG_FILE):
    st.error(f"Log file not found at {LOG_FILE}. Please run the monitoring system first.")
    st.stop()

# Auto-refresh logic
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = time.time()

def load_data():
    try:
        df = pd.read_csv(LOG_FILE)
        return df
    except Exception as e:
        st.error(f"Error reading log file: {e}")
        return pd.DataFrame()

# Main Loop placeholder for real-time emulation
placeholder = st.empty()

# Sidebar
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)

while True:
    df = load_data()
    
    with placeholder.container():
        if df.empty:
            st.warning("No data returned yet.")
        else:
            # KPIS
            latest = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Fatigue Score", f"{latest.get('fatigue_score', 0):.1f}%")
            with col2:
                alert_status = "ACTIVE 🚨" if latest.get('alert_active', 0) == 1 else "Normal"
                st.metric("Alert Status", alert_status)
            with col3:
                st.metric("Head Pitch", f"{latest.get('head_pitch', 0):.1f}")
            with col4:
                st.metric("Head Yaw", f"{latest.get('head_yaw', 0):.1f}")

            # Charts
            st.subheader("Fatigue Trend")
            
            # Simple line chart
            if 'fatigue_score' in df.columns:
                st.line_chart(df['fatigue_score'].tail(100))
            
            # Recent Events Log
            st.subheader("Recent Alerts")
            alerts = df[df['alert_active'] == 1].tail(10)
            if not alerts.empty:
                st.dataframe(alerts[['timestamp', 'fatigue_score', 'head_pitch', 'head_yaw']])
            else:
                st.info("No recent alerts.")

    time.sleep(refresh_rate)
