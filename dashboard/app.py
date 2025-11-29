"""
GHIA Doctor Dashboard - Streamlit Frontend
A simple but effective dashboard for doctors to view patient intakes.
"""
import streamlit as st
import requests
from datetime import datetime
import json

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="GHIA - Doctor Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .urgent-card {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .moderate-card {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .routine-card {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .hindi-text {
        font-size: 1.1em;
        color: #1565c0;
    }
    .stat-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_risk_color(risk_level: str) -> str:
    """Get color based on risk level"""
    colors = {
        "urgent": "🔴",
        "moderate": "🟡", 
        "routine": "🟢"
    }
    return colors.get(risk_level, "⚪")


def get_card_class(risk_level: str) -> str:
    """Get CSS class based on risk level"""
    return f"{risk_level}-card"


def fetch_intakes():
    """Fetch all intakes from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/dashboard/intakes", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.status_code}")
            return {"intakes": [], "count": 0}
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Cannot connect to backend. Make sure the server is running on port 8000.")
        return {"intakes": [], "count": 0}
    except Exception as e:
        st.error(f"Error: {e}")
        return {"intakes": [], "count": 0}


def fetch_stats():
    """Fetch dashboard statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/dashboard/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"total_intakes": 0, "urgent": 0, "moderate": 0, "routine": 0}
    except:
        return {"total_intakes": 0, "urgent": 0, "moderate": 0, "routine": 0}


def run_demo():
    """Run demo intake"""
    try:
        response = requests.post(f"{BACKEND_URL}/api/intake/demo", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Demo failed: {e}")
        return None


def main():
    # Header
    st.title("🏥 GHIA - Gramin Health Intake Assistant")
    st.markdown("*AI-powered multi-agent health intake system for rural India*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🎯 Run Demo Intake", use_container_width=True):
            with st.spinner("Processing demo intake..."):
                result = run_demo()
                if result:
                    st.success(f"✅ Demo intake created! ID: {result.get('id')}")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Risk Legend")
        st.markdown("🔴 **Urgent** - Immediate attention")
        st.markdown("🟡 **Moderate** - Within 24-48 hrs")
        st.markdown("🟢 **Routine** - Regular appointment")
        
        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown(f"[API Docs]({BACKEND_URL}/docs)")
    
    # Stats Row
    stats = fetch_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", stats.get("total_intakes", 0))
    with col2:
        st.metric("🔴 Urgent", stats.get("urgent", 0))
    with col3:
        st.metric("🟡 Moderate", stats.get("moderate", 0))
    with col4:
        st.metric("🟢 Routine", stats.get("routine", 0))
    
    st.markdown("---")
    
    # Filter tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 All Cases", "🔴 Urgent", "🟡 Moderate", "🟢 Routine"])
    
    # Fetch all intakes
    data = fetch_intakes()
    intakes = data.get("intakes", [])
    
    def display_intakes(filter_risk=None):
        """Display intake cards"""
        filtered = intakes if filter_risk is None else [i for i in intakes if i.get("risk_level") == filter_risk]
        
        if not filtered:
            st.info("No cases found" + (f" with {filter_risk} risk level" if filter_risk else ""))
            return
        
        for intake in filtered:
            risk = intake.get("risk_level", "routine")
            risk_icon = get_risk_color(risk)
            
            with st.expander(
                f"{risk_icon} Case #{intake.get('id')} - {intake.get('chief_complaint', 'Unknown')} | {risk.upper()}",
                expanded=(risk == "urgent")
            ):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### 🇬🇧 English Summary")
                    st.write(intake.get("summary_english", "No summary available"))
                    
                    st.markdown("#### 📝 Details")
                    st.write(f"**Chief Complaint:** {intake.get('chief_complaint', 'N/A')}")
                    st.write(f"**Duration:** {intake.get('duration', 'N/A')}")
                    st.write(f"**Severity:** {intake.get('severity', 'N/A')}")
                    
                    symptoms = intake.get("symptoms", [])
                    if symptoms:
                        st.write("**Symptoms:**")
                        for s in symptoms:
                            if isinstance(s, dict):
                                st.write(f"  - {s.get('symptom', 'Unknown')}")
                            else:
                                st.write(f"  - {s}")
                
                with col_b:
                    st.markdown("#### 🇮🇳 Hindi Summary")
                    st.markdown(f'<p class="hindi-text">{intake.get("summary_hindi", "सारांश उपलब्ध नहीं")}</p>', 
                               unsafe_allow_html=True)
                    
                    st.markdown("#### 💊 Recommended Action")
                    st.info(intake.get("recommended_action", "Consult physician"))
                    
                    assoc = intake.get("associated_symptoms", [])
                    if assoc:
                        st.write("**Associated Symptoms:**")
                        st.write(", ".join(assoc) if isinstance(assoc, list) else assoc)
                    
                    st.caption(f"Recorded: {intake.get('created_at', 'Unknown')}")
    
    with tab1:
        display_intakes()
    
    with tab2:
        display_intakes("urgent")
    
    with tab3:
        display_intakes("moderate")
    
    with tab4:
        display_intakes("routine")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center>GHIA - Multi-Agent Healthcare System | Built for Rural India 🇮🇳</center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
