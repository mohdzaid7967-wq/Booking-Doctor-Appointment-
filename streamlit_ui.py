import streamlit as st
import requests
import pandas as pd
import urllib3
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Doctor Appointment Assistant", page_icon="🩺", layout="wide")

API_URL = "http://127.0.0.1:8003/execute"

# --- Init Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! 👋 I am your Doctor Appointment Assistant. I can help you check availability and book, reschedule, or cancel your appointments. How can I help you today?"
    })

# --- Sidebar: ID Tracking & Booked Appointments ---
with st.sidebar:
    st.header("👤 Patient Dashboard")
    st.markdown("If you are a returning patient, please enter your ID to manage appointments. If you are a new patient, you will be assigned one when you book!")
    user_id_input = st.text_input("Enter your Patient ID (Optional):", "")
    
    st.divider()
    st.subheader("📅 Your Active Appointments")
    
    if user_id_input and user_id_input.strip().isdigit():
        try:
            df = pd.read_csv("data/doctor_availability.csv")
            user_appointments = df[
                (df["patient_to_attend"] == float(user_id_input)) & 
                (df["is_available"] == False)
            ].copy()

            if not user_appointments.empty:
                user_appointments["Status"] = "Booked 🟢"
                user_appointments.rename(columns={
                    "date_slot": "Date & Time",
                    "specialization": "Specialization",
                    "doctor_name": "Doctor Name",
                }, inplace=True)
                st.dataframe(
                    user_appointments[["Date & Time", "Specialization", "Doctor Name", "Status"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No active appointments found for this ID.")
        except Exception as e:
            st.error(f"Could not load appointment data: {e}")
    else:
        st.info("Enter a valid Patient ID to view your bookings.")
        
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main App: Chat Interface ---
st.title("🩺 Doctor Appointment Chatbot")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("E.g., Is there a general dentist available tomorrow?"):
    
    # 1. Store and display User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Checking..."):
            
            # Prepare Payload: Sends full history to FastAPI for memory
            payload = {
                "id_number": int(user_id_input) if user_id_input.strip().isdigit() else 0,
                "messages": st.session_state.messages
            }
            
            try:
                response = requests.post(API_URL, json=payload, verify=False, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("status") == "success":
                        reply = data.get("reply", "I'm sorry, I couldn't generate a response.")
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        
                        # If a booking occurred, auto-refresh to show it in the sidebar
                        if "successfully booked" in reply.lower() or "cancelled successfully" in reply.lower():
                            st.rerun()
                            
                    else:
                        error_msg = f"❌ Backend Error: {data.get('message', 'Unknown error')}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = f"❌ HTTP Error {response.status_code}: Could not process the request."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.Timeout:
                error_msg = "⏳ Request timed out. The server might be busy. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.ConnectionError:
                error_msg = "🔌 Could not connect. Make sure your FastAPI backend is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})