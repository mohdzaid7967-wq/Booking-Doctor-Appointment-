import streamlit as st
import requests
import re

API_URL = "http://127.0.0.1:8003/execute" 

st.title("🩺 Doctor Appointment System")

user_id = st.text_input("Enter your ID number:", "")
query = st.text_area("Enter your query:", "Can you check if a dentist is available tomorrow at 10 AM?")

if st.button("Submit Query"):
    if user_id and query:
        try:
            response = requests.post(
                API_URL,
                json={'messages': query, 'id_number': int(user_id)},
                verify=False
            )
            
            if response.status_code == 200:
                st.success("Response Received:")
                
                data = response.json()
                raw = data["messages"]

                # 🔥 Extract only AI message cleanly
                match = re.search(r"AIMessage\(content='(.*?)'", raw)
                
                if match:
                    st.success(match.group(1))  # clean output
                else:
                    st.write(raw)  # fallback

            else:
                st.error(f"Error {response.status_code}: Could not process the request.")
                
        except Exception as e:
            st.error(f"Exception occurred: {e}")
    else:
        st.warning("Please enter both ID and query.")