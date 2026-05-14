from fastapi import FastAPI
from pydantic import BaseModel
from agent import DoctorAppointmentAgent
from langchain_core.messages import HumanMessage
import os

# SSL fix (optional)
os.environ.pop("SSL_CERT_FILE", None)

app = FastAPI()

# Request model
class UserQuery(BaseModel):
    id_number: int
    messages: str

# Initialize agent + graph ONLY ONCE ✅
agent = DoctorAppointmentAgent()
app_graph = agent.workflow()

@app.post("/execute")
def execute_agent(user_input: UserQuery):
    
    try:
        # Prepare input messages
        user_messages = [
            HumanMessage(content=user_input.messages)
        ]

        query_data = {
            "messages": user_messages,
            "id_number": user_input.id_number,
            "next": "",
            "query": "",
            "current_reasoning": "",
        }

        # Run graph
        response = app_graph.invoke(query_data, config={"recursion_limit": 20})

        # Safe return
        return {
            "status": "success",
            "messages": str(response.get("messages", "No response")),
            "next": response.get("next", ""),
            "reasoning": response.get("current_reasoning", "")
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }