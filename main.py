from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from agent import DoctorAppointmentAgent
from langchain_core.messages import HumanMessage, AIMessage
import os

os.environ.pop("SSL_CERT_FILE", None)

app = FastAPI()

class UserQuery(BaseModel):
    id_number: Optional[int] = 0
    messages: List[Dict[str, str]]

# Initialize once at startup
agent = DoctorAppointmentAgent()
app_graph = agent.workflow()

@app.post("/execute")
def execute_agent(user_input: UserQuery):
    try:
        # Reconstruct LangChain Message History from Chatbot UI
        langchain_msgs = []
        for msg in user_input.messages:
            if msg["role"] == "user":
                langchain_msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_msgs.append(AIMessage(content=msg["content"]))

        query_data = {
            "messages": langchain_msgs,
            "id_number": user_input.id_number or 0,
            "next": "",
            "current_reasoning": "",
        }

        response = app_graph.invoke(query_data, config={"recursion_limit": 25})

        messages = response.get("messages", [])
        last_ai_content = "I'm sorry, I could not process your request. Please try again."

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                last_ai_content = msg.content.strip()
                break

        return {
            "status": "success",
            "reply": last_ai_content,
            "reasoning": response.get("current_reasoning", "")
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }