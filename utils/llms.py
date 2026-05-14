import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMModel:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file")

    def get_model(self):
        return ChatGroq(
            api_key=self.api_key,
            model="llama-3.3-70b-versatile",
            temperature=0,          # ✅ VERY IMPORTANT (stable JSON)
            max_tokens=512
        )