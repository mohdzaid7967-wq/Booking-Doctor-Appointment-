from typing import Literal, List, Any
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from utils.llms import LLMModel
from toolkit.toolkits import *

import json
import re

class Router(TypedDict):
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str

class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    id_number: int
    next: str
    query: str
    current_reasoning: str

class DoctorAppointmentAgent:
    def __init__(self):
        llm_model = LLMModel()
        self.llm_model = llm_model.get_model()

    # ✅ FIXED SUPERVISOR NODE
    def supervisor_node(self, state: AgentState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
        
        print("STATE:", state)

        system_prompt = """
            You are a supervisor managing workers.

            Workers:
            - information_node → ONLY for availability or FAQs
            - booking_node → ONLY for booking/cancel/reschedule
            - FINISH → if task is completed

            RULES:
            - If user asks availability → call information_node ONCE
            - If user asks booking → call booking_node
            - If answer already given → return FINISH

            IMPORTANT:
            DO NOT call same node repeatedly.

            Return ONLY JSON:
            {
            "next": "information_node" or "booking_node" or "FINISH",
            "reasoning": "short explanation"
            }
            """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user id is {state['id_number']}"}
        ] + state["messages"]

        raw_response = self.llm_model.invoke(messages)

        print("RAW OUTPUT:", raw_response.content)

        # ✅ FIX: CLEAN JSON OUTPUT
        raw_text = raw_response.content.strip()
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)

        if not match:
            raise Exception("❌ No JSON found in output")

        json_text = match.group()

        try:
            response = json.loads(json_text)
        except:
            print("BAD JSON:", json_text)
            raise Exception("❌ JSON parsing failed")

        goto = response["next"]

        if goto == "FINISH":
            goto = END

        return Command(
            goto=goto,
            update={
                "next": goto,
                "current_reasoning": response["reasoning"]
            }
        )

    # ✅ INFORMATION NODE
    def information_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        print("called information node")

        system_prompt = ChatPromptTemplate.from_messages([
            ("system", "You provide doctor availability & FAQs."),
            ("placeholder", "{messages}")
        ])

        agent = create_react_agent(
            model=self.llm_model,
            tools=[check_availability_by_doctor, check_availability_by_specialization],
            prompt=system_prompt
        )

        result = agent.invoke(state)

        return Command(
            update={
                "messages": state["messages"] + [
                    AIMessage(content=result["messages"][-1].content, name="information_node")
                ]
            },
            goto=END
        )

    # ✅ BOOKING NODE
    def booking_node(self, state: AgentState) -> Command[Literal['supervisor']]:
        print("called booking node")

        system_prompt = ChatPromptTemplate.from_messages([
            ("system", "You handle booking, cancel, reschedule."),
            ("placeholder", "{messages}")
        ])

        agent = create_react_agent(
            model=self.llm_model,
            tools=[set_appointment, cancel_appointment, reschedule_appointment],
            prompt=system_prompt
        )

        result = agent.invoke(state)

        return Command(
            update={
                "messages": state["messages"] + [
                    AIMessage(content=result["messages"][-1].content, name="booking_node")
                ]
            },
            goto=END
        )

    
    def workflow(self):
        graph = StateGraph(AgentState)

        graph.add_node("supervisor", self.supervisor_node)
        graph.add_node("information_node", self.information_node)
        graph.add_node("booking_node", self.booking_node)

        graph.add_edge(START, "supervisor")

        return graph.compile()