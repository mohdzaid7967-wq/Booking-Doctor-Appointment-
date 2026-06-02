from typing import Literal, Any
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from utils.llms import LLMModel
from toolkit.toolkits import (
    check_availability_by_doctor,
    check_availability_by_specialization,
    set_appointment,
    cancel_appointment,
    reschedule_appointment,
)

import json
import re


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    id_number: int
    next: str
    current_reasoning: str


SUPERVISOR_SYSTEM = """
You are a supervisor managing a doctor appointment system.

Workers available:
- information_node: handles availability checks and FAQs about doctors/schedules
- booking_node: handles booking, cancelling, and rescheduling appointments
- FINISH: use when the user's request has been fully answered

Rules:
1. Route to information_node for any availability or schedule query.
2. Route to booking_node for any book/cancel/reschedule request.
3. If the last message already contains a complete answer, return FINISH.
4. Never route to the same node twice for the same query.

Respond ONLY with valid JSON. No explanation, no markdown:
{"next": "information_node" | "booking_node" | "FINISH", "reasoning": "one line"}
"""

INFORMATION_SYSTEM = """
You are a helpful medical receptionist AI for a doctor appointment system.

Your job:
- Check doctor availability and answer schedule questions.
- Remember the context of the conversation. Look closely at previous messages to understand follow-up questions.
- Always respond in clear, friendly natural language.

Important rules:
- If the user does not provide a date, ask for it (format: DD-MM-YYYY).
- Available specializations: general_dentist, cosmetic_dentist, oral_surgeon, orthodontist, pediatric_dentist, emergency_dentist.
- Always call the tool before saying no slots are available.
"""

BOOKING_SYSTEM = """
You are a medical booking assistant AI.

Your job:
- Book, cancel, or reschedule doctor appointments
- Look closely at previous messages to pull doctor names, dates, and times if the user already mentioned them.
- Always respond in friendly natural language.

Important rules:
- To book: you need doctor_name, date (DD-MM-YYYY), and time (HH:MM).
- When passing the `doctor_name` to the tool, do NOT include "Dr." (e.g., pass "john doe", not "Dr. John Doe").
- IF PATIENT DOES NOT HAVE AN ID: Pass 0 as the `patient_id` to the booking tool. It will generate a new ID for them.
- YOU MUST explicitly highlight the new `patient_id` returned by the tool to the user so they can save it.
- For cancel or reschedule: a valid `patient_id` is REQUIRED. If they haven't given it, ask them for it.
"""


class DoctorAppointmentAgent:
    def __init__(self):
        llm_model = LLMModel()
        self.llm = llm_model.get_model()

    def supervisor_node(self, state: AgentState) -> Command[Literal["information_node", "booking_node", "__end__"]]:
        recent_messages = state["messages"][-6:]

        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {"role": "user", "content": f"Patient ID in session context: {state.get('id_number', 0)}"}
        ] + [
            {"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content}
            for m in recent_messages
        ]

        raw_response = self.llm.invoke(messages)
        raw_text = raw_response.content.strip()

        match = re.search(r'\{[^{}]*"next"[^{}]*\}', raw_text, re.DOTALL)
        if not match:
            lower = raw_text.lower()
            if any(k in lower for k in ["finish", "completed", "answered"]):
                return Command(goto=END, update={"next": "FINISH", "current_reasoning": "Supervisor fallback to FINISH"})
            goto = "information_node"
            reasoning = "Fallback routing"
        else:
            try:
                response = json.loads(match.group())
                goto = response.get("next", "information_node")
                reasoning = response.get("reasoning", "")
            except Exception:
                goto = "information_node"
                reasoning = "JSON parse fallback"

        if goto == "FINISH":
            goto = END

        return Command(
            goto=goto,
            update={"next": str(goto), "current_reasoning": reasoning}
        )

    def information_node(self, state: AgentState) -> Command[Literal["__end__"]]:
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", INFORMATION_SYSTEM),
            ("placeholder", "{messages}")
        ])

        agent = create_react_agent(
            model=self.llm,
            tools=[check_availability_by_doctor, check_availability_by_specialization],
            prompt=system_prompt,
        )

        result = agent.invoke(
            {"messages": state["messages"][-6:]},
            config={"recursion_limit": 10}
        )

        last_content = result["messages"][-1].content

        return Command(
            update={
                "messages": [
                    AIMessage(content=last_content, name="information_node")
                ]
            },
            goto=END
        )

    def booking_node(self, state: AgentState) -> Command[Literal["__end__"]]:
        patient_id = state.get("id_number", 0)
        id_context = f"\n\nCurrent patient ID provided by user: {patient_id}" if patient_id else "\n\nNo patient ID provided. If booking, pass 0 for patient_id."

        system_with_id = BOOKING_SYSTEM + id_context

        system_prompt = ChatPromptTemplate.from_messages([
            ("system", system_with_id),
            ("placeholder", "{messages}")
        ])

        agent = create_react_agent(
            model=self.llm,
            tools=[set_appointment, cancel_appointment, reschedule_appointment],
            prompt=system_prompt,
        )

        result = agent.invoke(
            {"messages": state["messages"][-6:]},
            config={"recursion_limit": 15}
        )

        last_content = result["messages"][-1].content

        return Command(
            update={
                "messages": [
                    AIMessage(content=last_content, name="booking_node")
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