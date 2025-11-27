from typing import List, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0)

class agentState(TypedDict):
    messages: List[HumanMessage]


def process(state : agentState) -> agentState:
    batch = [state["messages"]]
    response = llm.generate(batch)
    text = response.generations[0][0].text
    print(f"AI : {text}")
    return state

graph = StateGraph(agentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("USER: ")
agent.invoke({
    "messages" : [HumanMessage(content=user_input)]
})
