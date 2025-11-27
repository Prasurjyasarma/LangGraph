# THIS BOT HAS MEMORY 

from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0)

class agentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

def process(state: agentState) -> agentState:
    """This node will solve the request you input"""

    batch = [state["messages"]]
    response = llm.generate(batch)
    text = response.generations[0][0].text
    state["messages"].append(AIMessage(content=text))
    print(f"AI : {text}")
    return state


graph = StateGraph(agentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []

user_input = input("USER: ")

while user_input.lower() not in ["exit", "quit"]:
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({
        "messages" : conversation_history
    })

    conversation_history = result["messages"]
    user_input = input("USER: ")


