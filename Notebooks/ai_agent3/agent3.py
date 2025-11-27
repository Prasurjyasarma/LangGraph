# This is a ReAct agent 

"""
BaseMessage : it is the parent class for all message 
add_messages : it is a reducer function that appends to the current state 
Annotated : it is used to add metadata to the a type 
Sequence : it tells that the message can be list or tuple 

"""


from typing import List, TypedDict, Union, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0)

class agentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages] # This will append to the current state. 


@tool 
def add(a: int, b:int) -> int:
    """ This will return the addition of two numbers """
    return a + b

@tool
def sub(a: int, b:int) -> int:
    """ This will return the subtarction of two numbers """
    return a - b

tools = [add, sub]

model_with_tools = model.bind_tools(tools) # This is how are bind tools to the model

def model_call(state : agentState) -> agentState:
    prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query"
    )
    response = model_with_tools.invoke([prompt] + state["messages"])
    return {"messages": [response]}


# This is a loop function it will keep on calling the tools until and unless it is not required. 
def should_continue(state: agentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(agentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"] [-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {
    "messages" : [("user", "Add 12 and 15, After that do 100 - 89")]
}
print_stream(app.stream(inputs, stream_mode="values"))