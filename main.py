from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

llm = OllamaLLM(model="llama3.2")

class MessageClassifier(BaseModel):
    messageType: Literal["reader", "writer"] = Field(
        ...,
        description="Classify if the message requires an reader or writer response"
    )

#state control
class State(TypedDict):
    messages: Annotated[list, add_messages]
    messageType: str | None

def classify_message(state: State):
    lastMessage = state["messages"][-1]

    result = llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either 'reader' or 'writer'. Respond with ONLY the word 'reader' or 'writer', DO NOT respond with anything else.
            - 'reader': if the user asks for emotional support, therapy, deals with feelings, or personal problems
            - 'writer': if the user asks for facts, information, logical analysis, or practical solutions
            """ 
        },
        {"role": "user", "content": lastMessage.content}
    ])
    
    messageType = "writer"
    if "reader" in result.lower():
        messageType = "reader"
    
    return {"messageType": messageType}

def reading_agent(state: State):
    msgHistory = state["messages"][:-1][-20:]  # all messages except the last one, max 20.
    lastMessage = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         } 
         ] + msgHistory + [
        {
            "role": "user",
            "content": lastMessage.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply}]}

def writing_agent(state: State):
    msgHistory = state["messages"][:-1][-20:]  # all messages except the last one, max 20.
    lastMessage = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
                        Provide clear, concise answers based on logic and evidence.
                        Do not address emotions or provide emotional support.
                        Be direct and straightforward in your responses."""
        }
          ] + msgHistory + [
        {
            "role": "user",
            "content": lastMessage.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply}]}

def router(state: State):
    messageType = state.get("messageType", "writer")
    if messageType == "reader":
        return {"next": "reading"}
    
    return {"next": "writing"}

#build graph architecture
graphBuilder = StateGraph(State)

#nodes
graphBuilder.add_node("classifier", classify_message)
graphBuilder.add_node("router", router)
graphBuilder.add_node("reading", reading_agent)
graphBuilder.add_node("writing", writing_agent)
#edges
graphBuilder.add_edge(START, "classifier")
graphBuilder.add_edge("classifier", "router")
#conditional edges
graphBuilder.add_conditional_edges(
    "router", lambda state: state.get("next"),
    {"reading": "reading", "writing": "writing"}
)
#end
graphBuilder.add_edge("reading", END)
graphBuilder.add_edge("writing", END)

graph = graphBuilder.compile()

def run_chatbot():
    state = {"messages": [], "messageType": None}

    while True:
        userInput = input("Message:")
        if userInput.lower() in ["exit", "quit"]:
            print("Logging out. Goodbye!")
            break

        state["messages"] = state.get("messages", []) +[
            {"role": "user", "content": userInput}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            lastResponse = state["messages"][-1]
            print(f"Assistant: {lastResponse.content}")

if __name__ == "__main__":
    run_chatbot()