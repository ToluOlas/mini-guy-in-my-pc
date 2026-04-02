from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model("gpt-4o")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    userQuestion: str | None
    googleResults: str | None
    bingResults: str | None
    redditResults: str | None
    selectedRedditUrls: list[str] | None
    redditPostData: list | None
    googleAnalysis: str | None
    bingAnalysis: str | None
    redditAnalysis: str | None
    finalAnswer: str | None

def google_search(state: State):
    return

def bing_search(state: State):
    return

def reddit_search(state: State):
    return


def analyse_reddit_posts(state: State):
    return

def retrieve_reddit_posts(state: State):
    return



def analyse_google_results(state: State):
    return

def analyse_bing_results(state: State):
    return

def analyse_reddit_results(state: State):
    return


def synthesize_analyses(state: State):
    return

#build graph architecture
graphBuilder = StateGraph(State)

graphBuilder.add_node("google_search", google_search)
graphBuilder.add_node("bing_search", bing_search)
graphBuilder.add_node("reddit_search", reddit_search)
graphBuilder.add_node("analyse_reddit_posts", analyse_reddit_posts)
graphBuilder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)
graphBuilder.add_node("analyse_google_results", analyse_google_results)
graphBuilder.add_node("analyse_bing_results", analyse_bing_results)
graphBuilder.add_node("analyse_reddit_results", analyse_reddit_results)
graphBuilder.add_node("synthesize_analyses", synthesize_analyses)

#1. all 3 searched upon receiving prompt
graphBuilder.add_edge(START, "google_search")
graphBuilder.add_edge(START, "bing_search")
graphBuilder.add_edge(START, "reddit_search")

#2. wait for reddit post analysis (the longest process)
graphBuilder.add_edge("google_search", "analyse_reddit_posts")
graphBuilder.add_edge("bing_search", "analyse_reddit_posts")
graphBuilder.add_edge("reddit_search", "analyse_reddit_posts")
graphBuilder.add_edge("analyse_reddit_posts", "retrieve_reddit_posts")

#3. Analyse all results
graphBuilder.add_edge("retrieve_reddit_posts", "analyse_google_results")
graphBuilder.add_edge("retrieve_reddit_posts", "analyse_bing_results")
graphBuilder.add_edge("retrieve_reddit_posts", "analyse_reddit_results")

#4. Results are combined into one analysis
graphBuilder.add_edge("analyse_google_results", "synthesize_analyses")
graphBuilder.add_edge("analyse_bing_results", "synthesize_analyses")
graphBuilder.add_edge("analyse_reddit_results", "synthesize_analyses")

graphBuilder.add_edge("synthesize_analyses", END)

graph = graphBuilder.compile()

def run_chatbot():
    print("Multi-Source Research Agent")
    print("Type 'exit' to quit\n'")

    while True:
        userInput = input("Ask me anything: ")
        if userInput.lower() == "exit":
            print("I'm outta here")
            break

        state = {
            "messages": [{"role": "user", "content": userInput}],
            "userQuestion": userInput,
            "googleResults": None,
            "bingResults": None,
            "redditResults": None,
            "selectedRedditUrls": None,
            "redditPostData": None,
            "googleAnalysis": None,
            "bingAnalysis": None,
            "redditAnalysis": None,
            "finalAnswer": None,
        }

        print("\n Starting parallel resarch process...")
        print("Launching Google, Bing and Reddit searches...\n")
        finalState = graph.invoke(state)

        if finalState.get("finalAnswer"):
            print(f"\nFinal Answer:\n{finalState.get('finalAnswer')}\n")

        print("-" * 50)

if __name__ == "__main__":
    run_chatbot()

