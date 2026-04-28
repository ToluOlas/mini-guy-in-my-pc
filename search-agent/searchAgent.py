from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama.llms import OllamaLLM
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from prompts import (get_reddit_analysis_messages,
                     get_google_analysis_messages,
                     get_bing_analysis_messages,
                     get_reddit_url_analysis_messages,
                     get_synthesis_messages
                     )

load_dotenv()

llm = OllamaLLM(model="llama3.2")

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

class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(
        description="List of Reddit URLs that contain valuable information for answering the user's question")


def google_search(state: State):
    userQuestion = state.get("userQuestion", "")
    print(f"Searching Google for {userQuestion}")

    googleResults = serp_search(userQuestion, engine="google")
    print(googleResults)

    return {"googleResults": googleResults}

def bing_search(state: State):
    userQuestion = state.get("userQuestion", "")
    print(f"Searching Bing for {userQuestion}")

    bingResults = serp_search(userQuestion, engine="bing")
    print(bingResults)

    return {"googleResults": bingResults}

def reddit_search(state: State):
    userQuestion = state.get("userQuestion", "")
    print(f"Searching Reddit for {userQuestion}")

    redditResults = reddit_search_api(userQuestion)
    print(redditResults)

    return


def analyse_reddit_posts(state: State):
    userQuestion = state.get("userQuestion", "")
    redditResults = state.get("redditResults", "")

    if not redditResults:
        return {"selectedRedditUrls": []}

    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(userQuestion, redditResults)

    try:
        analysis = structured_llm.invoke(messages)
        selected_urls = analysis.selected_urls

        print("Selected URLs:")
        for i, url in enumerate(selected_urls, 1):
            print(f"   {i}. {url}")

    except Exception as e:
        print(e)
        selected_urls = []

    return {"selectedRedditUrls": selected_urls}

def retrieve_reddit_posts(state: State):
    print("Getting reddit post comments")

    selected_urls = state.get("selectedRedditUrls", [])

    if not selected_urls:
        return {"redditPostData": []}

    print(f"Processing {len(selected_urls)} Reddit URLs")

    reddit_post_data = reddit_post_retrieval(selected_urls)

    if reddit_post_data:
        print(f"Successfully got {len(reddit_post_data)} posts")
    else:
        print("Failed to get post data")
        reddit_post_data = []

    print(reddit_post_data)
    return {"redditPostData": reddit_post_data}


def analyse_google_results(state: State):
    print("Analyzing google search results")

    user_question = state.get("userQuestion", "")
    google_results = state.get("googleResults", "")

    messages = get_google_analysis_messages(user_question, google_results)
    reply = llm.invoke(messages)

    return {"google_analysis": reply.content}

def analyse_bing_results(state: State):
    print("Analyzing bing search results")

    user_question = state.get("userQuestion", "")
    bing_results = state.get("bingResults", "")

    messages = get_bing_analysis_messages(user_question, bing_results)
    reply = llm.invoke(messages)

    return {"bingAnalysis": reply.content}

def analyse_reddit_results(state: State):
    print("Analyzing reddit search results")

    user_question = state.get("userQuestion", "")
    reddit_results = state.get("redditResults", "")
    reddit_post_data = state.get("redditPostData", "")

    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)
    reply = llm.invoke(messages)

    return {"redditAnalysis": reply.content}


def synthesize_analyses(state: State):
    print("Combining all results together")

    user_question = state.get("userQuestion", "")
    google_analysis = state.get("googleAnalysis", "")
    bing_analysis = state.get("bingAnalysis", "")
    reddit_analysis = state.get("redditAnalysis", "")

    messages = get_synthesis_messages(
        user_question, google_analysis, bing_analysis, reddit_analysis
    )

    reply = llm.invoke(messages)
    final_answer = reply.content

    return {"finalAnswer": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}

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

