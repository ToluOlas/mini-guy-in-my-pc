from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama.llms import OllamaLLM
from typing_extensions import TypedDict
from ddgs import DDGS
import trafilatura
import re

llm = OllamaLLM(model="llama3.2")
queryLLM = OllamaLLM(model="deepseek-r1:8b")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    query: str | None
    searchResults: list | None
    answer: str | None


def extract_query(state: State):
    msgHistory = state["messages"][:-1][-20:]  # all messages except the last one, max 20.
    lastMessage = state["messages"][-1]

    result = queryLLM.invoke([
                                 {
                                     "role": "system",
                                     "content": """Turn the user's message into a Google/Bing search query.
            Output ONLY the query, NOTHING ELSE.

            A good query is 2–6 distinctive content words that would plausibly appear on a page answering the question. Build it like this:

            1. Identify the subject. If the user uses a pronoun ("it", "that", "the sequel") or relies on conversation context, substitute the actual subject from earlier in the conversation. 
            Use the name the relevant community most commonly uses (full name, abbreviation, or code name — whichever results in more search traction).

            2. Identify the question. Phrase it in the vocabulary of the likely answer source, not the user's words. 
            EXAMPLE: "How do I stop my React component re-rendering" becomes "React prevent re-render", because that's the language docs and Stack Overflow use.

            3. Do not use quotes, site:, or minus operators unless the user specifically needs exact-phrase matching or domain restriction.

            Examples:

            Conversation about Final Fantasy XIV.
            User: "what's the best DPS class right now?"
            Query: FFXIV best DPS 2026

            Conversation about a pandas SettingWithCopyWarning the user just hit.
            User: "how do I fix it?"
            Query: pandas SettingWithCopyWarning fix

            User: "I want to know more about how the Krebs cycle produces ATP"
            Query: Krebs cycle ATP production

            User: "who's the current prime minister of Japan"
            Query: prime minister of Japan

            User: "is Obsidian still being actively developed?"
            Query: Obsidian note-taking app development status 2026
            """
                                 }
                             ] + msgHistory + [
                                 {
                                     "role": "user",
                                     "content": lastMessage.content
                                 }
                             ])

    return {"query": result.strip()}


def run_search(state: State):
    query = state["query"]

    print(f"--- Searching " + query + " ---")

    raw = DDGS().text(query, max_results=5)
    results = [
        {"title": r["title"], "url": r["href"], "description": r["body"]}
        for r in (raw or [])
    ]

    return {"searchResults": results}


def extract_content(state: State):
    results = state["searchResults"]

    for r in (results or []):
        html = trafilatura.fetch_url(r["url"])
        if html:
            text = trafilatura.extract(html, include_comments=False, include_tables=False)
            if text:
                r["description"] = text

    return {"searchResults": results}


def synthesize_answer(state: State):
    query = state["query"]
    results = state["searchResults"]

    if not results:
        return {"answer": "No search results found.",
                "messages": [{"role": "assistant", "content": "No search results found."}]}

    results_text = "\n\n".join([
        f"Title: {r['title']}\nURL: {r['url']}\nContent:\n{r['description']}"
        for r in results
    ])

    reply = llm.invoke([
        {
            "role": "system",
            "content": """You are a research assistant that answers the users questions based on search results. Using the search results provided, give a clear and concise answer to the user's query.
            From the search results, focus on the text that is related to the query. 
            If different sources can agree on a point, consider it more reliable.
            If different sources do not agree on a point, mention BOTH sides and make sure to reference which source each point came from.
            Cite the sources by referencing their titles in brackets where relevant. List their URLs at the end of your response.
            Address the user when answering."""
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nSearch Results:\n{results_text}"
        }
    ])

    return {
        "answer": reply,
        "messages": [{"role": "assistant", "content": reply}]
    }


graphBuilder = StateGraph(State)

graphBuilder.add_node("extract_query", extract_query)
graphBuilder.add_node("run_search", run_search)
graphBuilder.add_node("extract_content", extract_content)
graphBuilder.add_node("synthesize_answer", synthesize_answer)

graphBuilder.add_edge(START, "extract_query")
graphBuilder.add_edge("extract_query", "run_search")
graphBuilder.add_edge("run_search", "extract_content")
graphBuilder.add_edge("extract_content", "synthesize_answer")
graphBuilder.add_edge("synthesize_answer", END)

graph = graphBuilder.compile()


def run_search_agent():
    state = {"messages": [], "query": None, "searchResults": None, "answer": None}

    while True:
        userInput = input("Search: ")
        if userInput.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": userInput}
        ]

        state = graph.invoke(state)

        if state.get("answer"):
            print(f"\nAssistant: {state['answer']}\n")


if __name__ == "__main__":
    run_search_agent()
