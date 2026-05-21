# mini-guy-in-my-pc

A locally-hosted multi-agent AI system built with LangGraph. Different sub-agents are designed to handle different types of tasks like routing, conversation, and web research. All running on your own machine with no cloud LLM costs.

## Table of Contents

- [About The Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Main Program](#main-program-mainpy)
  - [Local Search Agent](#local-search-agent)
- [Roadmap](#roadmap)

---

## About The Project

`mini-guy-in-my-pc` is an experiment in building a fully local multi-agent AI pipeline. The goal is to allocate different types of tasks to specialised sub-agents, with all LLM inference handled locally via [Ollama](https://ollama.com) — no API keys, no usage costs.

Current agents:
- **Reader agent** — responds with empathy and emotional support (therapist-style)
- **Writer agent** — responds with facts and logical analysis
- **Searcher agent** — performs a live web search and returns a cited, synthesised answer

### Built With

* [LangGraph](https://github.com/langchain-ai/langgraph)
* [LangChain](https://github.com/langchain-ai/langchain)
* [Ollama](https://ollama.com)
* [duckduckgo-search](https://github.com/deedy5/duckduckgo_search)
* [Trafilatura](https://github.com/adbar/trafilatura)
* Python 3.11+

---

## Getting Started

### Prerequisites

- [Ollama](https://ollama.com) installed and running locally
- Python 3.11+
- The models you intend to use pulled via Ollama, e.g.:
  ```sh
  ollama pull llama3.2
  ollama pull deepseek-r1:8b
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ToluOlas/mini-guy-in-my-pc.git
   cd mini-guy-in-my-pc
   ```

2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

3. Install additional dependencies for the search agent
   ```sh
   pip install duckduckgo-search trafilatura
   ```

---

## Project Structure

```
mini-guy-in-my-pc/
├── main.py                          # Main entry point — classifies and routes user messages
├── web_search.py                    # Search pipeline called by main when a search is required
├── local-search-agent/
│   ├── searchAgent.py               # Standalone search agent (for development/testing)
│   └── query-to-search-test.py      # Test script for raw DuckDuckGo + Trafilatura output
└── search-agent/                    # Earlier search agent (requires Bright Data API - deprecated)
```

---

## Usage

### Main Program (`main.py`)

The main entry point for the system. Each message is classified and routed to the appropriate sub-agent:

```
START → classifier → router → reading / writing / searching → END
```

| Agent | Trigger | Behaviour |
|---|---|---|
| `reading` | Emotional support, feelings, personal problems | Responds as a compassionate therapist |
| `writing` | Logical analysis, practical solutions | Responds with facts and direct reasoning |
| `searching` | Questions requiring facts, research, or an internet search | Runs a live web search and returns a cited answer |

```sh
python main.py
```

```
Message: I've been feeling really overwhelmed lately
Assistant: It sounds like you're carrying a lot right now...

Message: What is the boiling point of nitrogen?
Assistant: The boiling point of nitrogen is -195.79°C...

Message: What is the current state of the UK job market?
Assistant: Based on recent sources, the UK job market is showing signs of...

Sources:
- [The Current State of the UK Labour Market](https://www.jobboardfinder.com/...)
```

Type `exit` or `quit` to stop.

---

### Web Search Module (`web_search.py`)

Called automatically by `main.py` when a search task is detected. Can also be imported directly:

```python
from web_search import search

answer = search("What are the latest developments in fusion energy?")
print(answer)
```

The pipeline runs four nodes internally:

```
extract_query → run_search → extract_content → synthesize_answer
```

| Node | Description |
|---|---|
| `extract_query` | LLM distils the user's message into a clean search query |
| `run_search` | Searches DuckDuckGo, returns the top 5 results |
| `extract_content` | Fetches and cleans full page content via Trafilatura |
| `synthesize_answer` | LLM writes a cited answer from the extracted content |

---

### Local Search Agent (`local-search-agent/`)

Standalone versions of the search pipeline used during development and testing.

#### `searchAgent.py`
A self-contained version of the search pipeline with its own CLI loop. Used for testing the search pipeline independently of the main program.

```sh
python local-search-agent/searchAgent.py
```

#### `query-to-search-test.py`
A minimal test script that prints raw DuckDuckGo results and Trafilatura-extracted page content with no LLM involvement. Useful for inspecting what the search agent receives before synthesis.

```sh
python local-search-agent/query-to-search-test.py
```

---

## Roadmap

- [x] Chatbot agent with routing based on user query
- [x] Local search agent with DuckDuckGo
- [x] Full page extraction with Trafilatura
- [x] Integrate search agent into main routing graph
- [ ] Add more specialised sub-agents (e.g. file reader, summariser)
- [ ] Persistent memory across sessions
- [ ] Web UI
- [ ] Expand on existing main sub-agents
