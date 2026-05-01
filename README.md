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
  - [Chatbot Agent](#chatbot-agent-mainpy)
  - [Local Search Agent](#local-search-agent)
- [Roadmap](#roadmap)

---

## About The Project

`mini-guy-in-my-pc` is an experiment in building a fully local multi-agent AI pipeline. The goal is to allocate different types of tasks to specialised sub-agents, with all LLM inference handled locally via [Ollama](https://ollama.com) — no API keys, no usage costs.

Current agents:
- **Chatbot agent** — classifies user messages as emotional or logical and routes them to the appropriate response agent. 
- **Local search agent** — takes a user query, searches the web via DuckDuckGo, extracts full page content with Trafilatura, and synthesises a cited answer

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

3. Install additional dependencies for the local search agent
   ```sh
   pip install duckduckgo-search trafilatura
   ```

---

## Project Structure (For Now)

```
mini-guy-in-my-pc/
├── main.py                          # Chatbot agent (emotional/logical routing)
├── local-search-agent/
│   ├── searchAgent.py               # Full web search + synthesis pipeline
│   └── query-to-search-test.py      # Test script for raw DuckDuckGo + Trafilatura output
└── search-agent/                    # Earlier search agent (requires Bright Data API - deprecated)
```

---

## Usage

### Chatbot Agent (`main.py`)

A dual-agent chatbot that classifies each message and routes it to the appropriate agent:
- **Reader agent** — responds with empathy and emotional support (therapist-style)
- **Writer agent** — responds with facts and logical analysis

```sh
python main.py
```

```
Message: I've been feeling really overwhelmed lately
Assistant: It sounds like you're carrying a lot right now...

Message: What causes inflation?
Assistant: Inflation is caused by...
```

In future, this will be changed so instead of responses based on tone, it will respond based on what type of task a question requires.

---

### Local Search Agent

#### `local-search-agent/searchAgent.py`

A four-node LangGraph pipeline that answers questions using live web search:

```
extract_query → run_search → extract_content → synthesize_answer
```

| Node | Description                                              |
|---|----------------------------------------------------------|
| `extract_query` | LLM distils the user's message into a clean search query |
| `run_search` | Searches DuckDuckGo, returns the top X results           |
| `extract_content` | Fetches and cleans full page content via Trafilatura     |
| `synthesize_answer` | LLM writes a cited answer from the extracted content     |

```sh
python local-search-agent/searchAgent.py
```

```
Search: What is the current state of the UK job market?
Assistant: Based on recent sources, the UK job market is showing signs of...

Sources:
- [The Current State of the UK Labour Market](https://www.jobboardfinder.com/...)
```

Type `exit` or `quit` to stop.

---

#### `local-search-agent/query-to-search-test.py`

A lightweight test script for inspecting raw DuckDuckGo search results and Trafilatura page extraction without any LLM involvement. Useful for debugging what the search agent actually receives before synthesis.

```sh
python local-search-agent/query-to-search-test.py
```

---

## Roadmap

- [x] Chatbot agent with routing based on user query
- [x] Local search agent with DuckDuckGo
- [x] Full page extraction with Trafilatura
- [ ] Integrate search agent into main routing graph
- [ ] Add more specialised sub-agents (e.g. file reader, summariser)
- [ ] Persistent memory across sessions
- [ ] Web UI
- [ ] Expand on existing main sub-agents 
