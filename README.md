# Olist E-commerce Chatbot

An AI-powered customer service chatbot built on the [Olist Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This project demonstrates an iterative architectural evolution from a basic LangChain implementation to a full Multi-Agent system built on LangGraph.

---

## Version History

| Version | Architecture | Key Features |
| --- | --- | --- |
| v1.0 LangChain | if-else Router + Chain | Baseline RAG + SQL Agent |
| v2.0 LangGraph | Graph + Conditional Edges | Multi-turn Memory, SQL retry, Token management, REST API |
| v3.0 Multi-Agent | LangGraph + Autonomous Agent Nodes | Each node upgraded to a self-managing Agent |

---

## Architecture

### v3.0 — Multi-Agent (Latest)

The latest version upgrades each graph node from fixed workflow logic to an **autonomous Agent**. Each agent independently decides how to use its tools, retry on errors, and format responses — no hardcoded logic required.

```
User Input
    ↓
router_node (LLM classifies intent)
    ↓
┌─────────────┬──────────────┬────────────────┬────────────┐
│  sql_agent  │  rag_agent   │ hybrid_agent   │  llm_agent │
│  SQL query  │ FAISS search │ SQL → RAG      │ General QA │
│  + retry    │  + summary   │ combined answer│            │
└─────────────┴──────────────┴────────────────┴────────────┘
    ↓
Final Answer (with MemorySaver across turns)
```

**4-path routing system:**

- **SQL path** — structured queries: order counts, sales stats, product rankings
- **RAG path** — semantic search over customer reviews using FAISS
- **Hybrid path** — SQL result passed as context into RAG search (e.g. "What do customers say about top-rated orders?")
- **LLM path** — general conversation and out-of-scope questions

**Key upgrades in v3.0:**

- `sql_agent`: autonomously generates SQL, executes via tool, retries on failure using error feedback in prompt — no hardcoded retry counter
- `rag_agent`: retrieves and summarizes reviews independently
- `hybrid_agent`: runs SQL first, passes structured results as context into RAG retrieval
- `llm_agent`: lightweight fallback with no tools, handles general questions gracefully

### v2.0 — LangGraph

![Architecture](architecture.png)

Fixed-logic graph with conditional edges. SQL retry and error handling implemented via explicit `result_check` and `error_count` nodes.

---

## Key Features

- **Multi-turn Memory** — conversation history preserved across turns using LangGraph `MemorySaver` + `thread_id`
- **Autonomous Agent Nodes** — each agent manages its own tool usage, retry logic, and output formatting
- **Hybrid Retrieval** — SQL structured results combined with FAISS semantic search for richer answers
- **SQL Error Retry** — agent retries up to 3 times with error feedback when SQL execution fails
- **Token Optimization** — SQL results truncated and RAG results limited to prevent context overflow
- **Graceful Fallback** — LLM agent handles unsupported questions with friendly responses
- **REST API** — FastAPI backend exposing the chatbot as an HTTP endpoint with thread-based memory
- **Streamlit UI** — browser-based chat interface for live demonstration

---

## Tech Stack

| Category | Tools |
| --- | --- |
| Agent Framework | LangGraph, LangChain |
| Language Model | OpenAI GPT-4o-mini |
| Vector Store | FAISS |
| Database | SQLite + LangChain SQLDatabase |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Development | Python, Jupyter Notebook, Cursor |

---

## Project Structure

```
olist-chatbot/
├── agent.py              # v2.0 Streamlit chatbot (LangGraph + Memory)
├── agent.ipynb           # v2.0 Jupyter notebook version
├── multi_agent.ipynb     # v3.0 Multi-Agent version (latest)
├── api_agent.py          # FastAPI REST API version
├── app.py                # v1.0 Original LangChain version (baseline)
├── faiss_db/             # FAISS vector index
├── olist.db              # SQLite database
├── architecture.png      # v2.0 graph architecture diagram
├── .env                  # API keys (not included)
└── README.md
```

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/linhuifan1-gif/olist-chatbot.git
cd olist-chatbot
```

### 2. Install dependencies

```bash
pip install langchain langgraph langchain-openai langchain-community faiss-cpu python-dotenv streamlit fastapi uvicorn
```

### 3. Set up environment

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=your_base_url_here
```

### 4. Run

```bash
# Option 1: Streamlit UI (v2.0 LangGraph)
streamlit run agent.py

# Option 2: FastAPI REST API
uvicorn api_agent:app --reload
# Visit http://127.0.0.1:8000/docs to test

# Option 3: Multi-Agent (v3.0) — run in Jupyter
# Open multi_agent.ipynb
```

---

## Architecture Comparison

| Feature | v1.0 LangChain | v2.0 LangGraph | v3.0 Multi-Agent |
| --- | --- | --- | --- |
| Multi-turn Memory | ❌ | ✅ | ✅ |
| SQL Error Retry | ❌ | ✅ hardcoded | ✅ autonomous |
| Token Management | ❌ | ✅ | ✅ |
| Graph Visualization | ❌ | ✅ | ✅ |
| REST API | ❌ | ✅ | ✅ |
| Routing | if-else | Conditional edges | LLM + Conditional edges |
| Node Logic | Fixed functions | Fixed functions | Autonomous Agents |
| Retry Logic | None | Hardcoded counter | LLM self-managed |

---

## Dataset

[Olist Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100K+ orders, 9 relational tables, customer reviews in Portuguese.

---

## About

Built as a capstone AI application project demonstrating the architectural evolution from rule-based workflows to autonomous multi-agent systems, using real-world e-commerce data.
